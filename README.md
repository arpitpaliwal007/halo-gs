# HaloGS — physically-grounded glow decomposition for 3D Gaussian Splatting

Night-time photographs contain halos around lamps, headlights and windows. Vanilla
3DGS has no primitive that moves with the *camera* rather than with the *scene*,
so it explains the glow with floaters and view-dependent colour and breaks at
novel views. HaloGS adds one physically named layer:

```
observed_v  =  clip( S_v  +  k(q, T, γ, w) ⊛ E_v )
```

* **S_v** – glow-free scene render (ordinary SH Gaussians)
* **E_v** – rendered *emission* map: every Gaussian carries an emission attribute,
  rendered with the same geometry / opacity (occluded emitters do not glow)
* **k** – radially symmetric kernel: atmospheric point-spread function of
  Narasimhan & Nayar (forward-scattering *q*, optical thickness *T*) mixed with a
  lens glare-spread function (width *γ* px). Four scalars, fitted jointly.

The scene is parallax-consistent across views; the glow is a convolution of the
*projected* emitters. That asymmetry is what lets multi-view training separate
them with **no glow-free supervision, no diffusion model, no VFM** — the whole
thing is two rasteriser passes and one FFT per iteration, fine on a T4.

Closest prior work: GlowGS (CVPR Findings 2026) treats glow as appearance and
learns it from generated videos; the 2D APSF literature has the physics but no
3D model; lens-flare 3DGS papers model lens *ghosts*, not scatter halos. See
`paper/outline.md` for the positioning and `data/SCENES.md` for the real scenes.

## Layout

```
halogs/        apsf.py (kernels) · compositing.py (FFT conv) · gaussians.py (emission lifecycle)
               halo.py (two-pass render) · losses.py · masks.py · metrics.py · splits.py
train_glow.py  ours (--mode halo) and baselines (--mode vanilla | masked), same split/schedule
render_glow.py composite / scene / glow / emission layers + evaluation manifest
eval_glow.py   full, in-glow and out-of-glow PSNR / SSIM / LPIPS, paired by image name
data/          download + prepare real glow scenes (Bilarf, RawNeRF, LLNeRF); synthetic control
scripts/       run_scene.sh (3 methods) · ablations.sh
notebooks/     HaloGS_colab.ipynb — end-to-end on a Colab T4
tests/         CPU tests (kernel, FFT, masks/metrics/splits, emission lifecycle, pipeline smoke)
paper/         4-page outline + experiment grid
```

## Quick start

```bash
git clone https://github.com/arpitpaliwal007/halo-gs && cd halo-gs
pip install -r requirements.txt
bash setup_gs.sh ../gaussian-splatting official      # clones + builds the CUDA rasteriser; sets GS_ROOT
export GS_ROOT=$PWD/../gaussian-splatting

bash data/download_bilarf.sh data/raw                # real night scenes (HF, 7.8 GB)
python data/prepare_scene.py --scene data/raw/bilarf/pondbike --out data/scenes/bilarf/pondbike --max_side 1600

bash scripts/run_scene.sh data/scenes/bilarf/pondbike out/pondbike 30000 --data_device cpu --per_view_exposure
cat out/pondbike/results.md
```

Or open `notebooks/HaloGS_colab.ipynb` in Colab.

### Training options that matter

| flag | default | meaning |
|---|---|---|
| `--mode halo\|vanilla\|masked` | halo | ours / plain 3DGS / 3DGS with glow pixels dropped from the loss |
| `--kernel both\|apsf\|lens\|radial` | both | kernel family (ablation) |
| `--kernel_radius` | 0.1·max(H,W) | support in px |
| `--lambda_emission` | 0.01 | L1 sparsity on emission (few emitters) |
| `--glow_from_iter` | 500 | plain 3DGS warm-up before the glow branch is enabled |
| `--per_view_exposure` | off | learn a log-gain per training view (Bilarf varies exposure per shot) |
| `--split every8\|glowgs` | every8 | LLFF hold-out, or the GlowGS 6-view few-shot protocol |
| `--mask_thresh/--mask_radius` | 0.92 / 48 | glow-region definition (from the *observed* image, method-independent) |

All official 3DGS flags (`-r`, `--iterations`, `--densify_grad_threshold`, …) are accepted.

### Outputs

```
out/<scene>/<method>/
  split.json          train / test image names (evaluation pairs by NAME, never by directory order)
  config.json log.csv test_curve.jsonl
  point_cloud/iteration_N/{point_cloud.ply, emission.pt, halo.pt}
  renders/iteration_N/test/{render,scene,glow,emission,gt}/<name>.png + manifest.json
  renders/iteration_N/test/{metrics.json, per_view.csv}
```

## Evaluation protocol

Real scenes have no glow-free reference, so held-out views are scored
**composite vs. observed** (full image, in-glow region, out-of-glow region;
glow regions come from thresholding + dilating light sources in the reference).
The scene/glow separation itself is validated on `data/make_synthetic_glow.py`
scenes, which composite an APSF halo with known (q, T, γ) onto a clean posed
capture, so the recovered glow-free render and kernel parameters can be scored
exactly (`eval_glow.py --clean_gt_dir`).

Baselines inside this repo: vanilla 3DGS and masked 3DGS (identical split,
schedule, initialisation). External: Mip-Splatting, LLNeRF, Aleth-NeRF and the
GlowGS numbers on their split (its code and generated-video step are not
reproducible here; say so in the paper).

## Tests

```bash
pytest -q tests                          # kernel / FFT / masks / metrics / splits (no GPU, no 3DGS needed)
GS_ROOT=../gaussian-splatting pytest -q  # + emission lifecycle against the real GaussianModel, + CPU pipeline smoke
```

The lifecycle tests check that emission follows Gaussian *identity* through
prune / clone / split (children inherit, Adam moments survive, equal-count
replacement is handled) on both the current official repo and the 2023 fork.

## Status

Code complete and CPU-tested; **no GPU runs yet** — all numbers must come from
the Colab runs. Known gaps: NightGlow (GlowGS) is not public; Bilarf archive
folder names should be checked against `data/SCENES.md` after download.

## License

HaloGS code: MIT. It depends on the Inria/MPII gaussian-splatting code, which
is released for **non-commercial research use only** — see its LICENSE.md.
