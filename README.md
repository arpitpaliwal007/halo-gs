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
train_glow.py  vanilla-3DGS backbone: ours (--mode halo) and baselines (--mode vanilla | masked)
train_lita_halo.py  LITA-GS backbone: ours (--mode halo) and plain LITA-GS (--mode lita)
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

## LITA-GS backbone (recommended for dark scenes with lamps)

The official [LITA-GS](https://github.com/LowLevelAI/LITA-GS) (CVPR 2025, built on
Gaussian-DK) is a much stronger illumination model than plain 3DGS: 3-channel
per-Gaussian illumination, a learned tone-mapper, a denoiser, Marigold depth
and CIConv structure priors. `train_lita_halo.py` plugs HaloGS's emission +
scatter kernel into it with no CUDA rebuild (the emission is rasterised in a
second pass through LITA-GS's own 11-channel rasteriser):

```
low_v    = tonemapper( illu_v * denoise(scene_v) )  +  k * E_v
bright_v = scene_v                                  +  s_v * (k * E_v)      s_v = theta / mean(low_v)
```

`scene_v` stays glow-free; `--mode lita` runs plain LITA-GS through the same
script, so the baseline shares split, schedule and initialisation.

```bash
bash setup_lita.sh ../LITA-GS && export LITA_ROOT=$PWD/../LITA-GS
python data/prepare_lita_scene.py --scene data/scenes/bilarf/pondbike --out data/lita/pondbike --prior
bash scripts/run_scene_lita.sh data/lita/pondbike out_lita/pondbike 20000 --no_depth --split every8
# official LOM protocol (eval_index [0,4,10], theta = 0.49, 20k iterations), needs their LOM zip with depth_maps/ + W_0.8/
python train_lita_halo.py -s data/LOM_low_buu_colmap -m out_lita/buu/halo --mode halo --split lita --configs $LITA_ROOT/arguments/LOM/buu_low.py
```

Three held-out numbers are logged per view: `lita_*` (their protocol: glow-free
denoised bright render vs `images/`), `bright_*` (bright render + scaled glow
vs `images/`, the fair one when the reference contains halos) and `low_*`
(tone-mapped composite vs the observed dark photo, with glow-region split).
Note LITA-GS's own script selects the checkpoint with the best *test* PSNR
every 100 iterations; this script does not (fixed iteration count), so numbers
will be slightly lower than theirs and comparable across methods.

Backends cannot be mixed in one process (both define `scene`, `gaussian_renderer`,
`arguments`, `utils`); `scripts/run_tests.sh` runs each test group separately.

## LitaLite backbone (corrected) and LitaLite + HaloGS

LitaLite is the earlier reference-free low-light model (the "Direct Method"): per-Gaussian
reflectance (SH colour) and a scalar illumination, a bright render `R = G(c)` pulled toward a
linearly brightened input, and a dark render `G(c * L)` tied to the captured photo.
It runs on the vanilla 3DGS backend (`setup_gs.sh`), so it is much faster than LITA-GS.

Corrections relative to the original `litalite.py` (see `implementation_audit_2026-10-02.md`):

| Issue in the original | Fix here |
|---|---|
| illumination in a side tensor, truncated / mean-padded by *count* on prune and densify | per-Gaussian optimiser group; follows identity through prune / clone / split, keeps Adam moments (`tests/test_illum_lifecycle.py`) |
| densification statistics from the bright branch only | from both branches (`--dense_from both`, default; `bright` = original) |
| target clipped per channel, which changes hue where it saturates | hue-safe target: saturating pixels scaled as a whole (`--legacy_clip` = original) |
| per-image gain only | `--target_gain image` (original) or `scene` (one gain for all views) |
| evaluation paired renders and references by directory order | references matched by image name (`--clean_gt_dir`), never by order |

`--mode litalite_halo` adds HaloGS: the dark render becomes `clip(G(c * L) + k * E)`, so lamps and
their halos are explained by emission + scattering kernel instead of being baked into reflectance.

```bash
# LOM buu: dark images in images/, normal-light references in gt/
python train_glow.py -s datasets/buu -m out/buu/litalite --mode litalite \
    --iterations 15000 --save_iterations 7000 15000 \
    --densify_until_iter 5000 --densification_interval 100 --opacity_reset_interval 3000 \
    --clean_gt_dir datasets/buu/gt
python render_glow.py -m out/buu/litalite --iteration 15000 --skip_train
python eval_glow.py   -m out/buu/litalite --iteration 15000 --clean_gt_dir datasets/buu/gt
# same with glow (night scenes with lamps):  --mode litalite_halo
# original behaviour for an ablation:         --dense_from bright --legacy_clip
```

Three cheap upgrades toward LITA-GS (each a flag, so each can be ablated):

| Flag | What it adds |
|---|---|
| `--illum_channels 3` | RGB illumination per Gaussian, so coloured light is not pushed into reflectance |
| `--tone_curve` | learned per-view response `g_v * x^gamma_v` on the dark branch (identity at start; test views use the mean) |
| `--denoise_target bilateral` | edge-preserving filter on the brightened target, where the gain amplifies sensor noise |

`test_curve.jsonl` then holds both the dark-composite score against the observed photo
(`psnr`) and the bright output against the normal-light reference (`clean_psnr`).

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
