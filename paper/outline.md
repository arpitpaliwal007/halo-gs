# HaloGS: Physically-Grounded Glow Decomposition for 3D Gaussian Splatting — 4-page outline

Target: 4 pages + references (CVPR/ICCV workshop, ICIP/ICASSP style). One contribution, one figure-1.

## Problem statement

Given posed multi-view photographs of a static night scene containing light
sources, reconstruct a glow-free scene and a separate camera-dependent glow
layer, without glow-free supervision, such that novel views render correctly
*with* glow and the scene can be rendered *without* it.

## Why it is open (positioning, verified Oct 2026)

| Work | Glow model | Supervision / priors | Gap we fill |
|---|---|---|---|
| GlowGS (CVPR Findings 2026) | appearance: glow regions have no structure | Pika/PromeAI generated videos + DINO/CLIP feature bank; A5000 | no physical model, no scene/glow separation, not reproducible without video generation; authors list "physics-based constraints" as future work |
| Lens Flare Removal & Reconstruction (Sep 2026), FlareGS | lens ghosts / windshield flare | diffusion remover trained on paired synthetic flare; LiDAR | different physics (internal reflections), not scatter halos |
| DehazeGS, SeaSplat, Tri-DehazeGS | medium along the ray (airlight / backscatter) | depth priors | world-space medium, not point-source scatter around emitters |
| 2D nighttime dehazing (Jin et al. 2023; IJCAI 2023 glow removal) | APSF of Narasimhan & Nayar | single image | the physics, but no 3D model, no multi-view consistency |
| NOVA-GS, Luminance-GS, LITA-GS, FLED-GS | low light, noise, blur | — | no glow term |

Claim: the first explicit, physically named glow layer inside a radiance field,
separable from multi-view evidence alone because glow follows the camera and
the scene follows parallax.

## Method (half a page + one diagram)

* Emission attribute per Gaussian, rendered with the same geometry → E_v.
* Kernel k = (1−w)·APSF(q,T) + w·GSF(γ): four scalars, differentiable; T ≥ 0.8
  (validity of the Legendre expansion) — state this.
* observed = clip(S + k ⊛ E); losses: L1+DSSIM on the composite, L1 sparsity on
  emission; optional per-view exposure. Two rasteriser passes + one FFT.
* What it is not: not HDR, not denoising, not flare ghosts; emission magnitude
  carries energy, kernel carries shape only.

## Experiments (page 3)

1. **Real scenes** — Bilarf-night {building, pondbike, strat} and RawNeRF-Glow
   {candlefiat, gardenlights, notchbush, parkstatue, stove, streetcorner}, sRGB,
   given poses, every-8th hold-out (and the GlowGS 6-view protocol for the
   comparison table). Metrics: composite vs. observed — full / in-glow /
   out-of-glow PSNR, SSIM, LPIPS (crop). Baselines: 3DGS, masked-3DGS (same
   code), Mip-Splatting, LLNeRF, Aleth-NeRF; GlowGS as reported.
2. **Synthetic control** — APSF halo with known (q, T, γ, w) composited onto a
   clean posed scene (LOM `buu` normal-light, Mip-NeRF360 garden). Report
   scene-vs-clean PSNR in glow regions, recovered (q, T, γ) vs. truth, sweep
   T ∈ {1, 1.5, 3}, intensity ∈ {2, 4}, emitters ∈ {1, 3, 6}.
3. **Qualitative** — composite / scene / glow triplets at novel views; floaters
   in vanilla vs. ours at the same view.

## Ablations (page 4)

kernel both / apsf-only / lens-only / radial (non-physical) / frozen kernel;
λ_emission = 0; per-view exposure on/off; glow warm-up on/off. Three seeds for
the headline rows; report scene-level means.

## Identifiability and limitations (must be in the paper)

* PSF width trades off against source extent and radiance; the synthetic sweep
  shows where recovery is unique and where it is not.
* Glow that is clipped everywhere, or a lamp seen only through its halo, cannot
  be separated — report the fraction of such pixels.
* Camera auto-exposure / white balance are nuisance variables → per-view gain.
* T < 0.8 (thin media) is outside the APSF series' validity; the lens term
  covers tight halos.

## Pre-registered go / no-go

* Synthetic: scene-vs-clean PSNR in glow regions ≥ +3 dB over masked-3DGS and
  recovered T within 20 % of truth for at least two of three settings.
* Real: in-glow PSNR ≥ masked-3DGS and non-glow PSNR within 0.2 dB of vanilla
  on ≥ 5 of 9 scenes, three seeds. Otherwise the paper becomes an analysis of
  why multi-view glow separation fails, not a method paper.
