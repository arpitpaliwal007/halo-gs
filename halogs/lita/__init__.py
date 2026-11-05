"""LITA-GS backend: HaloGS emission + scatter kernel on top of the official
LITA-GS (CVPR 2025) reflectance / illumination / denoiser / tone-mapper model.

    low_v    = tonemapper( illu_v * denoise(scene_v) )  +  k * E_v
    bright_v = scene_v                                  +  s_v * (k * E_v)

with s_v = theta / mean(low_v) the per-view brightening LITA-GS uses to build
its pseudo bright target.  ``scene_v`` stays glow-free, ``E_v`` is the rendered
per-Gaussian emission, ``k`` the APSF + lens-glare kernel.
"""
