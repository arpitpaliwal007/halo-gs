"""Two-pass rendering on the LITA-GS backend.

Pass 1: LITA-GS's own ``render`` (RGB + depth + prior + noise + illu channels,
        then denoiser and tone-mapper -> ``render_low``).
Pass 2: the same rasteriser with the emission as the colour channels (the
        extra feature channels are zero-filled so the compiled NUM_CHANNELS=11
        layout is unchanged -> no CUDA rebuild).

Composite (HaloGS physics):
    low_composite    = clip( render_low     + k * E )
    bright_composite = clip( render         + s * (k * E) )   # s = theta / mean(low)
    denoised_comp    = clip( denoised[-1]   + s * (k * E) )
"""
from __future__ import annotations

import math

import torch

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer  # type: ignore
from gaussian_renderer import render as lita_render  # type: ignore

from ..compositing import fft_conv2d


def render_emission(viewpoint_camera, pc, pipe, bg_color, depth_threshold=None, iteration=None) -> torch.Tensor:
    """Rasterise the per-Gaussian emission (N,3) with LITA-GS's rasteriser. Returns (3,H,W)."""
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=False, device=pc.get_xyz.device)
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height), image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx, tanfovy=tanfovy, bg=bg_color, scale_modifier=1.0,
        depth_threshold=depth_threshold, iteration=iteration,
        viewmatrix=viewpoint_camera.world_view_transform, projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree, campos=viewpoint_camera.camera_center, prefiltered=False, debug=pipe.debug)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    scales = rotations = cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(1.0)
    else:
        scales, rotations = pc.get_scaling, pc.get_rotation
    n_extra = pc.get_new_features.shape[1]
    emission = pc.get_emission
    colors = torch.cat([emission, torch.zeros(emission.shape[0], n_extra, device=emission.device, dtype=emission.dtype)], dim=-1)
    out = rasterizer(means3D=pc.get_xyz, means2D=screenspace_points, shs=None, colors_precomp=colors,
                     opacities=pc.get_opacity, scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp)
    rendered = out[0]
    return rendered[:3]


def render_lita_halo(viewpoint_camera, pc, pipe, bg_color, halo, scale_s: float, use_glow: bool,
                     depth_threshold=None, iteration=None) -> dict:
    pkg = lita_render(viewpoint_camera, pc, pipe, bg_color, depth_threshold=depth_threshold, iteration=iteration)
    scene = pkg["render"]
    if not use_glow or halo is None:
        z = torch.zeros_like(scene)
        pkg.update({"emission": z, "glow": z, "low_composite": pkg["render_low"],
                    "bright_composite": scene, "denoised_composite": pkg["denoised_list"][-1]})
        return pkg
    E = render_emission(viewpoint_camera, pc, pipe, bg_color, depth_threshold, iteration)
    glow = fft_conv2d(E, halo.kernel.kernel())
    pkg["emission"] = E
    pkg["glow"] = glow
    pkg["low_composite"] = (pkg["render_low"] + glow).clamp(0, 1)
    pkg["bright_composite"] = (scene + scale_s * glow).clamp(0, 1)
    pkg["denoised_composite"] = (pkg["denoised_list"][-1] + scale_s * glow).clamp(0, 1)
    return pkg
