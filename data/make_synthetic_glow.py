#!/usr/bin/env python
"""Controlled glow scenes with exact ground truth.

Takes any posed scene with *clean* images (a daytime capture, or a well-lit LOM
scene), places K point emitters in 3D (sampled from the sparse COLMAP cloud,
optionally biased to bright image regions), and composites a physically
generated halo into every view with the same kernel family the model fits:

    I_glow = clip( I_clean + k(q, T, gamma, w) * E_v )

E_v is the emitter map of view v (projected emitters as small discs, occlusion
ignored - emitters are placed in free space in front of surfaces).  Writes the
glowed images plus the clean references, the glow layers, the emitters and
the true kernel parameters, so recovery of (scene, glow, q, T, gamma) can be
measured exactly.  Camera poses are copied unchanged.

  python data/make_synthetic_glow.py --scene data/scenes/lom/buu_bright --out data/scenes/synth/buu_glow \
         --n_emitters 3 --T 1.5 --q 0.9 --gamma 6 --w_lens 0.4 --intensity 3.0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from halogs.apsf import GlowKernel  # noqa: E402
from halogs.compositing import fft_conv2d  # noqa: E402
from halogs.gs_path import add_gs_to_path  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, help="prepared COLMAP scene with clean images")
    p.add_argument("--out", required=True)
    p.add_argument("--gs_root", default=None)
    p.add_argument("--n_emitters", type=int, default=3)
    p.add_argument("--emitter_radius_px", type=float, default=6.0)
    p.add_argument("--intensity", type=float, default=3.0, help="emission radiance per emitter (>1 saturates the source)")
    p.add_argument("--q", type=float, default=0.9)
    p.add_argument("--T", type=float, default=1.5)
    p.add_argument("--gamma", type=float, default=6.0)
    p.add_argument("--w_lens", type=float, default=0.4)
    p.add_argument("--kernel_radius", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_side", type=int, default=0)
    args = p.parse_args()

    add_gs_to_path(args.gs_root)
    from scene.colmap_loader import (qvec2rotmat, read_extrinsics_binary, read_extrinsics_text,  # type: ignore
                                     read_intrinsics_binary, read_intrinsics_text, read_points3D_binary,
                                     read_points3D_text)

    sp = os.path.join(args.scene, "sparse", "0")
    try:
        cams = read_intrinsics_binary(os.path.join(sp, "cameras.bin"))
        imgs = read_extrinsics_binary(os.path.join(sp, "images.bin"))
        xyz, rgb, _ = read_points3D_binary(os.path.join(sp, "points3D.bin"))
    except Exception:
        cams = read_intrinsics_text(os.path.join(sp, "cameras.txt"))
        imgs = read_extrinsics_text(os.path.join(sp, "images.txt"))
        xyz, rgb, _ = read_points3D_text(os.path.join(sp, "points3D.txt"))

    rng = np.random.default_rng(args.seed)
    # emitters: random sparse points, pushed 10% towards the camera centroid so they sit in free space
    centers = np.stack([-(qvec2rotmat(im.qvec).T @ im.tvec) for im in imgs.values()])
    cam_centroid = centers.mean(0)
    idx = rng.choice(len(xyz), size=args.n_emitters, replace=False)
    emitters = xyz[idx] * 0.9 + cam_centroid * 0.1

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "gt_clean"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "glow"), exist_ok=True)
    if not os.path.exists(os.path.join(args.out, "sparse")):
        shutil.copytree(os.path.join(args.scene, "sparse"), os.path.join(args.out, "sparse"))

    kernel = None
    meta = {"emitters_xyz": emitters.tolist(), "q": args.q, "T": args.T, "gamma_px": args.gamma,
            "w_lens": args.w_lens, "intensity": args.intensity, "views": []}
    for im in sorted(imgs.values(), key=lambda i: i.name):
        cam = cams[im.camera_id]
        src = os.path.join(args.scene, "images", im.name)
        pil = Image.open(src).convert("RGB")
        scale = 1.0
        if args.max_side and max(pil.size) > args.max_side:
            scale = args.max_side / max(pil.size)
            pil = pil.resize((round(pil.size[0] * scale), round(pil.size[1] * scale)), Image.LANCZOS)
        img = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1)
        H, W = img.shape[1:]
        if cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params[:4]
        else:  # SIMPLE_PINHOLE / SIMPLE_RADIAL (distortion ignored for emitter projection)
            fx = fy = cam.params[0]
            cx, cy = cam.params[1], cam.params[2]
        fx, fy, cx, cy = fx * scale, fy * scale, cx * scale, cy * scale
        if kernel is None:
            radius = args.kernel_radius if args.kernel_radius > 0 else int(min(max(0.1 * max(H, W), 32), 160))
            kernel = GlowKernel(radius=radius, focal_px=float(fx), components="both", q_init=args.q, T_init=args.T,
                                gamma_init=args.gamma, w_init=args.w_lens, learn=False)
            K = kernel.kernel().detach()
            meta["kernel_radius"] = radius
            meta["focal_px"] = float(fx)

        R = qvec2rotmat(im.qvec)
        t = im.tvec
        E = torch.zeros(3, H, W)
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        vis = []
        for e in emitters:
            pc = R @ e + t
            if pc[2] <= 1e-3:
                vis.append(False)
                continue
            u, v = fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy
            if not (0 <= u < W and 0 <= v < H):
                vis.append(False)
                continue
            d2 = (xx - u) ** 2 + (yy - v) ** 2
            disc = torch.exp(-0.5 * d2 / (args.emitter_radius_px ** 2))
            E += args.intensity * disc.unsqueeze(0)
            vis.append(True)
        glow = fft_conv2d(E, K)
        out = (img + E + glow).clamp(0, 1)  # the source itself is also visible (saturated), plus its halo
        name = im.name  # keep the exact COLMAP image name so the 3DGS loader finds it
        kw = {"quality": 97} if name.lower().endswith((".jpg", ".jpeg")) else {}
        Image.fromarray((out.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(os.path.join(args.out, "images", name), **kw)
        Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(os.path.join(args.out, "gt_clean", name), **kw)
        np.save(os.path.join(args.out, "glow", os.path.splitext(name)[0] + ".npy"), glow.numpy())
        meta["views"].append({"image": name, "source": im.name, "emitters_visible": vis})
        print("wrote", name, "visible emitters:", sum(vis))

    with open(os.path.join(args.out, "synthetic_glow.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if args.max_side:
        print("NOTE: --max_side rescaled the images; train with the matching -r so intrinsics stay consistent, "
              "or prepare the clean scene at the target resolution first.")


if __name__ == "__main__":
    main()
