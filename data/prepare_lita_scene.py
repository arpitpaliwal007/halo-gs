#!/usr/bin/env python
"""Build a LITA-GS-layout scene from a prepared halo-gs scene (images/ + sparse/0).

LITA-GS expects:
  images_low/   dark inputs (training)
  images/       references used at test time (normal-light GT on LOM; for real night
                scenes without GT this is a copy of images_low, so held-out scoring
                is composite-vs-observed)
  sparse/0      COLMAP model
  depth_maps/   depth_<stem>.png   (Marigold, via LITA-GS "Depth Extraction/depth_extraction.py")   [optional]
  W_0.8/        <image name>       (CIConv structure prior)                                            [optional]

  python data/prepare_lita_scene.py --scene data/scenes/bilarf/pondbike --out data/lita/pondbike --prior
  python data/prepare_lita_scene.py --scene data/scenes/lom/buu_low --bright_dir data/scenes/lom/buu_bright/images \
         --out data/lita/buu --prior --depth

--prior runs the CIConv extraction here (needs the LITA-GS checkout, torch+CUDA).
--depth calls LITA-GS's Marigold script (needs their checkpoint under Depth Extraction/checkpoint/).
Without depth_maps/ or W_0.8/, train with --no_depth / --no_prior.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def link_or_copy(src, dst):
    if os.path.lexists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def extract_prior(lita_root, images_low, out_dir):
    import cv2
    import numpy as np
    import torch
    sys.path.insert(0, lita_root)
    from ciconv2d import CIConv2d  # type: ignore
    model = CIConv2d("W", k=3, scale=0.8).cuda()
    os.makedirs(out_dir, exist_ok=True)
    for n in sorted(os.listdir(images_low)):
        if not n.endswith(IMG_EXT):
            continue
        img = cv2.imread(os.path.join(images_low, n))
        t = torch.from_numpy(img.copy()).cuda().float().div(255.0).permute(2, 0, 1).unsqueeze(0).contiguous()
        with torch.no_grad():
            feat = model(t)
        g = (np.clip(feat[0][0].detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, n), g)
    print("[prepare_lita] wrote priors ->", out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, help="halo-gs scene: images/ + sparse/0 (dark images)")
    p.add_argument("--out", required=True)
    p.add_argument("--bright_dir", default=None, help="normal-light references with the same file names (LOM); default = copy of dark")
    p.add_argument("--prior", action="store_true")
    p.add_argument("--depth", action="store_true")
    p.add_argument("--lita_root", default=None)
    args = p.parse_args()

    from halogs.gs_path import find_lita_root
    lita_root = find_lita_root(args.lita_root) if (args.prior or args.depth) else None

    src_imgs = os.path.join(args.scene, "images")
    names = sorted(n for n in os.listdir(src_imgs) if n.endswith(IMG_EXT))
    os.makedirs(os.path.join(args.out, "images_low"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    for n in names:
        link_or_copy(os.path.join(src_imgs, n), os.path.join(args.out, "images_low", n))
        ref = os.path.join(args.bright_dir, n) if args.bright_dir else os.path.join(src_imgs, n)
        if not os.path.exists(ref):
            raise FileNotFoundError(f"reference missing for {n}: {ref}")
        link_or_copy(ref, os.path.join(args.out, "images", n))
    link_or_copy(os.path.join(args.scene, "sparse"), os.path.join(args.out, "sparse"))
    print(f"[prepare_lita] {len(names)} images -> {args.out}  (references: {'bright' if args.bright_dir else 'copy of dark'})")

    if args.prior:
        extract_prior(lita_root, os.path.join(args.out, "images_low"), os.path.join(args.out, "W_0.8"))
    if args.depth:
        script = os.path.join(lita_root, "Depth Extraction", "depth_extraction.py")
        subprocess.run([sys.executable, script, "-s", os.path.abspath(args.out)], cwd=lita_root, check=True)


if __name__ == "__main__":
    main()
