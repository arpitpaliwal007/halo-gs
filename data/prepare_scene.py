#!/usr/bin/env python
"""Prepare one real glow scene for 3DGS.

Handles the three layouts we meet in the wild and always produces
<out>/images + <out>/sparse/0:

  1. already COLMAP (scene has sparse/0)                -> copy / link
  2. LLFF (images + poses_bounds.npy, e.g. Bilarf)       -> data/llff_to_colmap.py (keeps the given poses)
  3. images only (or raw DNGs)                           -> gaussian-splatting convert.py (full COLMAP SfM)

RawNeRF scenes: if only raw/*.dng exist they are developed to sRGB with rawpy
(camera white balance, no auto-brightening, sRGB gamma) so the glow present in
the capture is kept as-is.

  python data/prepare_scene.py --scene data/raw/bilarf/pondbike --out data/scenes/bilarf/pondbike
  python data/prepare_scene.py --scene data/raw/rawnerf/scenes/candlefiat --out data/scenes/rawnerf/candlefiat --max_side 1600
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def develop_dngs(raw_dir, out_dir, max_side):
    import rawpy
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    for dng in sorted(glob.glob(os.path.join(raw_dir, "*.[dD][nN][gG]"))):
        dst = os.path.join(out_dir, os.path.splitext(os.path.basename(dng))[0] + ".png")
        if os.path.exists(dst):
            continue
        with rawpy.imread(dng) as r:
            rgb = r.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8, gamma=(2.222, 4.5))
        im = Image.fromarray(rgb)
        if max_side and max(im.size) > max_side:
            s = max_side / max(im.size)
            im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
        im.save(dst)
        print("developed", dst)


def resize_dir(src, dst, max_side):
    from PIL import Image
    os.makedirs(dst, exist_ok=True)
    for n in sorted(os.listdir(src)):
        if not n.endswith(IMG_EXT):
            continue
        im = Image.open(os.path.join(src, n)).convert("RGB")
        if max_side and max(im.size) > max_side:
            s = max_side / max(im.size)
            im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
        im.save(os.path.join(dst, os.path.splitext(n)[0] + ".png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--images_dir", default=None, help="which image folder to use (default: images, else images_4, else raw)")
    p.add_argument("--max_side", type=int, default=0, help="downscale copies to this longest side (0 = keep)")
    p.add_argument("--gs_root", default=os.environ.get("GS_ROOT", os.path.join(os.path.dirname(HERE), "..", "gaussian-splatting")))
    p.add_argument("--no_gpu", action="store_true")
    p.add_argument("--force_sfm", action="store_true", help="ignore provided poses and run full COLMAP")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    s = args.scene

    # pick an image folder
    cand = [args.images_dir] if args.images_dir else ["images", "images_2", "images_4", "input"]
    img_dir = next((os.path.join(s, c) for c in cand if c and os.path.isdir(os.path.join(s, c))), None)
    raw_dir = os.path.join(s, "raw")

    if img_dir is None and os.path.isdir(raw_dir):
        img_dir = os.path.join(args.out, "images_developed")
        develop_dngs(raw_dir, img_dir, args.max_side)
    assert img_dir is not None, f"no images found in {s}"

    has_sparse = os.path.isdir(os.path.join(s, "sparse", "0")) and not args.force_sfm
    has_llff = os.path.isfile(os.path.join(s, "poses_bounds.npy")) and not args.force_sfm

    if has_sparse and not args.max_side:
        print("[prepare] using provided COLMAP sparse/0")
        link_or_copy(os.path.join(s, "sparse"), os.path.join(args.out, "sparse"))
        os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
        for n in sorted(os.listdir(img_dir)):
            if n.endswith(IMG_EXT):
                link_or_copy(os.path.join(img_dir, n), os.path.join(args.out, "images", n))
        return

    if has_llff:
        print("[prepare] LLFF poses found -> triangulating with known poses")
        tmp_imgs = img_dir
        if args.max_side:
            tmp_imgs = os.path.join(args.out, "images_src")
            resize_dir(img_dir, tmp_imgs, args.max_side)
            # keep poses_bounds next to the resized images for llff_to_colmap's focal rescale
            shutil.copy2(os.path.join(s, "poses_bounds.npy"), os.path.join(args.out, "poses_bounds.npy"))
            scene_dir, images_dir = args.out, "images_src"
        else:
            scene_dir, images_dir = s, os.path.basename(img_dir)
        cmd = [sys.executable, os.path.join(HERE, "llff_to_colmap.py"), "--scene", scene_dir, "--out", args.out,
               "--images_dir", images_dir] + (["--no_gpu"] if args.no_gpu else [])
        subprocess.run(cmd, check=True)
        return

    print("[prepare] no poses -> full COLMAP SfM via gaussian-splatting convert.py")
    inp = os.path.join(args.out, "input")
    if args.max_side:
        resize_dir(img_dir, inp, args.max_side)
    else:
        os.makedirs(inp, exist_ok=True)
        for n in sorted(os.listdir(img_dir)):
            if n.endswith(IMG_EXT):
                link_or_copy(os.path.join(img_dir, n), os.path.join(inp, n))
    cmd = [sys.executable, os.path.join(args.gs_root, "convert.py"), "-s", args.out] + (["--no_gpu"] if args.no_gpu else [])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
