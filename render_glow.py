#!/usr/bin/env python
"""Render a trained HaloGS / baseline model and write an evaluation manifest.

  python render_glow.py -m out/pondbike/halo --iteration 30000

Writes <model_path>/renders/iteration_N/{test,train}/
    render/<name>.png     composite (what is compared to the reference)
    scene/<name>.png      glow-free scene
    glow/<name>.png       glow layer (x4 for visibility; raw .npy also saved when --save_npy)
    emission/<name>.png   rendered emission map
    gt/<name>.png         observed reference
    bright/<name>.png     litalite modes only: the enhanced (reflectance) output
    manifest.json         render <-> camera <-> reference pairing (by image name)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gs_root", type=str, default=None)
_pre_args, _ = _pre.parse_known_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from halogs.gs_path import add_gs_to_path  # noqa: E402

add_gs_to_path(_pre_args.gs_root)

from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402

from halogs.gaussians import EmissiveGaussianModel  # noqa: E402
from halogs.halo import HaloModel, render_halo  # noqa: E402
from halogs.litalite import ToneCurve, render_litalite  # noqa: E402
from halogs.splits import load_split  # noqa: E402


def to_png(t: torch.Tensor, path: str, scale: float = 1.0) -> None:
    a = (t.detach().clamp(0, 1 / max(scale, 1e-6)) * scale * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(a).save(path)


@torch.no_grad()
def render_set(out_dir, cams, gaussians, pipe, bg, halo, mode, save_npy, tone=None):
    lit = mode.startswith("litalite")
    subs = ("render", "scene", "glow", "emission", "gt") + (("bright",) if lit else ())
    for sub in subs:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    manifest = []
    for cam in cams:
        if lit:
            pkg = render_litalite(render, cam, gaussians, pipe, bg, halo, use_glow=halo is not None, tone=tone)
        else:
            pkg = render_halo(render, cam, gaussians, pipe, bg, halo, mode=mode)
        name = cam.image_name
        if lit:
            to_png(pkg["bright"], os.path.join(out_dir, "bright", f"{name}.png"))
        to_png(pkg["render"], os.path.join(out_dir, "render", f"{name}.png"))
        to_png(pkg["scene"], os.path.join(out_dir, "scene", f"{name}.png"))
        to_png(pkg["glow"], os.path.join(out_dir, "glow", f"{name}.png"), scale=4.0)
        to_png(pkg["emission"], os.path.join(out_dir, "emission", f"{name}.png"))
        to_png(cam.original_image, os.path.join(out_dir, "gt", f"{name}.png"))
        if save_npy:
            np.save(os.path.join(out_dir, "glow", f"{name}.npy"), pkg["glow"].cpu().numpy())
        manifest.append({"image_name": name, "colmap_id": int(getattr(cam, "colmap_id", -1)),
                         "render": f"render/{name}.png", "gt": f"gt/{name}.png",
                         "scene": f"scene/{name}.png", "glow": f"glow/{name}.png",
                         **({"bright": f"bright/{name}.png"} if lit else {})})
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    names = [m["image_name"] for m in manifest]
    assert len(names) == len(set(names)), "duplicate image names in manifest"
    return manifest


def main():
    parser = argparse.ArgumentParser(description="HaloGS rendering")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--gs_root", type=str, default=None)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--save_npy", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    args.eval = True
    safe_state(False)  # never let gaussian-splatting swallow stdout; --quiet only trims the progress line

    dataset, pipe = lp.extract(args), pp.extract(args)
    with open(os.path.join(dataset.model_path, "config.json")) as f:
        cfg = json.load(f)
    mode = cfg.get("mode", "halo")

    gaussians = EmissiveGaussianModel(dataset.sh_degree, with_illum=mode.startswith("litalite"),
                                      illum_channels=int(cfg.get("illum_channels", 1)))
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    it = scene.loaded_iter
    pc_dir = os.path.join(dataset.model_path, "point_cloud", f"iteration_{it}")

    halo = None
    if mode in ("halo", "litalite_halo"):
        halo = HaloModel.load(os.path.join(pc_dir, "halo.pt"), device="cuda")
        print("[halogs] kernel:", halo.kernel.describe())

    tone = None
    if os.path.exists(os.path.join(pc_dir, "tone.pt")):
        tone = ToneCurve.load(os.path.join(pc_dir, "tone.pt"), device="cuda")
        print("[halogs] tone curve:", tone.describe())

    split = load_split(os.path.join(dataset.model_path, "split.json"))
    by_name = {c.image_name: c for c in scene.getTrainCameras() + scene.getTestCameras()}
    test_cams = [by_name[n] for n in split["test"] if n in by_name]
    train_cams = [by_name[n] for n in split["train"] if n in by_name]
    missing = [n for n in split["test"] + split["train"] if n not in by_name]
    if missing:
        print("[halogs] WARNING: cameras in split.json not found in loaded scene:", missing[:5], "...")

    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    base = os.path.join(dataset.model_path, "renders", f"iteration_{it}")
    render_set(os.path.join(base, "test"), test_cams, gaussians, pipe, bg, halo, mode, args.save_npy, tone)
    if not args.skip_train:
        render_set(os.path.join(base, "train"), train_cams, gaussians, pipe, bg, halo, mode, args.save_npy, tone)
    print(f"[halogs] wrote {base}")


if __name__ == "__main__":
    main()
