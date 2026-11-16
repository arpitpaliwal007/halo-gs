#!/usr/bin/env python
"""Score rendered views against their references using the manifest.

  python eval_glow.py -m out/pondbike/halo --iteration 30000
  python eval_glow.py --renders out/pondbike/halo/renders/iteration_30000/test

Pairs by image name (never by directory order).  Reports full-image and
glow / non-glow region PSNR, SSIM, LPIPS (full + glow crop).  Masks come from
the *reference* image only, so every method is scored on identical pixels.
No CUDA / gaussian-splatting needed: runs anywhere with torch.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from halogs.masks import glow_mask  # noqa: E402
from halogs.metrics import all_metrics  # noqa: E402

def _nanmean(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    return float(np.mean(xs)) if xs else float("nan")



def load_img(path, device):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).to(device)


def evaluate_dir(renders_dir: str, thresh: float, radius: int, with_lpips: bool, device: str, save_masks: bool):
    with open(os.path.join(renders_dir, "manifest.json")) as f:
        manifest = json.load(f)
    rows = []
    if save_masks:
        os.makedirs(os.path.join(renders_dir, "mask"), exist_ok=True)
    for m in manifest:
        pred = load_img(os.path.join(renders_dir, m["render"]), device)
        gt = load_img(os.path.join(renders_dir, m["gt"]), device)
        assert pred.shape == gt.shape, f"shape mismatch for {m['image_name']}: {pred.shape} vs {gt.shape}"
        gm = glow_mask(gt, thresh, radius)
        r = all_metrics(pred, gt, gm, with_lpips=with_lpips)
        r["view"] = m["image_name"]
        rows.append(r)
        if save_masks:
            Image.fromarray((gm[0].cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(renders_dir, "mask", f"{m['image_name']}.png"))
    keys = [k for k in rows[0] if k != "view"]
    mean = {k: float(_nanmean([r[k] for r in rows])) for k in keys}
    mean["n_views"] = len(rows)
    with open(os.path.join(renders_dir, "per_view.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["view"] + keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(renders_dir, "metrics.json"), "w") as f:
        json.dump({"mean": mean, "mask": {"thresh": thresh, "radius": radius}}, f, indent=2)
    return mean


def evaluate_clean(renders_dir: str, clean_dir: str, thresh: float, radius: int, with_lpips: bool, device: str,
                   pred_key: str = "auto"):
    """Score a render against clean / normal-light references, paired by image name.

    pred_key: 'scene' (glow-free render; synthetic glow scenes), 'bright' (litalite enhanced
    output; e.g. LOM normal-light references) or 'auto' (bright when present, else scene).
    """
    with open(os.path.join(renders_dir, "manifest.json")) as f:
        manifest = json.load(f)
    clean = {os.path.splitext(n)[0]: os.path.join(clean_dir, n) for n in os.listdir(clean_dir)}
    rows = []
    for m in manifest:
        stem = os.path.splitext(m["image_name"])[0]  # camera names may or may not keep the extension
        if stem not in clean:
            print("no clean reference for", stem)
            continue
        key = pred_key if pred_key != "auto" else ("bright" if "bright" in m else "scene")
        pred = load_img(os.path.join(renders_dir, m[key]), device)
        gt_obs = load_img(os.path.join(renders_dir, m["gt"]), device)
        gt_clean = load_img(clean[stem], device)
        if gt_clean.shape != pred.shape:
            gt_clean = torch.nn.functional.interpolate(gt_clean[None], size=pred.shape[1:], mode="bilinear", align_corners=False)[0]
        gm = glow_mask(gt_obs, thresh, radius)  # regions defined on the observed (glowed) image
        r = all_metrics(pred, gt_clean, gm, with_lpips=with_lpips)
        r["view"] = stem
        rows.append(r)
    if not rows:
        raise RuntimeError(f"no clean reference in {clean_dir} matches any rendered view name")
    keys = [k for k in rows[0] if k != "view"]
    mean = {k: float(_nanmean([r[k] for r in rows])) for k in keys}
    mean["n_views"] = len(rows)
    with open(os.path.join(renders_dir, "metrics_clean.json"), "w") as f:
        json.dump({"mean": mean, "mask": {"thresh": thresh, "radius": radius}}, f, indent=2)
    return mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model_path", type=str, default=None)
    p.add_argument("--clean_gt_dir", type=str, default=None,
                   help="dir of clean references: gt_clean/ of a synthetic glow scene, or normal-light images (LOM high/)")
    p.add_argument("--clean_pred", choices=["auto", "scene", "bright"], default="auto",
                   help="which render to score against --clean_gt_dir (auto: bright for litalite models, else scene)")
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--renders", type=str, default=None, help="explicit renders dir (contains manifest.json)")
    p.add_argument("--mask_thresh", type=float, default=0.92)
    p.add_argument("--mask_radius", type=int, default=48)
    p.add_argument("--no_lpips", action="store_true")
    p.add_argument("--save_masks", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.renders:
        dirs = [args.renders]
    else:
        base = os.path.join(args.model_path, "renders")
        its = sorted(int(d.split("_")[1]) for d in os.listdir(base) if d.startswith("iteration_"))
        it = its[-1] if args.iteration < 0 else args.iteration
        dirs = [os.path.join(base, f"iteration_{it}", s) for s in ("test", "train")]
        dirs = [d for d in dirs if os.path.exists(os.path.join(d, "manifest.json"))]

    for d in dirs:
        mean = evaluate_dir(d, args.mask_thresh, args.mask_radius, not args.no_lpips, args.device, args.save_masks)
        print(f"{d}\n  PSNR {mean['psnr']:.2f}  SSIM {mean['ssim']:.4f}  LPIPS {mean.get('lpips', float('nan')):.4f}  "
              f"| glow: PSNR {mean['psnr_glow']:.2f} SSIM {mean['ssim_glow']:.4f} LPIPS-crop {mean.get('lpips_glowcrop', float('nan')):.4f} "
              f"| non-glow: PSNR {mean['psnr_nonglow']:.2f} SSIM {mean['ssim_nonglow']:.4f}  (glow frac {mean['glow_frac']:.3f}, n={mean['n_views']})")
        if args.clean_gt_dir:
            mc = evaluate_clean(d, args.clean_gt_dir, args.mask_thresh, args.mask_radius, not args.no_lpips, args.device,
                                args.clean_pred)
            print(f"  [{args.clean_pred} vs CLEAN gt]  PSNR {mc['psnr']:.2f}  SSIM {mc['ssim']:.4f}  | glow PSNR {mc['psnr_glow']:.2f} "
                  f"| non-glow PSNR {mc['psnr_nonglow']:.2f}")


if __name__ == "__main__":
    main()
