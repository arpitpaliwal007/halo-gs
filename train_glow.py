#!/usr/bin/env python
"""Train HaloGS (or the vanilla / masked baselines) on a posed night scene.

Examples
--------
  # ours
  python train_glow.py -s data/scenes/bilarf/pondbike -m out/pondbike/halo --mode halo --eval
  # baselines, same data / split / schedule
  python train_glow.py -s data/scenes/bilarf/pondbike -m out/pondbike/vanilla --mode vanilla --eval
  python train_glow.py -s data/scenes/bilarf/pondbike -m out/pondbike/masked  --mode masked  --eval

Everything the official 3DGS train.py accepts is accepted here (-r, --iterations,
--densify_grad_threshold, ...).  Outputs in <model_path>:
  split.json, config.json, log.csv, point_cloud/iteration_N/{point_cloud.ply,emission.pt,halo.pt}
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from random import randint

import numpy as np
import torch

# --- make gaussian-splatting importable before anything else -----------------------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gs_root", type=str, default=None)
_pre_args, _ = _pre.parse_known_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from halogs.gs_path import add_gs_to_path  # noqa: E402

GS_ROOT = add_gs_to_path(_pre_args.gs_root)

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402

from halogs.gaussians import EmissiveGaussianModel  # noqa: E402
from halogs.halo import HaloModel, focal_px_from_camera, render_halo  # noqa: E402
from halogs.losses import emission_sparsity, glow_tv, photometric_loss  # noqa: E402
from halogs.masks import glow_mask  # noqa: E402
from halogs.metrics import all_metrics  # noqa: E402
from halogs.splits import make_split, save_split  # noqa: E402

def _nanmean(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    return float(np.mean(xs)) if xs else float("nan")



def add_halo_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("halo")
    g.add_argument("--gs_root", type=str, default=None, help="path to gaussian-splatting checkout")
    g.add_argument("--mode", choices=["halo", "vanilla", "masked"], default="halo")
    g.add_argument("--kernel", choices=["both", "apsf", "lens", "radial"], default="both")
    g.add_argument("--kernel_radius", type=int, default=-1, help="px; -1 = 0.1*max(H,W) clamped to [32,160]")
    g.add_argument("--freeze_kernel", action="store_true")
    g.add_argument("--kernel_lr", type=float, default=2e-3)
    g.add_argument("--emission_lr", type=float, default=0.01)
    g.add_argument("--lambda_emission", type=float, default=0.01, help="L1 sparsity on per-Gaussian emission")
    g.add_argument("--lambda_glow_tv", type=float, default=0.0)
    g.add_argument("--glow_from_iter", type=int, default=500, help="iterations of plain 3DGS before glow is enabled")
    g.add_argument("--per_view_exposure", action="store_true", help="learn a log-gain per training view")
    g.add_argument("--exposure_lr", type=float, default=1e-3)
    g.add_argument("--split", choices=["every8", "glowgs"], default="every8")
    g.add_argument("--n_train_views", type=int, default=6, help="for --split glowgs")
    g.add_argument("--mask_thresh", type=float, default=0.92)
    g.add_argument("--mask_radius", type=int, default=48)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--test_every", type=int, default=1000)
    g.add_argument("--log_every", type=int, default=100)
    g.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    g.add_argument("--quiet", action="store_true")
    g.add_argument("--tag", type=str, default="", help="free-form label written to config.json")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="HaloGS training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    add_halo_args(parser)
    args = parser.parse_args(sys.argv[1:])
    args.eval = True  # we always need a held-out set; the split itself is ours (see --split)

    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    set_seed(args.seed)
    safe_state(False)  # never let gaussian-splatting swallow stdout; --quiet only trims the progress line
    set_seed(args.seed)

    os.makedirs(dataset.model_path, exist_ok=True)
    # the official render/eval helpers read model_path/cfg_args (see arguments.get_combined_args)
    with open(os.path.join(dataset.model_path, "cfg_args"), "w") as f:
        f.write(str(argparse.Namespace(**vars(args))))
    device = "cuda"

    gaussians = EmissiveGaussianModel(dataset.sh_degree, emission_lr=args.emission_lr)
    scene = Scene(dataset, gaussians)

    # ---- explicit split written to disk --------------------------------------------
    all_cams = scene.getTrainCameras() + scene.getTestCameras()
    train_cams, test_cams = make_split(all_cams, args.split, args.n_train_views)
    scene.train_cameras[1.0] = train_cams
    scene.test_cameras[1.0] = test_cams
    save_split(os.path.join(dataset.model_path, "split.json"), train_cams, test_cams, args.split)
    print(f"[halogs] {len(train_cams)} train / {len(test_cams)} test views ({args.split})")

    gaussians.training_setup(opt)

    # ---- glow model ------------------------------------------------------------------
    cam0 = train_cams[0]
    H, W = int(cam0.image_height), int(cam0.image_width)
    radius = args.kernel_radius if args.kernel_radius > 0 else int(min(max(0.1 * max(H, W), 32), 160))
    focal = focal_px_from_camera(cam0)
    halo = None
    halo_opt = None
    if args.mode == "halo":
        halo = HaloModel(kernel_kind=args.kernel, radius=radius, focal_px=focal,
                         per_view_exposure=args.per_view_exposure,
                         train_names=[c.image_name for c in train_cams],
                         learn_kernel=not args.freeze_kernel).to(device)
        groups = halo.param_groups(args.kernel_lr, args.exposure_lr)
        halo_opt = torch.optim.Adam(groups, lr=0.0) if groups else None
        print(f"[halogs] kernel={args.kernel} radius={radius}px focal={focal:.1f}px  H={H} W={W}")

    cfg = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    cfg.update({"gs_root": GS_ROOT, "kernel_radius_used": radius, "focal_px": focal, "image_hw": [H, W]})
    with open(os.path.join(dataset.model_path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device=device)

    # masks (observed-image based, cached per camera)
    mask_cache = {}

    def valid_mask(cam):
        if args.mode != "masked":
            return None
        if cam.image_name not in mask_cache:
            gt = cam.original_image.to(device)
            mask_cache[cam.image_name] = 1.0 - glow_mask(gt, args.mask_thresh, args.mask_radius)
        return mask_cache[cam.image_name]

    log_path = os.path.join(dataset.model_path, "log.csv")
    log_f = open(log_path, "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["iter", "loss", "l1", "n_gaussians", "emission_mean", "emission_max", "frac_emitters",
                  "q", "T", "gamma_px", "w_lens", "gain_mean", "peak_mem_gb", "sec_per_iter"])

    viewpoint_stack = None
    ema_loss = 0.0
    t0 = time.time()
    last_t = t0
    for iteration in range(1, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = train_cams.copy()
        cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        use_glow = args.mode == "halo" and iteration > args.glow_from_iter
        pkg = render_halo(render, cam, gaussians, pipe, bg, halo, mode="halo" if use_glow else "vanilla")
        image = pkg["render"]
        gt = cam.original_image.to(device)

        loss, l1 = photometric_loss(image, gt, opt.lambda_dssim, valid_mask(cam))
        if use_glow:
            if args.lambda_emission > 0:
                loss = loss + args.lambda_emission * emission_sparsity(gaussians.get_emission)
            if args.lambda_glow_tv > 0:
                loss = loss + args.lambda_glow_tv * glow_tv(pkg["glow"])
        loss.backward()

        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            if iteration % 10 == 0 and not args.quiet:
                print(f"\r[{iteration}/{opt.iterations}] loss {ema_loss:.5f} N={gaussians.get_xyz.shape[0]}", end="")

            # densification (scene-pass statistics, as in the official code)
            vis, radii, vsp = pkg["visibility_filter"], pkg["radii"], pkg["viewspace_points"]
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[vis] = torch.max(gaussians.max_radii2D[vis], radii[vis])
                gaussians.add_densification_stats(vsp, vis)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if halo_opt is not None:
                    halo_opt.step()
                    halo_opt.zero_grad(set_to_none=True)

            # logging
            if iteration % args.log_every == 0 or iteration == opt.iterations:
                now = time.time()
                es = gaussians.emission_stats()
                kd = halo.kernel.describe() if halo is not None else {}
                hd = halo.describe() if halo is not None else {}
                peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                log.writerow([iteration, f"{ema_loss:.6f}", f"{l1.item():.6f}", es["n_gaussians"],
                              f"{es['emission_mean']:.5f}", f"{es['emission_max']:.4f}", f"{es['frac_emitters_gt_0.1']:.4f}",
                              kd.get("q", ""), kd.get("T", ""), kd.get("gamma_px", ""), kd.get("w_lens", ""),
                              hd.get("gain_mean", ""), f"{peak:.3f}", f"{(now - last_t) / args.log_every:.4f}"])
                log_f.flush()
                last_t = now

            # held-out evaluation (composite vs observed reference)
            if iteration % args.test_every == 0 or iteration == opt.iterations:
                evaluate(iteration, test_cams, gaussians, pipe, bg, halo, args, dataset.model_path, use_glow or args.mode != "halo")

            if iteration in args.save_iterations or iteration == opt.iterations:
                print(f"\n[ITER {iteration}] saving")
                scene.save(iteration)
                pc_dir = os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}")
                gaussians.save_emission(os.path.join(pc_dir, "emission.pt"))
                if halo is not None:
                    halo.save(os.path.join(pc_dir, "halo.pt"))

    log_f.close()
    print(f"\n[halogs] done in {(time.time() - t0) / 60:.1f} min -> {dataset.model_path}")


@torch.no_grad()
def evaluate(iteration, test_cams, gaussians, pipe, bg, halo, args, model_path, glow_on):
    if not test_cams:
        return
    rows = []
    for cam in test_cams:
        pkg = render_halo(render, cam, gaussians, pipe, bg, halo, mode="halo" if (glow_on and args.mode == "halo") else "vanilla")
        gt = cam.original_image.to(pkg["render"].device)
        gm = glow_mask(gt, args.mask_thresh, args.mask_radius)
        m = all_metrics(pkg["render"].clamp(0, 1), gt, gm, with_lpips=False)
        m["view"] = cam.image_name
        rows.append(m)
    keys = [k for k in rows[0] if k != "view"]
    mean = {k: float(_nanmean([r[k] for r in rows])) for k in keys}
    print(f"\n[ITER {iteration}] test  PSNR {mean['psnr']:.2f}  SSIM {mean['ssim']:.3f}  "
          f"| glow PSNR {mean['psnr_glow']:.2f}  non-glow PSNR {mean['psnr_nonglow']:.2f}  (glow frac {mean['glow_frac']:.3f})")
    with open(os.path.join(model_path, "test_curve.jsonl"), "a") as f:
        f.write(json.dumps({"iter": iteration, **mean}) + "\n")


if __name__ == "__main__":
    main()
