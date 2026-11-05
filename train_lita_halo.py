#!/usr/bin/env python
"""HaloGS on the LITA-GS backbone: LITA-GS (CVPR 2025) reflectance / illumination /
denoiser / tone-mapper model + HaloGS per-Gaussian emission and scatter kernel.

  # ours (LITA-GS + explicit light sources)
  python train_lita_halo.py -s data/lita/pondbike -m out_lita/pondbike/halo --mode halo --split every8
  # plain LITA-GS through the same script (identical split / schedule) = the baseline
  python train_lita_halo.py -s data/lita/pondbike -m out_lita/pondbike/lita --mode lita --split every8
  # the official LOM protocol (their config file, eval_index [0,4,10], theta=0.49, 20k iters)
  python train_lita_halo.py -s data/LOM_low_buu_colmap -m out_lita/buu/halo --mode halo --split lita \
         --configs $LITA_ROOT/arguments/LOM/buu_low.py

Scene layout (LITA-GS): images_low/ (dark inputs), images/ (references: normal-light GT on LOM,
or a copy of images_low for real night scenes without GT), sparse/0, optional depth_maps/ and W_0.8/.
Use data/prepare_lita_scene.py to build it from a halo-gs scene.

Physics:  low = tonemapper(illu * denoise(scene)) + k*E ;  bright = scene + s*(k*E),  s = theta/mean(low)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import runpy
import sys
import time
from random import randint

import numpy as np
import torch

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--lita_root", type=str, default=None)
_pre_args, _ = _pre.parse_known_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from halogs.gs_path import add_lita_to_path  # noqa: E402

LITA_ROOT = add_lita_to_path(_pre_args.lita_root)

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from ciconv2d import CIConv2d  # noqa: E402
from scene import Scene  # noqa: E402
from scene.cameras import Camera  # noqa: E402
from utils.loss_utils import TVLoss, l1_loss, l2_loss, local_pearson_loss, pearson_depth_loss, ssim  # noqa: E402

from halogs.halo import HaloModel, focal_px_from_camera  # noqa: E402
from halogs.lita.model import EmissiveLitaModel  # noqa: E402
from halogs.lita.render import render_lita_halo  # noqa: E402
from halogs.losses import emission_sparsity  # noqa: E402
from halogs.masks import glow_mask  # noqa: E402
from halogs.metrics import all_metrics  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _nanmean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def add_halo_args(p):
    g = p.add_argument_group("halo")
    g.add_argument("--lita_root", type=str, default=None)
    g.add_argument("--mode", choices=["halo", "lita"], default="halo", help="halo = LITA-GS + emission/glow; lita = plain LITA-GS")
    g.add_argument("--kernel", choices=["both", "apsf", "lens", "radial"], default="both")
    g.add_argument("--kernel_radius", type=int, default=-1)
    g.add_argument("--freeze_kernel", action="store_true")
    g.add_argument("--kernel_lr", type=float, default=2e-3)
    g.add_argument("--emission_lr", type=float, default=0.01)
    g.add_argument("--lambda_emission", type=float, default=0.01)
    g.add_argument("--glow_from_iter", type=int, default=500)
    g.add_argument("--split", choices=["lita", "every8", "glowgs"], default="lita",
                   help="lita = eval_index from --configs (default [0,4,10]); every8 = LLFF hold-out; glowgs = 6-view few-shot")
    g.add_argument("--n_train_views", type=int, default=6)
    g.add_argument("--no_depth", action="store_true", help="disable LITA-GS depth prior (no depth_maps/ needed)")
    g.add_argument("--no_prior", action="store_true", help="disable LITA-GS structure prior (no W_0.8/ needed)")
    g.add_argument("--no_denoiser", action="store_true")
    g.add_argument("--mask_thresh", type=float, default=0.92)
    g.add_argument("--mask_radius", type=int, default=48)
    g.add_argument("--seed", type=int, default=10)
    g.add_argument("--test_every", type=int, default=1000)
    g.add_argument("--log_every", type=int, default=100)
    g.add_argument("--save_iterations", nargs="+", type=int, default=[])
    g.add_argument("--configs", type=str, default="", help="LITA-GS python config (arguments/LOM/*.py)")
    g.add_argument("--quiet", action="store_true")
    g.add_argument("--tag", type=str, default="")


def merge_config(args, path):
    """Same semantics as LITA-GS utils.params_utils.merge_hparams, without mmcv."""
    cfg = runpy.run_path(path)
    for param in ("OptimizationParams", "ModelParams", "PipelineParams"):
        for k, v in cfg.get(param, {}).items():
            if hasattr(args, k):
                setattr(args, k, v)
    return args


def compute_eval_index(source_path, split, n_train, default_index):
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(source_path, "images_low")) if f.endswith(IMG_EXT))
    n = len(names)
    if split == "lita":
        return list(default_index), names
    if split == "every8":
        return [i for i in range(n) if i % 8 == 0], names
    if split == "glowgs":
        tr = sorted({int(round(i * (n - 1) / (n_train - 1))) for i in range(n_train)})
        return [i for i in range(n) if i not in tr], names
    raise ValueError(split)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="HaloGS on LITA-GS")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    add_halo_args(parser)
    args = parser.parse_args(sys.argv[1:])
    if args.configs:
        args = merge_config(args, args.configs)
    if args.no_depth:
        args.use_depth = False
    if args.no_prior:
        args.use_prior = False
    if args.no_denoiser:
        args.use_denoiser = False
    args.eval_index, all_names = compute_eval_index(args.source_path, args.split, args.n_train_views, args.eval_index)
    if not args.save_iterations:
        args.save_iterations = [args.iterations]

    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    os.makedirs(dataset.model_path, exist_ok=True)
    with open(os.path.join(dataset.model_path, "cfg_args"), "w") as f:
        f.write(str(argparse.Namespace(**vars(args))))
    set_seed(args.seed)
    device = "cuda"

    gaussians = EmissiveLitaModel(sh_degree=dataset.sh_degree, depth_feat_dim=1, prior_feat_dim=1, noise_feat_dim=3,
                                  illu_feat_dim=3, denoiser_stage=dataset.denoiser_stage, denoiser_layers=dataset.denoiser_layers,
                                  denoiser_channel=dataset.denoiser_channel, denoiser_mode=dataset.denoiser_mode,
                                  emission_lr=args.emission_lr)
    scene = Scene(dataset, gaussians, opt.use_depth, opt.use_prior, gap=pipe.interval)
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    with open(os.path.join(dataset.model_path, "split.json"), "w") as f:
        json.dump({"mode": args.split, "eval_index": args.eval_index,
                   "train": [c.image_name for c in train_cams], "test": [c.image_name for c in test_cams]}, f, indent=2)
    print(f"[halogs/lita] {len(train_cams)} train / {len(test_cams)} test views ({args.split}, eval_index={args.eval_index})")

    gaussians.training_setup(opt)
    gaussians.compute_3D_filter(cameras=train_cams)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device=device)
    tv_loss = TVLoss()
    prior_model = CIConv2d("W", k=3, scale=0.8).cuda() if opt.use_prior else None

    # glow model
    cam0 = train_cams[0]
    H, W = int(cam0.image_height), int(cam0.image_width)
    radius = args.kernel_radius if args.kernel_radius > 0 else int(min(max(0.1 * max(H, W), 32), 160))
    focal = focal_px_from_camera(cam0)
    halo = halo_opt = None
    if args.mode == "halo":
        halo = HaloModel(kernel_kind=args.kernel, radius=radius, focal_px=focal, learn_kernel=not args.freeze_kernel).to(device)
        groups = halo.param_groups(args.kernel_lr)
        halo_opt = torch.optim.Adam(groups, lr=0.0) if groups else None
        print(f"[halogs/lita] kernel={args.kernel} radius={radius}px focal={focal:.1f}px H={H} W={W}")

    cfg = {k: v for k, v in vars(args).items()}
    cfg.update({"lita_root": LITA_ROOT, "backend": "lita", "kernel_radius_used": radius, "focal_px": focal, "image_hw": [H, W]})
    with open(os.path.join(dataset.model_path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    log_f = open(os.path.join(dataset.model_path, "log.csv"), "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["iter", "loss", "l1_bright", "l1_low", "n_gaussians", "emission_mean", "emission_max", "frac_emitters",
                  "q", "T", "gamma_px", "w_lens", "peak_mem_gb", "sec_per_iter"])

    viewpoint_stack = None
    ema = 0.0
    t0 = last_t = time.time()
    for iteration in range(1, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if not viewpoint_stack:
            viewpoint_stack = train_cams.copy()
        cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        low_image = cam.low_image
        s = float(opt.exposure / low_image.mean().clamp_min(1e-4))
        target_image = torch.clamp(low_image * s, 0, 1)
        use_glow = args.mode == "halo" and iteration > args.glow_from_iter

        pkg = render_lita_halo(cam, gaussians, pipe, bg, halo, s, use_glow,
                               depth_threshold=opt.depth_threshold * scene.cameras_extent, iteration=iteration)
        rendered = pkg["render"]
        vsp, vis, radii = pkg["viewspace_points"], pkg["visibility_filter"], pkg["radii"]

        # --- LITA-GS losses, with HaloGS composites in place of raw renders ------------
        bright = pkg["bright_composite"]
        l1_b = l1_loss(bright, target_image)
        loss_ssim = 1.0 - ssim(bright, target_image.unsqueeze(0))
        loss = ((1.0 - opt.lambda_dssim) * l1_b + opt.lambda_dssim * loss_ssim) * opt.lambda_rgb
        if opt.use_depth:
            gt_depth = cam.gt_depth.unsqueeze(0).to(dataset.data_device)
            loss = loss + (pearson_depth_loss(pkg["depth"][0], gt_depth[0]) + local_pearson_loss(pkg["depth"][0], gt_depth[0], 128, 0.5)) * opt.lambda_depth
        if opt.use_prior:
            with torch.no_grad():
                target_features = prior_model(target_image.unsqueeze(0))
            loss = loss + l1_loss(torch.clamp(target_features, 0, 1).squeeze(0), torch.clamp(pkg["prior"], 0, 1)) * opt.lambda_prior
        l1_low = torch.zeros((), device=device)
        if opt.use_denoiser:
            low = pkg["low_composite"]
            l1_low = l1_loss(low, low_image)
            loss = loss + (1.0 - opt.lambda_dssim_low) * l1_low + opt.lambda_dssim_low * (1.0 - ssim(low, low_image.unsqueeze(0)))
            loss = loss + l2_loss(pkg["denoised_list"][-1], rendered) + tv_loss(pkg["denoised_list"][-1].unsqueeze(0))
        if use_glow and args.lambda_emission > 0:
            loss = loss + args.lambda_emission * emission_sparsity(gaussians.get_emission)
        loss.backward()

        with torch.no_grad():
            ema = 0.4 * loss.item() + 0.6 * ema
            if iteration % 10 == 0 and not args.quiet:
                print(f"\r[{iteration}/{opt.iterations}] loss {ema:.5f} N={gaussians.get_xyz.shape[0]}", end="")

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[vis] = torch.max(gaussians.max_radii2D[vis], radii[vis])
                gaussians.add_densification_stats(vsp, vis)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    gaussians.compute_3D_filter(cameras=train_cams)
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity_3dgs()
            if iteration % 100 == 0 and iteration > opt.densify_until_iter and iteration < opt.iterations - 100:
                gaussians.compute_3D_filter(cameras=train_cams)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if halo_opt is not None:
                    halo_opt.step()
                    halo_opt.zero_grad(set_to_none=True)

            if iteration % args.log_every == 0 or iteration == opt.iterations:
                now = time.time()
                es = gaussians.emission_stats()
                kd = halo.kernel.describe() if halo is not None else {}
                log.writerow([iteration, f"{ema:.6f}", f"{l1_b.item():.6f}", f"{l1_low.item():.6f}", es["n_gaussians"],
                              f"{es['emission_mean']:.5f}", f"{es['emission_max']:.4f}", f"{es['frac_emitters_gt_0.1']:.4f}",
                              kd.get("q", ""), kd.get("T", ""), kd.get("gamma_px", ""), kd.get("w_lens", ""),
                              f"{torch.cuda.max_memory_allocated() / 1e9:.3f}", f"{(now - last_t) / args.log_every:.4f}"])
                log_f.flush()
                last_t = now

            if iteration % args.test_every == 0 or iteration == opt.iterations:
                evaluate(iteration, test_cams, gaussians, pipe, bg, halo, opt, scene, args, use_glow or args.mode == "lita")

            if iteration in args.save_iterations or iteration == opt.iterations:
                print(f"\n[ITER {iteration}] saving")
                scene.save(iteration)
                pc_dir = os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}")
                gaussians.save_emission(os.path.join(pc_dir, "emission.pt"))
                if halo is not None:
                    halo.save(os.path.join(pc_dir, "halo.pt"))

    log_f.close()
    dump_renders(opt.iterations, test_cams, gaussians, pipe, bg, halo, opt, scene, args, dataset.model_path)
    print(f"\n[halogs/lita] done in {(time.time() - t0) / 60:.1f} min -> {dataset.model_path}")


@torch.no_grad()
def _eval_pkg(cam, gaussians, pipe, bg, halo, opt, scene, glow_on):
    low_image = cam.low_image
    s = float(opt.exposure / low_image.mean().clamp_min(1e-4))
    return render_lita_halo(cam, gaussians, pipe, bg, halo, s, glow_on,
                            depth_threshold=opt.depth_threshold * scene.cameras_extent, iteration=None)


@torch.no_grad()
def evaluate(iteration, test_cams, gaussians, pipe, bg, halo, opt, scene, args, glow_on):
    """Three numbers per view, all against held-out images:
       lita   : LITA-GS protocol  - glow-free denoised bright render vs images/ (normal-light GT on LOM)
       bright : denoised + s*glow vs images/   (fair when the reference itself contains glow)
       low    : tone-mapped composite vs images_low/ (observed dark photo), with glow-region split."""
    if not test_cams:
        return
    rows = []
    for cam in test_cams:
        pkg = _eval_pkg(cam, gaussians, pipe, bg, halo, opt, scene, glow_on and args.mode == "halo")
        gt = cam.original_image.clamp(0, 1)
        low_gt = cam.low_image.clamp(0, 1)
        den = pkg["denoised_list"][-1].clamp(0, 1) if opt.use_denoiser else pkg["render"].clamp(0, 1)
        gm_b = glow_mask(gt, args.mask_thresh, args.mask_radius)
        gm_l = glow_mask(low_gt, args.mask_thresh, args.mask_radius)
        m_lita = all_metrics(den, gt, gm_b, with_lpips=False)
        m_bright = all_metrics(pkg["denoised_composite"].clamp(0, 1), gt, gm_b, with_lpips=False)
        m_low = all_metrics(pkg["low_composite"].clamp(0, 1), low_gt, gm_l, with_lpips=False)
        rows.append({"view": cam.image_name, **{f"lita_{k}": v for k, v in m_lita.items()},
                     **{f"bright_{k}": v for k, v in m_bright.items()}, **{f"low_{k}": v for k, v in m_low.items()}})
    keys = [k for k in rows[0] if k != "view"]
    mean = {k: _nanmean([r[k] for r in rows]) for k in keys}
    print(f"\n[ITER {iteration}] test  LITA-protocol PSNR {mean['lita_psnr']:.2f} SSIM {mean['lita_ssim']:.3f} | "
          f"bright+glow PSNR {mean['bright_psnr']:.2f} | low composite PSNR {mean['low_psnr']:.2f} "
          f"(glow {mean['low_psnr_glow']:.2f} / non-glow {mean['low_psnr_nonglow']:.2f}, frac {mean['low_glow_frac']:.3f})")
    with open(os.path.join(scene.model_path, "test_curve.jsonl"), "a") as f:
        f.write(json.dumps({"iter": iteration, **mean}) + "\n")


@torch.no_grad()
def dump_renders(iteration, test_cams, gaussians, pipe, bg, halo, opt, scene, args, model_path):
    """renders/iteration_N/test (bright pair) and test_low (dark pair), each with a manifest for eval_glow.py."""
    from PIL import Image

    def to_png(t, path, scale=1.0):
        a = (t.detach().clamp(0, 1 / max(scale, 1e-6)) * scale * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(a).save(path)

    base = os.path.join(model_path, "renders", f"iteration_{iteration}")
    tb, tl = os.path.join(base, "test"), os.path.join(base, "test_low")
    for d in (tb, tl):
        for sub in ("render", "scene", "glow", "emission", "gt"):
            os.makedirs(os.path.join(d, sub), exist_ok=True)
    man_b, man_l = [], []
    for cam in test_cams:
        pkg = _eval_pkg(cam, gaussians, pipe, bg, halo, opt, scene, args.mode == "halo")
        n = cam.image_name
        den = pkg["denoised_list"][-1] if opt.use_denoiser else pkg["render"]
        to_png(pkg["denoised_composite"], os.path.join(tb, "render", f"{n}.png"))
        to_png(den, os.path.join(tb, "scene", f"{n}.png"))
        to_png(pkg["glow"], os.path.join(tb, "glow", f"{n}.png"), 4.0)
        to_png(pkg["emission"], os.path.join(tb, "emission", f"{n}.png"))
        to_png(cam.original_image, os.path.join(tb, "gt", f"{n}.png"))
        man_b.append({"image_name": n, "render": f"render/{n}.png", "gt": f"gt/{n}.png", "scene": f"scene/{n}.png", "glow": f"glow/{n}.png"})
        to_png(pkg["low_composite"], os.path.join(tl, "render", f"{n}.png"))
        to_png(pkg["render_low"], os.path.join(tl, "scene", f"{n}.png"))
        to_png(pkg["glow"], os.path.join(tl, "glow", f"{n}.png"), 4.0)
        to_png(pkg["emission"], os.path.join(tl, "emission", f"{n}.png"))
        to_png(cam.low_image, os.path.join(tl, "gt", f"{n}.png"))
        man_l.append({"image_name": n, "render": f"render/{n}.png", "gt": f"gt/{n}.png", "scene": f"scene/{n}.png", "glow": f"glow/{n}.png"})
    with open(os.path.join(tb, "manifest.json"), "w") as f:
        json.dump(man_b, f, indent=2)
    with open(os.path.join(tl, "manifest.json"), "w") as f:
        json.dump(man_l, f, indent=2)
    print(f"[halogs/lita] wrote {tb} and {tl}")


if __name__ == "__main__":
    main()
