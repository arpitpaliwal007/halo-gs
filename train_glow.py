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
from halogs.litalite import ColourCurveCamera, ToneCurve, bilateral, colour_constancy_loss, exposure_loss, pseudo_weight, brighten_target, grayworld_gains, illum_init_from, render_litalite, scene_gain  # noqa: E402
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
    g.add_argument("--mode", choices=["halo", "vanilla", "masked", "litalite", "litalite_halo"], default="halo",
                   help="litalite = corrected reference-free low-light model; litalite_halo = litalite + emission/glow")
    g.add_argument("--kernel", choices=["both", "apsf", "lens", "radial"], default="both")
    g.add_argument("--kernel_radius", type=int, default=-1, help="px; -1 = 0.1*max(H,W) clamped to [32,160]")
    g.add_argument("--freeze_kernel", action="store_true")
    g.add_argument("--gamma_min", type=float, default=0.3, help="px floor for the lens glare width; 0.3 lets it collapse to a delta, 3 forces a real halo")
    g.add_argument("--gamma_init", type=float, default=6.0)
    g.add_argument("--w_init", type=float, default=0.5, help="initial lens/APSF mixture weight")
    g.add_argument("--kernel_lr", type=float, default=2e-3)
    g.add_argument("--emission_lr", type=float, default=0.01)
    g.add_argument("--lambda_emission", type=float, default=0.01, help="L1 sparsity on per-Gaussian emission")
    g.add_argument("--lambda_glow_tv", type=float, default=0.0)
    g.add_argument("--glow_from_iter", type=int, default=500, help="iterations of plain 3DGS before glow is enabled")
    g.add_argument("--per_view_exposure", action="store_true", help="learn a log-gain per training view")
    g.add_argument("--exposure_lr", type=float, default=1e-3)
    g.add_argument("--split", choices=["every8", "glowgs", "views"], default="every8")
    g.add_argument("--test_views", nargs="+", default=None,
                   help="with --split views: test views as indices (numeric name order) or names, e.g. 1 17 24 9")
    g.add_argument("--test_views_by", choices=["index", "name"], default="index")
    g.add_argument("--n_train_views", type=int, default=6, help="for --split glowgs")
    g.add_argument("--mask_thresh", type=float, default=0.92)
    g.add_argument("--mask_radius", type=int, default=48)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--test_every", type=int, default=1000)
    g.add_argument("--log_every", type=int, default=100)
    g.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    g.add_argument("--quiet", action="store_true")
    g.add_argument("--tag", type=str, default="", help="free-form label written to config.json")
    g = p.add_argument_group("litalite")
    g.add_argument("--theta", type=float, default=0.40, help="target mean brightness of the brightened pseudo-target")
    g.add_argument("--white_balance", choices=["none", "grayworld"], default="none",
                   help="grayworld = scene-wide grey-world white balance of the bright target (training views only)")
    g.add_argument("--target_gamma", type=float, default=1.0,
                   help="tone curve t**gamma on the bright target; <1 lifts shadows / lowers contrast")
    g.add_argument("--target_gain", choices=["image", "scene"], default="image",
                   help="image = theta/mean(this image) (original); scene = one gain for all views")
    g.add_argument("--legacy_clip", action="store_true", help="clip target channels independently (original, not hue-safe)")
    g.add_argument("--dense_from", choices=["both", "bright"], default="both",
                   help="densification statistics from both branches (fix) or the bright one only (original)")
    g.add_argument("--illum_lr", type=float, default=0.01)
    g.add_argument("--w_exp", type=float, default=1.0)
    g.add_argument("--w_rec", type=float, default=1.0)
    g.add_argument("--w_tv", type=float, default=0.01, help="TV on the bright render (as in litalite.py)")
    g.add_argument("--clean_gt_dir", type=str, default=None,
                   help="normal-light references (e.g. LOM high/), matched by image name; scores the bright render")
    g.add_argument("--illum_channels", type=int, choices=[1, 3], default=1,
                   help="1 = grey illumination (original); 3 = RGB illumination (coloured light)")
    g.add_argument("--tone_curve", action="store_true", help="learn a per-view response curve g*x^gamma on the dark branch")
    g.add_argument("--tone_lr", type=float, default=1e-3)
    g.add_argument("--cam_model", choices=["none", "power", "colour_curve"], default="none",
                   help="dark-branch camera model: power = g*x^gamma (same as --tone_curve); "
                        "colour_curve = per-view 3x3 colour matrix + knot curve tied to a global curve")
    g.add_argument("--cam_knots", type=int, default=16)
    g.add_argument("--w_tie", type=float, default=0.01, help="tie per-view colour/curve to identity/global")
    g.add_argument("--w_expo", type=float, default=0.0, help="Zero-DCE exposure loss on the bright render")
    g.add_argument("--expo_target", type=float, default=0.6)
    g.add_argument("--expo_patch", type=int, default=16)
    g.add_argument("--w_col", type=float, default=0.0, help="Zero-DCE colour-constancy loss on the bright render")
    g.add_argument("--prior_from", type=int, default=3000, help="iteration from which exposure/colour losses are on")
    g.add_argument("--pseudo_fade", type=int, nargs=2, default=[0, 0], metavar=("START", "END"),
                   help="fade the pseudo-target weight w_exp linearly to 0 between START and END (0 0 = never)")
    g.add_argument("--denoise_target", choices=["none", "bilateral", "n2n_unet"], default="none",
                   help="edge-preserving filter on the brightened target (the dark loss keeps the raw photo)")
    g.add_argument("--dn_radius", type=int, default=2)
    g.add_argument("--dn_sigma_s", type=float, default=1.5)
    g.add_argument("--dn_sigma_r", type=float, default=0.08)
    g.add_argument("--n2n_iters", type=int, default=2000, help="U-Net Neighbor2Neighbor training steps")
    g.add_argument("--n2n_crop", type=int, default=128)
    g.add_argument("--n2n_batch", type=int, default=4)
    g.add_argument("--n2n_width", type=int, default=32, help="U-Net base channels (32 -> ~3.7 M params)")
    g.add_argument("--n2n_lr", type=float, default=1e-3)


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

    lit = args.mode.startswith("litalite")
    glow_mode = args.mode in ("halo", "litalite_halo")
    gaussians = EmissiveGaussianModel(dataset.sh_degree, emission_lr=args.emission_lr,
                                      with_illum=lit, illum_lr=args.illum_lr, illum_channels=args.illum_channels)
    scene = Scene(dataset, gaussians)

    # ---- explicit split written to disk --------------------------------------------
    all_cams = scene.getTrainCameras() + scene.getTestCameras()
    train_cams, test_cams = make_split(all_cams, args.split, args.n_train_views,
                                       test_views=args.test_views, test_views_by=args.test_views_by)
    if args.split == "views":
        print(f"[halogs] explicit test views: {[c.image_name for c in test_cams]}")
    scene.train_cameras[1.0] = train_cams
    scene.test_cameras[1.0] = test_cams
    save_split(os.path.join(dataset.model_path, "split.json"), train_cams, test_cams, args.split)
    print(f"[halogs] {len(train_cams)} train / {len(test_cams)} test views ({args.split})")

    target_gain = None
    if lit:
        p0 = illum_init_from(train_cams, args.theta)
        gaussians.set_illum_init(p0)
        if args.target_gain == "scene":
            target_gain = scene_gain(train_cams, args.theta)
        print(f"[litalite] theta={args.theta} illumination init {p0:.4f} target gain "
              f"{'per image' if target_gain is None else f'{target_gain:.3f} (scene)'}  "
              f"hue-safe={not args.legacy_clip} dense_from={args.dense_from}")
    clean_refs = load_clean_refs(args.clean_gt_dir) if args.clean_gt_dir else None
    tone = tone_opt = None
    if lit and args.cam_model == "colour_curve":
        tone = ColourCurveCamera([c.image_name for c in train_cams], args.cam_knots).to(device)
        tone_opt = torch.optim.Adam(tone.parameters(), lr=args.tone_lr)
    elif lit and (args.tone_curve or args.cam_model == "power"):
        tone = ToneCurve([c.image_name for c in train_cams]).to(device)
        tone_opt = torch.optim.Adam(tone.parameters(), lr=args.tone_lr)
    target_cache = {}
    n2n_net, n2n_s = None, 1.0
    if lit and args.denoise_target == "n2n_unet":
        from halogs.denoise_n2n import apply as n2n_apply, count_params, n2n_scale, train_n2n
        dark = [c.original_image.to(device) for c in train_cams]  # training views only
        n2n_s = n2n_scale(dark)
        t_n2n = time.time()
        n2n_net = train_n2n([(d * n2n_s).clamp(0, 1) for d in dark], iters=args.n2n_iters, crop=args.n2n_crop,
                            batch=args.n2n_batch, width=args.n2n_width, lr=args.n2n_lr, device=device, seed=args.seed,
                            log_every=0 if args.quiet else 250)
        torch.save(n2n_net.state_dict(), os.path.join(dataset.model_path, "n2n_unet.pt"))
        print(f"[litalite] U-Net denoiser ({count_params(n2n_net) / 1e6:.2f} M params) trained on "
              f"{len(dark)} training images in {(time.time() - t_n2n) / 60:.1f} min (scale {n2n_s:.2f})")
        del dark

    wb_gains = grayworld_gains(train_cams) if (lit and args.white_balance == "grayworld") else None
    if lit:
        print(f"[litalite] white balance={args.white_balance}"
              f"{'' if wb_gains is None else ' gains R/G/B ' + str([round(float(v), 3) for v in wb_gains])}"
              f"  target gamma={args.target_gamma}")

    def lit_target(cam, gt):
        if cam.image_name not in target_cache:
            src = gt
            if n2n_net is not None:  # denoise the dark photo at a mid-grey scale, then brighten as usual
                src = (n2n_apply(n2n_net, (gt * n2n_s).clamp(0, 1)) / n2n_s).clamp(0, 1)
            t = brighten_target(src, args.theta, target_gain, hue_safe=not args.legacy_clip,
                                wb_gains=wb_gains, gamma=args.target_gamma)
            if args.denoise_target == "bilateral":
                with torch.no_grad():
                    t = bilateral(t, args.dn_radius, args.dn_sigma_s, args.dn_sigma_r)
            target_cache[cam.image_name] = t.detach()
        return target_cache[cam.image_name]
    if lit:
        print(f"[litalite] illumination channels={args.illum_channels} tone_curve={bool(tone)} "
              f"denoise_target={args.denoise_target}")

    gaussians.training_setup(opt)

    # ---- glow model ------------------------------------------------------------------
    cam0 = train_cams[0]
    H, W = int(cam0.image_height), int(cam0.image_width)
    radius = args.kernel_radius if args.kernel_radius > 0 else int(min(max(0.1 * max(H, W), 32), 160))
    focal = focal_px_from_camera(cam0)
    halo = None
    halo_opt = None
    if glow_mode:
        halo = HaloModel(kernel_kind=args.kernel, radius=radius, focal_px=focal,
                         per_view_exposure=args.per_view_exposure,
                         train_names=[c.image_name for c in train_cams],
                         learn_kernel=not args.freeze_kernel,
                         gamma_min=args.gamma_min, gamma_init=args.gamma_init, w_init=args.w_init).to(device)
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
                  "q", "T", "gamma_px", "w_lens", "gain_mean", "peak_mem_gb", "sec_per_iter",
                  "l_exp", "l_rec", "illum_mean", "tone"])

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

        use_glow = glow_mode and iteration > args.glow_from_iter
        gt = cam.original_image.to(device)
        l_exp = l_rec = None
        if lit:
            pkg = render_litalite(render, cam, gaussians, pipe, bg, halo, use_glow=use_glow, tone=tone)
            target = lit_target(cam, gt)
            l_exp, _ = photometric_loss(pkg["bright"], target, opt.lambda_dssim)
            l_rec, l1 = photometric_loss(pkg["render"], gt, opt.lambda_dssim)
            w_ps = args.w_exp * pseudo_weight(iteration, *args.pseudo_fade)
            loss = w_ps * l_exp + args.w_rec * l_rec
            if args.w_tv > 0:
                loss = loss + args.w_tv * glow_tv(pkg["bright"])
            if iteration > args.prior_from:
                if args.w_expo > 0:
                    loss = loss + args.w_expo * exposure_loss(pkg["bright"], args.expo_target, args.expo_patch)
                if args.w_col > 0:
                    loss = loss + args.w_col * colour_constancy_loss(pkg["bright"])
            if isinstance(tone, ColourCurveCamera) and args.w_tie > 0:
                loss = loss + args.w_tie * tone.tie_loss()
        else:
            pkg = render_halo(render, cam, gaussians, pipe, bg, halo, mode="halo" if use_glow else "vanilla")
            loss, l1 = photometric_loss(pkg["render"], gt, opt.lambda_dssim, valid_mask(cam))
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
            radii = pkg["radii"]
            if iteration < opt.densify_until_iter:
                branches = pkg.get("dense", [(pkg["viewspace_points"], pkg["visibility_filter"], pkg["radii"])])
                if args.dense_from == "bright":
                    branches = branches[:1]
                for vsp, vis, r in branches:
                    gaussians.max_radii2D[vis] = torch.max(gaussians.max_radii2D[vis], r[vis])
                    gaussians.add_densification_stats(vsp, vis)
                if len(branches) > 1:
                    radii = torch.max(branches[0][2], branches[1][2])
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
                if tone_opt is not None:
                    tone_opt.step()
                    tone_opt.zero_grad(set_to_none=True)

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
                              hd.get("gain_mean", ""), f"{peak:.3f}", f"{(now - last_t) / args.log_every:.4f}",
                              f"{l_exp.item():.6f}" if l_exp is not None else "",
                              f"{l_rec.item():.6f}" if l_rec is not None else "",
                              f"{gaussians.get_illum.mean().item():.4f}" if lit else "",
                              json.dumps(tone.describe()) if tone is not None else ""])
                log_f.flush()
                last_t = now

            # held-out evaluation (composite vs observed reference)
            if iteration % args.test_every == 0 or iteration == opt.iterations:
                evaluate(iteration, test_cams, gaussians, pipe, bg, halo, args, dataset.model_path, use_glow, clean_refs, tone)

            if iteration in args.save_iterations or iteration == opt.iterations:
                print(f"\n[ITER {iteration}] saving")
                scene.save(iteration)
                pc_dir = os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}")
                gaussians.save_emission(os.path.join(pc_dir, "emission.pt"))
                if halo is not None:
                    halo.save(os.path.join(pc_dir, "halo.pt"))
                if tone is not None:
                    tone.save(os.path.join(pc_dir, "tone.pt"))

    log_f.close()
    print(f"\n[halogs] done in {(time.time() - t0) / 60:.1f} min -> {dataset.model_path}")


def load_clean_refs(d: str) -> dict:
    """image stem -> path of a normal-light reference (pairing by name, never by order)."""
    exts = (".png", ".jpg", ".jpeg", ".JPG", ".PNG")
    refs = {os.path.splitext(n)[0]: os.path.join(d, n) for n in sorted(os.listdir(d)) if n.endswith(exts)}
    print(f"[halogs] {len(refs)} clean references in {d}")
    return refs


def _load_ref(path, like):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1).to(like.device)
    if t.shape != like.shape:
        t = torch.nn.functional.interpolate(t[None], size=like.shape[1:], mode="area")[0]
    return t


@torch.no_grad()
def evaluate(iteration, test_cams, gaussians, pipe, bg, halo, args, model_path, glow_on, clean_refs=None, tone=None):
    if not test_cams:
        return
    lit = args.mode.startswith("litalite")
    rows, crow = [], []
    for cam in test_cams:
        if lit:
            pkg = render_litalite(render, cam, gaussians, pipe, bg, halo, use_glow=glow_on, tone=tone)
        else:
            pkg = render_halo(render, cam, gaussians, pipe, bg, halo, mode="halo" if glow_on else "vanilla")
        gt = cam.original_image.to(pkg["render"].device)
        gm = glow_mask(gt, args.mask_thresh, args.mask_radius)
        m = all_metrics(pkg["render"].clamp(0, 1), gt, gm, with_lpips=False)
        m["view"] = cam.image_name
        rows.append(m)
        if clean_refs is not None:
            out = pkg["bright"] if lit else pkg["scene"]
            stem = os.path.splitext(cam.image_name)[0]
            if stem in clean_refs:
                c = all_metrics(out.clamp(0, 1), _load_ref(clean_refs[stem], out), gm, with_lpips=False)
                crow.append(c)
    keys = [k for k in rows[0] if k != "view"]
    mean = {k: float(_nanmean([r[k] for r in rows])) for k in keys}
    print(f"\n[ITER {iteration}] test  PSNR {mean['psnr']:.2f}  SSIM {mean['ssim']:.3f}  "
          f"| glow PSNR {mean['psnr_glow']:.2f}  non-glow PSNR {mean['psnr_nonglow']:.2f}  (glow frac {mean['glow_frac']:.3f})")
    if crow:
        cm = {f"clean_{k}": float(_nanmean([r[k] for r in crow])) for k in crow[0]}
        cm["clean_n"] = len(crow)
        mean.update(cm)
        print(f"[ITER {iteration}] {'bright' if lit else 'scene'} vs clean reference  "
              f"PSNR {cm['clean_psnr']:.2f}  SSIM {cm['clean_ssim']:.3f}  (n={len(crow)})")
    elif clean_refs is not None:
        print(f"[ITER {iteration}] WARNING: no clean reference matched any test view name")
    with open(os.path.join(model_path, "test_curve.jsonl"), "a") as f:
        f.write(json.dumps({"iter": iteration, **mean}) + "\n")


if __name__ == "__main__":
    main()
