#!/usr/bin/env python
"""End-to-end CPU smoke test: train -> render -> eval on a tiny synthetic COLMAP
scene with the CUDA rasterizer replaced by a naive differentiable splatter.

This does NOT validate rendering quality - only that the whole pipeline
(argument plumbing, Scene loading, explicit split, two-pass render, losses,
densification lifecycle, checkpoint files, manifest, metrics) runs without a
GPU.  Usage:

  GS_ROOT=/path/to/gaussian-splatting python tools/smoke_cpu.py [--iters 60]
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import tempfile
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


# ----------------------------------------------------------------------------- patches
def patch_torch_cuda_to_cpu():
    """Map every explicit device='cuda' / .cuda() in gaussian-splatting onto the CPU."""
    def _fix_kw(kw):
        d = kw.get("device")
        if d is not None and "cuda" in str(d):
            kw["device"] = "cpu"
        return kw

    for name in ("zeros", "ones", "empty", "full", "tensor", "rand", "randn", "arange", "eye", "linspace", "as_tensor", "normal",
                 "zeros_like", "ones_like", "empty_like", "full_like", "rand_like", "randn_like", "randint"):
        orig = getattr(torch, name)

        def wrapped(*a, _orig=orig, **kw):
            return _orig(*a, **_fix_kw(kw))
        setattr(torch, name, wrapped)

    torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore
    _to = torch.Tensor.to
    _dev = torch.device  # real class, captured before it is shadowed below

    def to(self, *a, **k):
        a = tuple("cpu" if (isinstance(x, (str, _dev)) and "cuda" in str(x)) else x for x in a)
        if "device" in k and "cuda" in str(k["device"]):
            k["device"] = "cpu"
        return _to(self, *a, **k)
    torch.Tensor.to = to  # type: ignore
    _mod_to = torch.nn.Module.to

    def module_to(self, *a, **k):
        a = tuple("cpu" if (isinstance(x, (str, _dev)) and "cuda" in str(x)) else x for x in a)
        if "device" in k and "cuda" in str(k["device"]):
            k["device"] = "cpu"
        return _mod_to(self, *a, **k)
    torch.nn.Module.to = module_to  # type: ignore
    # torch.device("cuda:0") objects are harmless without a GPU as long as nothing is allocated on them
    torch.cuda.set_device = lambda *a, **k: None
    torch.cuda.empty_cache = lambda: None
    torch.cuda.max_memory_allocated = lambda *a, **k: 0
    torch.cuda.synchronize = lambda *a, **k: None


def install_mock_rasterizer(gs_root):
    """A tiny additive isotropic splatter standing in for diff_gaussian_rasterization."""
    with open(os.path.join(gs_root, "gaussian_renderer", "__init__.py")) as f:
        three_outputs = "depth_image" in f.read()

    class Settings:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class Rasterizer:
        def __init__(self, raster_settings):
            self.s = raster_settings

        def __call__(self, means3D, means2D, shs=None, colors_precomp=None, opacities=None,
                     scales=None, rotations=None, cov3D_precomp=None, dc=None):
            s = self.s
            H, W = int(s.image_height), int(s.image_width)
            N = means3D.shape[0]
            hom = torch.cat([means3D, torch.ones(N, 1)], 1) @ s.projmatrix  # (N,4) row-vector convention
            w = hom[:, 3:4].clamp_min(1e-4)
            ndc = hom[:, :3] / w
            px = ((ndc[:, 0] + 1) * 0.5 * W) + means2D[:, 0]
            py = ((ndc[:, 1] + 1) * 0.5 * H) + means2D[:, 1]
            if colors_precomp is None:
                col = (shs[:, 0, :] * 0.28209479177387814 + 0.5).clamp_min(0)  # SH DC only
            else:
                col = colors_precomp
            sig = (scales.mean(1) if scales is not None else torch.full((N,), 0.05)) * 0.5 * W / (w[:, 0] * math.tan(s.tanfovx) + 1e-6)
            sig = sig.clamp(0.5, 20.0)
            yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
            vis = (hom[:, 3] > 0.01) & (px > -3 * sig) & (px < W + 3 * sig) & (py > -3 * sig) & (py < H + 3 * sig)
            img = torch.zeros(3, H, W) + s.bg.view(3, 1, 1)
            idx = torch.nonzero(vis).flatten()
            for k in idx.tolist():
                d2 = (xx - px[k]) ** 2 + (yy - py[k]) ** 2
                wgt = opacities[k] * torch.exp(-0.5 * d2 / (sig[k] ** 2))
                img = img + wgt.unsqueeze(0) * col[k].view(3, 1, 1)
            radii = (3 * sig).int() * vis.int()
            if three_outputs:
                return img, radii, torch.zeros(1, H, W)
            return img, radii

    mod = types.ModuleType("diff_gaussian_rasterization")
    mod.GaussianRasterizationSettings = Settings
    mod.GaussianRasterizer = Rasterizer
    sys.modules["diff_gaussian_rasterization"] = mod

    sk = types.ModuleType("simple_knn")
    skc = types.ModuleType("simple_knn._C")

    def distCUDA2(x):
        d = torch.cdist(x, x)
        d.fill_diagonal_(float("inf"))
        return d.topk(3, largest=False).values.pow(2).mean(1)
    skc.distCUDA2 = distCUDA2
    sk._C = skc
    sys.modules["simple_knn"] = sk
    sys.modules["simple_knn._C"] = skc


# ----------------------------------------------------------------------------- data
def write_tiny_colmap_scene(root, n_views=6, H=24, W=32, n_pts=60):
    from PIL import Image
    rng = np.random.default_rng(0)
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    sp = os.path.join(root, "sparse", "0")
    os.makedirs(sp, exist_ok=True)
    f = 0.9 * W
    with open(os.path.join(sp, "cameras.txt"), "w") as fc:
        fc.write(f"1 PINHOLE {W} {H} {f} {f} {W / 2} {H / 2}\n")
    pts = rng.normal(0, 0.4, (n_pts, 3))
    with open(os.path.join(sp, "points3D.txt"), "w") as fp:
        for i, p in enumerate(pts):
            fp.write(f"{i + 1} {p[0]} {p[1]} {p[2]} 128 128 128 0.5 1 1 2 1\n")
    lines = []
    for i in range(n_views):
        ang = 2 * math.pi * i / n_views * 0.25 - 0.4
        C = np.array([3 * math.sin(ang), 0.2, 3 * math.cos(ang)])
        fwd = -C / np.linalg.norm(C)
        up = np.array([0, 1.0, 0])
        right = np.cross(fwd, up); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        Rw2c = np.stack([right, -up, fwd])  # OpenCV: x right, y down, z forward
        t = -Rw2c @ C
        # quaternion from rotation
        from data.llff_to_colmap import rotmat2qvec
        q = rotmat2qvec(Rw2c)
        name = f"v{i:02d}.png"
        lines.append(f"{i + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {name}\n\n")
        img = np.full((H, W, 3), 25, np.uint8)
        img[H // 2 - 2: H // 2 + 2, W // 2 - 2 + i: W // 2 + 2 + i] = 255  # a 'lamp'
        img[:, :, 2] += (rng.random((H, W)) * 20).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(root, "images", name))
        os.makedirs(os.path.join(root, "high"), exist_ok=True)  # fake normal-light references for litalite
        Image.fromarray(np.clip(img.astype(np.int32) * 3, 0, 255).astype(np.uint8)).save(os.path.join(root, "high", name))
    with open(os.path.join(sp, "images.txt"), "w") as fi:
        fi.writelines(lines)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    from halogs.gs_path import find_gs_root
    gs_root = find_gs_root(os.environ.get("GS_ROOT"))
    patch_torch_cuda_to_cpu()
    install_mock_rasterizer(gs_root)

    work = tempfile.mkdtemp(prefix="halogs_smoke_")
    scene = os.path.join(work, "scene")
    write_tiny_colmap_scene(scene)
    print("[smoke] scene at", scene)

    import train_glow, render_glow, eval_glow  # noqa: E402  (after patches)

    for run in ("halo", "masked", "vanilla", "litalite", "litalite_halo", "litalite_plus", "litalite_unet", "litalite_nodn", "litalite_relight"):
        mode = "litalite" if run in ("litalite_plus", "litalite_unet", "litalite_nodn", "litalite_relight") else run
        out = os.path.join(work, "out", "tiny", run)
        sys.argv = ["train_glow.py", "-s", scene, "-m", out, "--mode", mode, "--iterations", str(args.iters),
                    "--densify_from_iter", "10", "--densification_interval", "20", "--densify_until_iter", str(args.iters),
                    "--opacity_reset_interval", "40", "--glow_from_iter", "5", "--test_every", "30", "--log_every", "10",
                    "--save_iterations", str(args.iters), "--kernel_radius", "6", "--mask_radius", "4", "--quiet",
                    "--data_device", "cpu", "--gs_root", gs_root]
        if mode.startswith("litalite"):
            sys.argv += ["--clean_gt_dir", os.path.join(scene, "high")]
        if run == "litalite_plus":
            sys.argv += ["--illum_channels", "3", "--tone_curve", "--denoise_target", "bilateral",
                         "--split", "views", "--test_views", "1", "4",
                         "--white_balance", "grayworld", "--target_gamma", "0.8"]
        if run == "litalite_unet":
            sys.argv += ["--illum_channels", "3", "--cam_model", "colour_curve", "--cam_knots", "8",
                         "--w_expo", "0.5", "--w_col", "0.05", "--prior_from", "5", "--pseudo_fade", "10", "40",
                         "--target_curve", "nova", "--nova_radius", "4", "--w_sh_noise", "1.0", "--w_spa", "1.0", "--w_ill", "0.1", "--rel_l1",
                         "--denoise_target", "n2n_unet",
                         "--n2n_iters", "3", "--n2n_crop", "16", "--n2n_batch", "2", "--n2n_width", "8", "--dump_targets", "2"]
        if run == "litalite_relight":  # Idea A: 3D relighting of the learned illumination field
            sys.argv += ["--illum_channels", "3", "--target_curve", "nova", "--nova_radius", "4", "--w_sh_noise", "1.0",
                         "--denoise_target", "n2n_unet", "--n2n_iters", "3", "--n2n_crop", "16", "--n2n_batch", "2", "--n2n_width", "8",
                         "--relight3d", "--relight_from", "20", "--relight_every", "20", "--w_anchor", "0.1"]
        if run == "litalite_nodn":  # v4 without the denoiser: SH penalty uses the guided-filter noise map
            sys.argv += ["--illum_channels", "3", "--target_curve", "nova", "--nova_radius", "4", "--w_sh_noise", "1.0",
                         "--sh_noise_src", "guided", "--denoise_target", "none", "--dump_targets", "2", "--nova_mean_rule", "reinhard"]
        train_glow.main()
        if run == "litalite_plus":
            import json as _js
            assert _js.load(open(os.path.join(out, "split.json")))["test"] == ["v01.png", "v04.png"]
            assert os.path.exists(os.path.join(out, f"point_cloud/iteration_{args.iters}/tone.pt"))
            assert torch.load(os.path.join(out, f"point_cloud/iteration_{args.iters}/emission.pt"))["illum_raw"].shape[1] == 3
        if run == "litalite_unet":
            assert os.path.exists(os.path.join(out, "n2n_unet.pt"))
        for fn in ("split.json", "config.json", "log.csv", "cfg_args", f"point_cloud/iteration_{args.iters}/point_cloud.ply",
                   f"point_cloud/iteration_{args.iters}/emission.pt"):
            assert os.path.exists(os.path.join(out, fn)), fn
        if mode in ("halo", "litalite_halo"):
            assert os.path.exists(os.path.join(out, f"point_cloud/iteration_{args.iters}/halo.pt"))
        if mode.startswith("litalite"):
            blob = torch.load(os.path.join(out, f"point_cloud/iteration_{args.iters}/emission.pt"))
            assert "illum_raw" in blob and blob["illum_raw"].shape[0] == blob["emission_raw"].shape[0]
            import json as _json
            last = [_json.loads(l) for l in open(os.path.join(out, "test_curve.jsonl"))][-1]
            assert "clean_psnr" in last, last

        sys.argv = ["render_glow.py", "-m", out, "--iteration", str(args.iters), "--quiet", "--gs_root", gs_root]
        render_glow.main()
        rd = os.path.join(out, "renders", f"iteration_{args.iters}", "test")
        assert os.path.exists(os.path.join(rd, "manifest.json"))

        sys.argv = ["eval_glow.py", "-m", out, "--iteration", str(args.iters), "--no_lpips", "--mask_radius", "4", "--device", "cpu"]
        if mode.startswith("litalite"):
            assert os.path.isdir(os.path.join(rd, "bright"))
            sys.argv += ["--clean_gt_dir", os.path.join(scene, "high")]
        eval_glow.main()
        assert os.path.exists(os.path.join(rd, "metrics.json"))
        if mode.startswith("litalite"):
            assert os.path.exists(os.path.join(rd, "metrics_clean.json"))

    import aggregate_results
    sys.argv = ["aggregate_results.py", os.path.join(work, "out"), "--iteration", str(args.iters)]
    aggregate_results.main()
    print("\n[smoke] OK ->", work)
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
