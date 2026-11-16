"""Emissive Gaussian model: standard 3DGS Gaussians plus a per-Gaussian emission.

The emission attribute lives *inside* the optimiser's parameter groups and goes
through exactly the same prune / clone / split path as every other attribute,
so a surviving Gaussian always keeps its own emission, children inherit their
parent's, and Adam moments are kept for surviving rows.  (This is the failure
mode found in the earlier LitaLite illumination implementation, where a
side-car tensor was resynchronised by *count* rather than by identity.)

Works with both the 2023 graphdeco / camenduru fork (no ``tmp_radii``, no
exposure) and the current official repository (``tmp_radii``, exposure).
``gaussian-splatting`` must be importable (``scene``, ``utils`` on sys.path).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from scene.gaussian_model import GaussianModel  # type: ignore
from utils.general_utils import build_rotation  # type: ignore


class EmissiveGaussianModel(GaussianModel):
    """GaussianModel + emission (N,3), activation softplus, stored as raw logits."""

    EMISSION_INIT_RAW = -6.0  # softplus(-6) ~ 0.0025: start essentially dark

    def __init__(self, sh_degree: int, *args, emission_lr: float = 0.01,
                 with_illum: bool = False, illum_lr: float = 0.01, illum_init: float = 0.5, **kwargs):
        super().__init__(sh_degree, *args, **kwargs)
        self._emission = torch.empty(0)
        self.emission_lr = float(emission_lr)
        # LitaLite illumination: one scalar per Gaussian, sigmoid-activated, tracked by identity
        # exactly like emission (fixes the count-based resync of the original litalite.py).
        self.with_illum = bool(with_illum)
        self.illum_lr = float(illum_lr)
        self.illum_init = float(illum_init)
        self._illum = torch.empty(0)

    # ------------------------------------------------------------------ accessors
    @property
    def get_emission(self) -> torch.Tensor:
        return F.softplus(self._emission)

    @property
    def device(self):
        return self._xyz.device

    @property
    def get_illum(self) -> torch.Tensor:
        return torch.sigmoid(self._illum)

    def _init_emission(self, n: int, device=None) -> None:
        device = device or (self._xyz.device if self._xyz.numel() else "cuda")
        self._emission = nn.Parameter(torch.full((n, 3), self.EMISSION_INIT_RAW, device=device).requires_grad_(True))

    def _init_illum(self, n: int, device=None) -> None:
        device = device or (self._xyz.device if self._xyz.numel() else "cuda")
        p = min(max(self.illum_init, 1e-3), 1 - 1e-3)
        raw = float(torch.logit(torch.tensor(p)))
        self._illum = nn.Parameter(torch.full((n, 1), raw, device=device).requires_grad_(True))

    def set_illum_init(self, p: float) -> None:
        """Reset every Gaussian's illumination to ``p`` (call before ``training_setup``)."""
        self.illum_init = float(p)
        if self.with_illum and self._xyz.numel():
            self._init_illum(self._xyz.shape[0])

    # ------------------------------------------------------------------ creation
    def create_from_pcd(self, *args, **kwargs):
        super().create_from_pcd(*args, **kwargs)
        self._init_emission(self._xyz.shape[0])
        if self.with_illum:
            self._init_illum(self._xyz.shape[0])

    # ------------------------------------------------------------------ checkpoints
    def capture(self):
        base = super().capture()
        illum = self._illum.detach().clone() if self.with_illum else None
        return (base, self._emission.detach().clone(), illum)

    def restore(self, model_args, training_args):
        base, emission = model_args[0], model_args[1]
        illum = model_args[2] if len(model_args) > 2 else None
        super().restore(base, training_args)
        self._emission = nn.Parameter(emission.to(self._xyz.device).requires_grad_(True))
        if self.with_illum:
            if illum is None:
                self._init_illum(self._xyz.shape[0])
            else:
                self._illum = nn.Parameter(illum.to(self._xyz.device).requires_grad_(True))
        # the optimizer state restored by the parent does not know about emission:
        self._add_emission_group()

    # ------------------------------------------------------------------ optimiser
    def training_setup(self, training_args):
        super().training_setup(training_args)
        if self._emission.numel() == 0:
            self._init_emission(self._xyz.shape[0])
        if self.with_illum and self._illum.numel() == 0:
            self._init_illum(self._xyz.shape[0])
        self._add_emission_group()

    def _add_emission_group(self):
        names = {g.get("name") for g in self.optimizer.param_groups}
        if "emission" not in names:
            self.optimizer.add_param_group({"params": [self._emission], "lr": self.emission_lr, "name": "emission"})
        if self.with_illum and "illum" not in names:
            self.optimizer.add_param_group({"params": [self._illum], "lr": self.illum_lr, "name": "illum"})

    # ------------------------------------------------------------------ lifecycle
    def prune_points(self, mask):
        valid = ~mask
        optimizable_tensors = self._prune_optimizer(valid)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._emission = optimizable_tensors["emission"]
        if "illum" in optimizable_tensors:
            self._illum = optimizable_tensors["illum"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid]
        self.denom = self.denom[valid]
        self.max_radii2D = self.max_radii2D[valid]
        if getattr(self, "tmp_radii", None) is not None:
            self.tmp_radii = self.tmp_radii[valid]

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities,
                              new_scaling, new_rotation, new_emission, new_tmp_radii=None, new_illum=None):
        d = {"xyz": new_xyz, "f_dc": new_features_dc, "f_rest": new_features_rest,
             "opacity": new_opacities, "scaling": new_scaling, "rotation": new_rotation,
             "emission": new_emission}
        if self.with_illum:
            assert new_illum is not None, "illumination must be passed for new Gaussians"
            d["illum"] = new_illum
        optimizable_tensors = self._cat_tensors_to_optimizer_selective(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._emission = optimizable_tensors["emission"]
        if "illum" in optimizable_tensors:
            self._illum = optimizable_tensors["illum"]
        if getattr(self, "tmp_radii", None) is not None and new_tmp_radii is not None:
            self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        dev = self._xyz.device
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device=dev)
        self.denom = torch.zeros((self._xyz.shape[0], 1), device=dev)
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device=dev)

    _PER_GAUSSIAN_GROUPS = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation", "emission", "illum")

    def _cat_tensors_to_optimizer_selective(self, tensors_dict):
        """Like the parent's cat_tensors_to_optimizer but only touches per-Gaussian groups."""
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            name = group.get("name")
            if name not in self._PER_GAUSSIAN_GROUPS:
                continue
            assert len(group["params"]) == 1
            ext = tensors_dict[name]
            stored = self.optimizer.state.get(group["params"][0], None)
            if stored is not None:
                stored["exp_avg"] = torch.cat((stored["exp_avg"], torch.zeros_like(ext)), dim=0)
                stored["exp_avg_sq"] = torch.cat((stored["exp_avg_sq"], torch.zeros_like(ext)), dim=0)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], ext), dim=0).requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], ext), dim=0).requires_grad_(True))
            optimizable_tensors[name] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """Parent version prunes *every* group; skip groups that are not per-Gaussian (exposure)."""
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            name = group.get("name")
            if name not in self._PER_GAUSSIAN_GROUPS:
                continue
            stored = self.optimizer.state.get(group["params"][0], None)
            if stored is not None:
                stored["exp_avg"] = stored["exp_avg"][mask]
                stored["exp_avg_sq"] = stored["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group["params"][0]] = stored
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
            optimizable_tensors[name] = group["params"][0]
        return optimizable_tensors

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        dev = self._xyz.device
        n_init = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init), device=dev)
        padded_grad[: grads.shape[0]] = grads.squeeze()
        sel = padded_grad >= grad_threshold
        sel = torch.logical_and(sel, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[sel].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=dev)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[sel]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[sel].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[sel].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[sel].repeat(N, 1)
        new_features_dc = self._features_dc[sel].repeat(N, 1, 1)
        new_features_rest = self._features_rest[sel].repeat(N, 1, 1)
        new_opacity = self._opacity[sel].repeat(N, 1)
        new_emission = self._emission[sel].repeat(N, 1)  # children inherit the parent's emission
        new_illum = self._illum[sel].repeat(N, 1) if self.with_illum else None  # ... and illumination
        new_tmp_radii = self.tmp_radii[sel].repeat(N) if getattr(self, "tmp_radii", None) is not None else None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity,
                                   new_scaling, new_rotation, new_emission, new_tmp_radii, new_illum)

        prune_filter = torch.cat((sel, torch.zeros(N * int(sel.sum()), device=dev, dtype=torch.bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        sel = torch.norm(grads, dim=-1) >= grad_threshold
        sel = torch.logical_and(sel, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
        new_tmp_radii = self.tmp_radii[sel] if getattr(self, "tmp_radii", None) is not None else None
        self.densification_postfix(self._xyz[sel], self._features_dc[sel], self._features_rest[sel],
                                   self._opacity[sel], self._scaling[sel], self._rotation[sel],
                                   self._emission[sel], new_tmp_radii,
                                   self._illum[sel] if self.with_illum else None)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.tmp_radii = radii  # None on the old fork, (N,) on the current one
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_vs = self.max_radii2D > max_screen_size
            big_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_vs), big_ws)
        self.prune_points(prune_mask)
        self.tmp_radii = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ i/o
    def save_emission(self, path: str) -> None:
        blob = {"emission_raw": self._emission.detach().cpu()}
        if self.with_illum:
            blob["illum_raw"] = self._illum.detach().cpu()
        torch.save(blob, path)

    def load_emission(self, path: str, strict: bool = True) -> None:
        blob = torch.load(path, map_location="cpu")
        e = blob["emission_raw"]
        if e.shape[0] != self._xyz.shape[0]:
            msg = f"emission rows {e.shape[0]} != gaussians {self._xyz.shape[0]}"
            if strict:
                raise RuntimeError(msg)
            print("[halogs] WARNING:", msg, "- resetting emission")
            self._init_emission(self._xyz.shape[0])
            if self.with_illum:
                self._init_illum(self._xyz.shape[0])
            return
        self._emission = nn.Parameter(e.to(self._xyz.device).requires_grad_(True))
        if self.with_illum:
            if "illum_raw" in blob:
                self._illum = nn.Parameter(blob["illum_raw"].to(self._xyz.device).requires_grad_(True))
            elif strict:
                raise RuntimeError("checkpoint has no illumination but the model expects one")
            else:
                self._init_illum(self._xyz.shape[0])

    def load_ply(self, path, *args, **kwargs):
        super().load_ply(path, *args, **kwargs)
        side = os.path.join(os.path.dirname(path), "emission.pt")
        if os.path.exists(side):
            self.load_emission(side, strict=False)
        else:
            self._init_emission(self._xyz.shape[0])
            if self.with_illum:
                self._init_illum(self._xyz.shape[0])

    def emission_stats(self) -> dict:
        with torch.no_grad():
            e = self.get_emission
            lum = e.mean(1)
            return {"n_gaussians": int(e.shape[0]),
                    "emission_mean": float(lum.mean()),
                    "emission_max": float(lum.max()) if lum.numel() else 0.0,
                    "frac_emitters_gt_0.1": float((lum > 0.1).float().mean()) if lum.numel() else 0.0}
