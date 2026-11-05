"""EmissiveLitaModel: LITA-GS GaussianModel + per-Gaussian emission.

LITA-GS (built on Gaussian-DK) already carries depth / prior / noise /
illumination features per Gaussian and rasterises them as extra channels.
We add a 3-channel emission attribute that follows the same identity-safe
prune / clone / split path, exactly like ``halogs.gaussians`` does for vanilla
3DGS.  LITA-GS's ``_prune_optimizer`` / ``cat_tensors_to_optimizer`` iterate
over every optimizer group except ``tonemapper`` / ``denoiser``, so registering
``f_emission`` as a group is enough for pruning; the clone / split /
postfix methods have fixed argument lists and are re-implemented here.

Requires the LITA-GS checkout on sys.path (see ``halogs.gs_path.add_lita_to_path``).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from scene.gaussian_model import GaussianModel as LitaGaussianModel  # type: ignore
from utils.general_utils import build_rotation  # type: ignore


class EmissiveLitaModel(LitaGaussianModel):
    EMISSION_INIT_RAW = -6.0

    def __init__(self, *args, emission_lr: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self._features_emission = torch.empty(0)
        self.emission_lr = float(emission_lr)

    # ------------------------------------------------------------------ accessors
    @property
    def get_emission(self) -> torch.Tensor:
        return F.softplus(self._features_emission)

    def _init_emission(self, n: int, device=None):
        device = device or (self._xyz.device if self._xyz.numel() else "cuda")
        self._features_emission = nn.Parameter(
            torch.full((n, 3), self.EMISSION_INIT_RAW, device=device).contiguous().requires_grad_(True))

    def create_from_pcd(self, *args, **kwargs):
        super().create_from_pcd(*args, **kwargs)
        self._init_emission(self._xyz.shape[0])

    # ------------------------------------------------------------------ optimiser
    def training_setup(self, training_args):
        super().training_setup(training_args)
        if self._features_emission.numel() == 0:
            self._init_emission(self._xyz.shape[0])
        if not any(g.get("name") == "f_emission" for g in self.optimizer.param_groups):
            self.optimizer.add_param_group({"params": [self._features_emission], "lr": self.emission_lr, "name": "f_emission"})

    # ------------------------------------------------------------------ checkpoints
    def capture(self):
        return (super().capture(), self._features_emission.detach().clone())

    def restore(self, model_args, training_args):
        base, emission = model_args
        self._features_emission = nn.Parameter(emission.cuda().requires_grad_(True))
        super().restore(base, training_args)

    # ------------------------------------------------------------------ lifecycle
    def prune_points(self, mask):
        valid = ~mask
        ot = self._prune_optimizer(valid)
        self._xyz = ot["xyz"]
        self._features_dc = ot["f_dc"]
        self._features_rest = ot["f_rest"]
        self._features_depth = ot["f_depth"]
        self._features_prior = ot["f_prior"]
        self._features_noise = ot["f_noise"]
        self._features_illu = ot["f_illu"]
        self._features_emission = ot["f_emission"]
        self._opacity = ot["opacity"]
        self._scaling = ot["scaling"]
        self._rotation = ot["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid]
        self.denom = self.denom[valid]
        self.max_radii2D = self.max_radii2D[valid]
        self.min_radii2D = self.min_radii2D[valid]

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_features_depth,
                              new_features_prior, new_features_noise, new_features_illu, new_opacities,
                              new_scaling, new_rotation, new_emission=None):
        if new_emission is None:  # defensive: a parent call path we did not override
            new_emission = torch.full((new_xyz.shape[0], 3), self.EMISSION_INIT_RAW, device=new_xyz.device)
        d = {"xyz": new_xyz, "f_dc": new_features_dc, "f_rest": new_features_rest,
             "f_depth": new_features_depth, "f_prior": new_features_prior, "f_noise": new_features_noise,
             "f_illu": new_features_illu, "f_emission": new_emission,
             "opacity": new_opacities, "scaling": new_scaling, "rotation": new_rotation}
        ot = self.cat_tensors_to_optimizer(d)
        self._xyz = ot["xyz"]
        self._features_dc = ot["f_dc"]
        self._features_rest = ot["f_rest"]
        self._features_depth = ot["f_depth"]
        self._features_prior = ot["f_prior"]
        self._features_noise = ot["f_noise"]
        self._features_illu = ot["f_illu"]
        self._features_emission = ot["f_emission"]
        self._opacity = ot["opacity"]
        self._scaling = ot["scaling"]
        self._rotation = ot["rotation"]
        dev = self._xyz.device
        n = self._xyz.shape[0]
        self.xyz_gradient_accum = torch.zeros((n, 1), device=dev)
        self.xyz_gradient_accum_abs = torch.zeros((n, 1), device=dev)
        self.xyz_gradient_accum_abs_max = torch.zeros((n, 1), device=dev)
        self.denom = torch.zeros((n, 1), device=dev)
        self.max_radii2D = torch.zeros((n), device=dev)
        self.min_radii2D = torch.zeros((n), device=dev)

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
        self.densification_postfix(
            new_xyz, self._features_dc[sel].repeat(N, 1, 1), self._features_rest[sel].repeat(N, 1, 1),
            self._features_depth[sel].repeat(N, 1), self._features_prior[sel].repeat(N, 1),
            self._features_noise[sel].repeat(N, 1), self._features_illu[sel].repeat(N, 1),
            self._opacity[sel].repeat(N, 1), new_scaling, self._rotation[sel].repeat(N, 1),
            self._features_emission[sel].repeat(N, 1))
        prune_filter = torch.cat((sel, torch.zeros(N * int(sel.sum()), device=dev, dtype=torch.bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        dev = self._xyz.device
        sel = torch.norm(grads, dim=-1) >= grad_threshold
        sel = torch.logical_and(sel, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
        # LITA-GS samples the clone position inside the parent instead of copying it
        stds = self.get_scaling[sel]
        means = torch.zeros((stds.size(0), 3), device=dev)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[sel])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[sel]
        self.densification_postfix(
            new_xyz, self._features_dc[sel], self._features_rest[sel], self._features_depth[sel],
            self._features_prior[sel], self._features_noise[sel], self._features_illu[sel],
            self._opacity[sel], self._scaling[sel], self._rotation[sel], self._features_emission[sel])

    # ------------------------------------------------------------------ i/o
    def save_emission(self, path: str) -> None:
        torch.save({"emission_raw": self._features_emission.detach().cpu()}, path)

    def load_emission(self, path: str, strict: bool = True) -> None:
        e = torch.load(path, map_location="cpu")["emission_raw"]
        if e.shape[0] != self._xyz.shape[0]:
            msg = f"emission rows {e.shape[0]} != gaussians {self._xyz.shape[0]}"
            if strict:
                raise RuntimeError(msg)
            print("[halogs] WARNING:", msg, "- resetting emission")
            self._init_emission(self._xyz.shape[0])
            return
        self._features_emission = nn.Parameter(e.to(self._xyz.device).contiguous().requires_grad_(True))

    def load_ply(self, path, *args, **kwargs):
        super().load_ply(path, *args, **kwargs)
        side = os.path.join(os.path.dirname(path), "emission.pt")
        if os.path.exists(side):
            self.load_emission(side, strict=False)
        else:
            self._init_emission(self._xyz.shape[0])

    def emission_stats(self) -> dict:
        with torch.no_grad():
            lum = self.get_emission.mean(1)
            return {"n_gaussians": int(lum.shape[0]),
                    "emission_mean": float(lum.mean()) if lum.numel() else 0.0,
                    "emission_max": float(lum.max()) if lum.numel() else 0.0,
                    "frac_emitters_gt_0.1": float((lum > 0.1).float().mean()) if lum.numel() else 0.0}
