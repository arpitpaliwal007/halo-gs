"""HaloModel: the glow side of the model (kernel + optional per-view exposure),
and the two-pass renderer that produces scene / emission / glow / composite.
"""
from __future__ import annotations

import inspect
import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .apsf import make_kernel
from .compositing import composite as _composite


class HaloModel(nn.Module):
    """Glow kernel + per-view log-exposure.

    ``exposure``: if enabled, one learnable log-gain per *training* camera name
    (night captures often vary exposure/ISO per shot); test cameras get the
    mean training gain.  Off by default so the composite stays comparable to a
    plain 3DGS render.
    """

    def __init__(self, kernel_kind: str = "both", radius: int = 96, focal_px: float = 600.0,
                 per_view_exposure: bool = False, train_names: Optional[list] = None,
                 learn_kernel: bool = True, **kernel_kw):
        super().__init__()
        self.kernel_kind = kernel_kind
        self.kernel_kw = dict(kernel_kw)
        self.kernel = make_kernel(kernel_kind, radius, focal_px, learn=learn_kernel, **kernel_kw)
        self.per_view_exposure = per_view_exposure
        self.name_to_idx: Dict[str, int] = {}
        if per_view_exposure and train_names:
            self.name_to_idx = {n: i for i, n in enumerate(sorted(train_names))}
            self.log_gain = nn.Parameter(torch.zeros(len(self.name_to_idx)))
        else:
            self.log_gain = nn.Parameter(torch.zeros(1), requires_grad=False)

    def gain(self, image_name: Optional[str]) -> torch.Tensor:
        if not self.per_view_exposure or not self.name_to_idx:
            return torch.ones((), device=self.log_gain.device)
        if image_name in self.name_to_idx:
            return torch.exp(self.log_gain[self.name_to_idx[image_name]])
        return torch.exp(self.log_gain.mean())  # unseen (test) view

    def param_groups(self, kernel_lr: float = 1e-3, exposure_lr: float = 1e-3):
        groups = [{"params": [p for p in self.kernel.parameters() if p.requires_grad], "lr": kernel_lr, "name": "kernel"}]
        if self.per_view_exposure and self.log_gain.requires_grad:
            groups.append({"params": [self.log_gain], "lr": exposure_lr, "name": "exposure"})
        return [g for g in groups if g["params"]]

    def describe(self) -> dict:
        d = {"kernel": self.kernel.describe(), "per_view_exposure": self.per_view_exposure}
        if self.per_view_exposure:
            with torch.no_grad():
                d["gain_mean"] = float(torch.exp(self.log_gain).mean())
                d["gain_min"] = float(torch.exp(self.log_gain).min())
                d["gain_max"] = float(torch.exp(self.log_gain).max())
        return d

    # ---- i/o ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({"kernel_kind": self.kernel_kind, "kernel_kw": getattr(self, "kernel_kw", {}),
                    "radius": getattr(self.kernel, "radius", None),
                    "focal_px": getattr(self.kernel, "focal_px", None),
                    "per_view_exposure": self.per_view_exposure,
                    "name_to_idx": self.name_to_idx,
                    "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device="cuda") -> "HaloModel":
        blob = torch.load(path, map_location="cpu")
        names = list(blob["name_to_idx"].keys()) if blob["per_view_exposure"] else None
        m = cls(kernel_kind=blob["kernel_kind"], radius=blob["radius"], focal_px=blob["focal_px"] or 600.0,
                per_view_exposure=blob["per_view_exposure"], train_names=names, **blob.get("kernel_kw", {}))
        m.name_to_idx = blob["name_to_idx"]
        m.load_state_dict(blob["state_dict"])
        return m.to(device)


def focal_px_from_camera(cam) -> float:
    """Focal length in pixels from a 3DGS Camera (FoVx + image width)."""
    return 0.5 * float(cam.image_width) / math.tan(0.5 * float(cam.FoVx))


def _render_kwargs(render_fn):
    """The current official repo takes extra kwargs; the 2023 fork does not."""
    sig = inspect.signature(render_fn)
    kw = {}
    if "use_trained_exp" in sig.parameters:
        kw["use_trained_exp"] = False
    if "separate_sh" in sig.parameters:
        kw["separate_sh"] = False
    return kw


def render_halo(render_fn, cam, gaussians, pipe, bg_color: torch.Tensor, halo: Optional[HaloModel],
                mode: str = "halo", clamp: bool = True) -> dict:
    """Two-pass render.

    Returns dict with keys: render (composite or plain), scene, glow, emission,
    viewspace_points, visibility_filter, radii (from the *scene* pass).
    mode: 'halo' (scene + glow) | 'vanilla' / 'masked' (scene only).
    """
    kw = _render_kwargs(render_fn)
    pkg = render_fn(cam, gaussians, pipe, bg_color, **kw)
    scene = pkg["render"]
    out = {"scene": scene, "viewspace_points": pkg["viewspace_points"],
           "visibility_filter": pkg["visibility_filter"], "radii": pkg["radii"]}
    if mode != "halo" or halo is None:
        out["render"] = scene
        out["glow"] = torch.zeros_like(scene)
        out["emission"] = torch.zeros_like(scene)
        return out
    black = torch.zeros_like(bg_color)
    epkg = render_fn(cam, gaussians, pipe, black, override_color=gaussians.get_emission, **kw)
    emission = epkg["render"]
    kernel = halo.kernel.kernel()
    comp, glow = _composite(scene, emission, kernel, gain=halo.gain(getattr(cam, "image_name", None)), clamp=clamp)
    out.update({"render": comp, "glow": glow, "emission": emission})
    return out
