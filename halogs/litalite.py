"""LitaLite (corrected) and LitaLite + HaloGS.

LitaLite is the reference-free low-light model from the earlier project (the
"Direct Method"): each Gaussian has a view-dependent reflectance (its SH
colour) and one scalar illumination L in (0, 1).

    bright render   R      = G(c)          -> the enhanced output
    dark render     I_dark = G(c * L)      -> must reproduce the captured photo
    L_exp = photometric(R, target(I_in))   target: linear brightening of the input
    L_rec = photometric(I_dark, I_in)

With HaloGS the dark render gains the glow layer of the emissive Gaussians:

    I_low = clip(I_dark + k * E)           E = G(emission), k = scattering kernel

so light sources and their halos are explained by emission + kernel instead of
being baked into reflectance (which would then get brightened by L_exp).

Corrections relative to the original litalite.py (see implementation_audit_2026-10-02.md):
  1. illumination is a per-Gaussian optimiser group (halogs.gaussians), so it
     follows Gaussian identity through prune / clone / split and keeps Adam moments;
  2. densification statistics can be accumulated from both branches (``dense_from='both'``),
     not only from the bright one;
  3. the brightened target is hue-safe: pixels that would clip are scaled down as a
     whole instead of clipping channels independently (which changed R:G:B ratios);
  4. the gain can be shared by the whole scene (``target_gain='scene'``) instead of per image.

Optional improvements (closing the gap to LITA-GS cheaply):
  * RGB illumination (``illum_channels=3``) so coloured light is not pushed into reflectance;
  * a learned per-view tone curve on the dark branch (:class:`ToneCurve`);
  * a light edge-preserving denoiser on the brightened target (:func:`bilateral`).
"""
from __future__ import annotations

from typing import Optional

import torch

from .compositing import composite as _composite
from .halo import HaloModel, _render_kwargs


def reflectance_colours(pc, cam) -> torch.Tensor:
    """View-dependent RGB of every Gaussian from its SH (as in the official renderer), clipped to [0, 1]."""
    from utils.sh_utils import eval_sh  # type: ignore  (gaussian-splatting)

    feats = pc.get_features
    shs = feats.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    d = pc.get_xyz - cam.camera_center.repeat(feats.shape[0], 1)
    d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return (eval_sh(pc.active_sh_degree, shs, d) + 0.5).clamp(0.0, 1.0)


def brighten_target(img: torch.Tensor, theta: float = 0.40, gain: Optional[float] = None,
                    hue_safe: bool = True) -> torch.Tensor:
    """Linear brightening of a dark image (3,H,W).

    gain = theta / mean(img) unless a (scene-wide) gain is given.  With ``hue_safe``
    a pixel whose brightest channel would exceed 1 is divided by that channel, so
    R:G:B ratios are preserved exactly; without it channels are clipped independently
    (the original behaviour).
    """
    g = float(gain) if gain is not None else theta / float(img.mean().clamp_min(1e-4))
    t = img * g
    if hue_safe:
        peak = t.max(dim=0, keepdim=True).values.clamp_min(1.0)
        t = t / peak
    return t.clamp(0.0, 1.0)


def scene_gain(cams, theta: float = 0.40) -> float:
    """One gain for every view: theta / mean brightness over all training images."""
    with torch.no_grad():
        m = torch.stack([c.original_image.mean() for c in cams]).mean()
    return theta / float(m.clamp_min(1e-4))


def illum_init_from(cams, theta: float = 0.40) -> float:
    """Initial illumination so that R * L starts near the observed brightness (as in litalite.py)."""
    with torch.no_grad():
        m = float(torch.stack([c.original_image.mean() for c in cams]).mean())
    return min(max(m / max(theta, 1e-3), 1e-3), 0.99)


class ToneCurve(torch.nn.Module):
    """Learned per-view camera response on the *dark* branch: y = g_v * x ** gamma_v.

    The captured photo is not a linear copy of the scene: exposure, ISO and the phone's tone
    curve change from shot to shot.  LITA-GS handles this with a learned tonemapper on the
    dark side while keeping a fixed exposure for the bright output; this is the same idea in
    two parameters per view.  It starts as the identity (g = 1, gamma = 1), is monotone, and
    unseen (test) views use the mean training curve.  The bright target keeps its fixed theta:
    a freely learned brightness target has nothing to anchor it without a reference image.
    """

    def __init__(self, train_names):
        super().__init__()
        self.name_to_idx = {n: i for i, n in enumerate(sorted(train_names))}
        n = max(len(self.name_to_idx), 1)
        self.log_gain = torch.nn.Parameter(torch.zeros(n))
        self.log_gamma = torch.nn.Parameter(torch.zeros(n))

    def params(self, name):
        i = self.name_to_idx.get(name)
        if i is None:
            return torch.exp(self.log_gain.mean()), torch.exp(self.log_gamma.mean())
        return torch.exp(self.log_gain[i]), torch.exp(self.log_gamma[i])

    def forward(self, x: torch.Tensor, name) -> torch.Tensor:
        g, gamma = self.params(name)
        return g * x.clamp_min(1e-6) ** gamma

    def describe(self) -> dict:
        with torch.no_grad():
            g, gm = torch.exp(self.log_gain), torch.exp(self.log_gamma)
            return {"gain_mean": float(g.mean()), "gain_min": float(g.min()), "gain_max": float(g.max()),
                    "gamma_mean": float(gm.mean()), "gamma_min": float(gm.min()), "gamma_max": float(gm.max())}

    def save(self, path):
        torch.save({"name_to_idx": self.name_to_idx, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, device="cuda"):
        blob = torch.load(path, map_location="cpu")
        m = cls(list(blob["name_to_idx"].keys()))
        m.name_to_idx = blob["name_to_idx"]
        m.load_state_dict(blob["state_dict"])
        return m.to(device)


def bilateral(img: torch.Tensor, radius: int = 2, sigma_s: float = 1.5, sigma_r: float = 0.08) -> torch.Tensor:
    """Edge-preserving bilateral filter of a (3,H,W) image in [0,1]; range weights use the RGB distance.

    Used once per training image on the *brightened* target, where the gain (often 4-10x on dark
    captures) amplifies sensor noise that the bright render would otherwise learn as speckle.
    The dark reconstruction loss still uses the raw photo.
    """
    C, H, W = img.shape
    k = 2 * radius + 1
    x = img.unsqueeze(0)
    pad = torch.nn.functional.pad(x, (radius,) * 4, mode="reflect")
    patches = torch.nn.functional.unfold(pad, k).view(1, C, k * k, H, W)
    ax = torch.arange(-radius, radius + 1, dtype=img.dtype, device=img.device)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    ws = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_s ** 2)).view(1, 1, k * k, 1, 1)
    d2 = ((patches - x.unsqueeze(2)) ** 2).sum(1, keepdim=True)
    w = ws * torch.exp(-d2 / (2 * sigma_r ** 2))
    out = (w * patches).sum(2) / w.sum(2).clamp_min(1e-8)
    return out.squeeze(0)


def render_litalite(render_fn, cam, pc, pipe, bg_color: torch.Tensor, halo: Optional[HaloModel] = None,
                    use_glow: bool = False, clamp: bool = True,
                    tone: Optional["ToneCurve"] = None) -> dict:
    """Bright (reflectance) render, dark render (reflectance * illumination) and optional glow.

    Returned keys:
      bright   enhanced output R                          (compare with a normal-light reference)
      scene    dark glow-free render G(c * L)
      render   dark composite clip(scene + k * E)         (compare with the observed photo)
      glow, emission
      dense    list of (viewspace_points, visibility_filter, radii), bright branch first
      viewspace_points / visibility_filter / radii  from the bright branch (for compatibility)
    """
    kw = _render_kwargs(render_fn)
    refl = reflectance_colours(pc, cam)
    b = render_fn(cam, pc, pipe, bg_color, override_color=refl, **kw)
    d = render_fn(cam, pc, pipe, bg_color, override_color=refl * pc.get_illum, **kw)
    bright, dark = b["render"], d["render"]
    out = {"bright": bright, "scene": dark,
           "viewspace_points": b["viewspace_points"], "visibility_filter": b["visibility_filter"], "radii": b["radii"],
           "dense": [(b["viewspace_points"], b["visibility_filter"], b["radii"]),
                     (d["viewspace_points"], d["visibility_filter"], d["radii"])]}
    name = getattr(cam, "image_name", None)
    if use_glow and halo is not None:
        black = torch.zeros_like(bg_color)
        e = render_fn(cam, pc, pipe, black, override_color=pc.get_emission, **kw)["render"]
        comp, glow = _composite(dark, e, halo.kernel.kernel(), gain=halo.gain(name), clamp=False)
    else:
        comp, glow, e = dark, torch.zeros_like(dark), torch.zeros_like(dark)
    if tone is not None:  # camera response acts on all light reaching the sensor (scene + glow)
        comp = tone(comp, name)
    out.update({"render": comp.clamp(0, 1) if clamp else comp, "glow": glow, "emission": e})
    return out
