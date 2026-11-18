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
                    hue_safe: bool = True, wb_gains: Optional[torch.Tensor] = None,
                    gamma: float = 1.0) -> torch.Tensor:
    """Brightening of a dark image (3,H,W) into the bright pseudo-target.

    1. optional white balance: multiply channel c by ``wb_gains[c]`` (see :func:`grayworld_gains`);
    2. linear gain = theta / mean(img) unless a (scene-wide) gain is given;
    3. with ``hue_safe`` a pixel whose brightest channel would exceed 1 is divided by that
       channel, so R:G:B ratios are preserved; otherwise channels are clipped independently
       (the original behaviour);
    4. optional tone curve t ** gamma (gamma < 1 lifts shadows and lowers contrast).
    """
    x = img if wb_gains is None else img * wb_gains.to(img).view(3, 1, 1)
    g = float(gain) if gain is not None else theta / float(x.mean().clamp_min(1e-4))
    t = x * g
    if hue_safe:
        peak = t.max(dim=0, keepdim=True).values.clamp_min(1.0)
        t = t / peak
    t = t.clamp(0.0, 1.0)
    if gamma != 1.0:
        t = t.clamp_min(1e-6) ** gamma
    return t


def grayworld_gains(cams) -> torch.Tensor:
    """Scene-wide grey-world white balance from the *training* dark images only.

    Gains g_c = mean(all channels) / mean(channel c), so the average colour of the scene
    becomes neutral grey while overall brightness is unchanged.  No reference image is used.
    """
    with torch.no_grad():
        m = torch.stack([c.original_image.float().mean(dim=(1, 2)) for c in cams]).mean(0)
    return (m.mean() / m.clamp_min(1e-6)).detach().cpu()


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


class ColourCurveCamera(torch.nn.Module):
    """Per-view camera model on the dark branch (Luminance-GS style), tied to a global one.

        y = T_v( M_v @ x + b_v )

    M_v (3x3) and b_v start at identity / 0; T_v is a monotone piecewise-linear curve with
    ``knots`` segments, parameterised by softmax increments (so it is monotone and maps 0->0,
    1->scale).  Each view's curve is the global curve plus a per-view offset; the tie loss
    keeps the offsets and the colour matrices small, so per-view parameters model exposure /
    white-balance changes of the phone, not scene content.  Unseen views use the global curve
    with identity colour.  Same interface as :class:`ToneCurve` (forward(x, name), describe, save, load).
    """

    kind = "colour_curve"

    def __init__(self, train_names, knots: int = 16):
        super().__init__()
        self.name_to_idx = {n: i for i, n in enumerate(sorted(train_names))}
        n = max(len(self.name_to_idx), 1)
        self.knots = knots
        self.g_logits = torch.nn.Parameter(torch.zeros(knots))           # global curve shape (identity)
        self.g_logscale = torch.nn.Parameter(torch.zeros(1))
        self.v_logits = torch.nn.Parameter(torch.zeros(n, knots))        # per-view offsets
        self.v_logscale = torch.nn.Parameter(torch.zeros(n))
        self.v_mat = torch.nn.Parameter(torch.zeros(n, 3, 3))            # M_v = I + v_mat
        self.v_bias = torch.nn.Parameter(torch.zeros(n, 3))

    def _curve(self, logits, logscale, x):
        inc = torch.softmax(logits, 0)                                    # (K,) sums to 1
        ys = torch.cat([inc.new_zeros(1), torch.cumsum(inc, 0)]) * torch.exp(logscale)
        K = self.knots
        u = x.clamp(0, 1) * K
        i = u.floor().clamp(max=K - 1).long()
        f = u - i.to(u.dtype)
        return ys[i] * (1 - f) + ys[i + 1] * f

    def forward(self, x: torch.Tensor, name) -> torch.Tensor:
        i = self.name_to_idx.get(name)
        if i is None:
            return self._curve(self.g_logits, self.g_logscale[0], x)
        M = torch.eye(3, device=x.device, dtype=x.dtype) + self.v_mat[i]
        y = torch.einsum("ij,jhw->ihw", M, x) + self.v_bias[i].view(3, 1, 1)
        return self._curve(self.g_logits + self.v_logits[i], self.g_logscale[0] + self.v_logscale[i], y)

    def tie_loss(self) -> torch.Tensor:
        return (self.v_logits.pow(2).mean() + self.v_logscale.pow(2).mean()
                + self.v_mat.pow(2).mean() + self.v_bias.pow(2).mean())

    def describe(self) -> dict:
        with torch.no_grad():
            ident = self._curve(self.g_logits, self.g_logscale[0], torch.tensor([0.1, 0.5, 1.0], device=self.g_logits.device))
            return {"global_at_0.1_0.5_1": [round(float(v), 4) for v in ident],
                    "mat_dev": float(self.v_mat.abs().mean()), "curve_dev": float(self.v_logits.abs().mean())}

    def save(self, path):
        torch.save({"kind": self.kind, "knots": self.knots, "name_to_idx": self.name_to_idx,
                    "state_dict": self.state_dict()}, path)


def load_camera_model(path, device="cuda"):
    """Load either a :class:`ToneCurve` or a :class:`ColourCurveCamera` saved as tone.pt."""
    blob = torch.load(path, map_location="cpu")
    if blob.get("kind") == "colour_curve":
        m = ColourCurveCamera(list(blob["name_to_idx"].keys()), blob["knots"])
        m.name_to_idx = blob["name_to_idx"]
        m.load_state_dict(blob["state_dict"])
        return m.to(device)
    return ToneCurve.load(path, device)


def exposure_loss(img: torch.Tensor, target: float = 0.6, patch: int = 16) -> torch.Tensor:
    """Zero-DCE exposure control: mean luminance of each patch pulled toward ``target``."""
    y = img.mean(0, keepdim=True).unsqueeze(0)
    m = torch.nn.functional.avg_pool2d(y, patch, patch)
    return (m - target).pow(2).mean()


def colour_constancy_loss(img: torch.Tensor) -> torch.Tensor:
    """Zero-DCE colour constancy: channel means of the image should be close to each other."""
    r, g, b = img.mean(dim=(1, 2))
    return (r - g).pow(2) + (r - b).pow(2) + (g - b).pow(2)


def he_curve(cams, theta: float = 0.40, blend: float = 0.5, bins: int = 256, smooth: int = 5,
             denoise=None) -> torch.Tensor:
    """Scene-wide luminance tone curve (LUT of ``bins`` values) from the *training* dark images only.

    Luminance-GS initialises its curve from the CDF of the histogram-equalised input; plain HE
    over-flattens, so the LUT is ``blend * CDF + (1 - blend) * linear``, where linear is the
    original LitaLite gain curve min(g x, 1) with g = theta / mean.  ``blend`` is fixed a priori
    (0.5), it is never tuned on a reference.  Monotone by construction, 0 -> 0.
    """
    with torch.no_grad():
        ys = []
        for c in cams:
            im = c.original_image.float()
            if denoise is not None:
                im = denoise(im)
            ys.append(im.mean(0).flatten())
        y = torch.cat(ys).clamp(0, 1).cpu()
        h = torch.histc(y, bins=bins, min=0.0, max=1.0)
        if smooth > 1:
            k = torch.ones(1, 1, smooth) / smooth
            h = torch.nn.functional.conv1d(h.view(1, 1, -1), k, padding=smooth // 2).view(-1)[:bins]
        cdf = torch.cumsum(h, 0)
        cdf = torch.cat([torch.zeros(1), cdf / cdf[-1].clamp_min(1e-8)])   # bins+1 knots on [0,1]
        x = torch.linspace(0, 1, bins + 1)
        lin = (x * theta / float(y.mean().clamp_min(1e-4))).clamp(0, 1)
        return (blend * cdf + (1 - blend) * lin).clamp(0, 1)


def apply_luma_curve(img: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Brighten (3,H,W) by a luminance LUT, scaling R,G,B by the same factor (hue preserved), hue-safe clip."""
    lut = lut.to(img)
    K = lut.numel() - 1
    y = img.mean(0, keepdim=True).clamp(0, 1)
    u = y * K
    i = u.floor().clamp(max=K - 1).long()
    f = u - i.to(u.dtype)
    yo = lut[i] * (1 - f) + lut[i + 1] * f
    t = img * (yo / y.clamp_min(1e-4))
    peak = t.max(dim=0, keepdim=True).values.clamp_min(1.0)
    return (t / peak).clamp(0, 1)


def spatial_consistency_loss(out: torch.Tensor, inp: torch.Tensor, patch: int = 4) -> torch.Tensor:
    """Zero-DCE spatial consistency with Luminance-GS's adaptive scale 0.5 / mean(input).

    The 4-neighbour luminance differences of the (patch-pooled) output must match those of the
    input, scaled up by 0.5 / mean(input), so local structure is kept while brightening.
    """
    def pooled(x):
        return torch.nn.functional.avg_pool2d(x.mean(0, keepdim=True).unsqueeze(0), patch, patch)[0, 0]
    po, pi = pooled(out), pooled(inp) * (0.5 / float(inp.mean().clamp_min(1e-4)))
    loss = 0.0
    for dy, dx in ((0, 1), (1, 0)):
        do = po[dy:, dx:] - po[:po.shape[0] - dy, :po.shape[1] - dx]
        di = pi[dy:, dx:] - pi[:pi.shape[0] - dy, :pi.shape[1] - dx]
        loss = loss + (do.abs() - di.abs()).pow(2).mean()
    return loss


def illum_prior_loss(illum_map: torch.Tensor, dark: torch.Tensor, target: torch.Tensor,
                     w_smooth: float = 0.001) -> torch.Tensor:
    """LL-Gaussian max-channel illumination prior, adapted to dark = R * L with R ~ bright target.

    LL-Gaussian ties S to max_c(I) (Retinex with max_c R = 1).  Here the bright output is R, so the
    consistent max-channel target is max_c(I) / max_c(T): the per-pixel darkening between the
    brightened target and the photo.  Plus an edge-aware smoothness so L cannot carry texture.
    """
    s = illum_map.max(0, keepdim=True).values
    tgt = (dark.max(0, keepdim=True).values / target.max(0, keepdim=True).values.clamp_min(1e-3)).clamp(0, 1)
    loss = (s - tgt.detach()).abs().mean()
    if w_smooth > 0:
        g = dark.mean(0, keepdim=True)
        wx = torch.exp(-10 * (g[:, :, 1:] - g[:, :, :-1]).abs())
        wy = torch.exp(-10 * (g[:, 1:] - g[:, :-1]).abs())
        loss = loss + w_smooth * ((wx * (s[:, :, 1:] - s[:, :, :-1]).abs()).mean()
                                  + (wy * (s[:, 1:] - s[:, :-1]).abs()).mean())
    return loss


def relative_photometric_loss(pred: torch.Tensor, gt: torch.Tensor, lambda_dssim: float = 0.2,
                              eps: float = 0.01):
    """LL-Gaussian brightness-relative L1: |pred - gt| / (sg(pred) + eps), renormalised to mean weight 1
    (so the loss scale matches plain L1), + D-SSIM.  Dark regions count as much as bright ones."""
    from .losses import ssim_map
    w = 1.0 / (pred.detach() + eps)
    w = w / w.mean()
    l1 = (w * (pred - gt).abs()).mean()
    return (1 - lambda_dssim) * l1 + lambda_dssim * (1 - ssim_map(pred, gt).mean()), (pred - gt).abs().mean()


# ----------------------------------------------------------------------------- NOVA-GS style target
def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    """Mean over a (2r+1)^2 window, reflect-padded; x (C,H,W)."""
    r = int(min(r, (x.shape[-1] - 1), (x.shape[-2] - 1)))
    if r < 1:
        return x
    p = torch.nn.functional.pad(x.unsqueeze(0), (r, r, r, r), mode="reflect")
    return torch.nn.functional.avg_pool2d(p, 2 * r + 1, stride=1)[0]


def guided_filter_self(x: torch.Tensor, r: int = 16, eps: float = 1e-4) -> torch.Tensor:
    """Self-guided filter (He et al.) of every channel of x (C,H,W): smooth, edge-preserving brightness map."""
    m = _box(x, r)
    var = (_box(x * x, r) - m * m).clamp_min(0)
    a = var / (var + eps)
    b = m - a * m
    return _box(a, r) * x + _box(b, r)


def noise_sigma(img: torch.Tensor) -> float:
    """Immerkaer (1996) fast noise std estimate on the channel mean of (3,H,W)."""
    y = img.mean(0)[None, None]
    k = torch.tensor([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]], dtype=y.dtype, device=y.device)[None, None]
    H, W = y.shape[-2:]
    s = torch.nn.functional.conv2d(y, k).abs().sum()
    return float(s * (3.14159265 / 2) ** 0.5 / (6 * max(W - 2, 1) * max(H - 2, 1)))


def nova_maps(img: torch.Tensor, gain: float, r: int = 16, eps: Optional[float] = None) -> torch.Tensor:
    """Smooth per-channel brightness map of the exposure-normalised image clip(gain * img), in [0, 1]."""
    x = (img * gain).clamp(0, 1)
    if eps is None:  # NOVA-GS: eps from the image's noise level (here 4 sigma^2 of the normalised image)
        eps = max(4.0 * (noise_sigma(img) * gain) ** 2, 1e-5)
    return guided_filter_self(x, r, eps).clamp(1e-4, 1.0), x


def nova_gamma(maps, target_mean: float = 0.5, lo: float = 0.05, hi: float = 3.0, iters: int = 50) -> float:
    """Binary search (NOVA-GS) for one gamma with mean(L ** gamma) = target_mean over all given maps."""
    v = torch.cat([m.flatten() for m in maps]).clamp(1e-4, 1)
    if v.numel() > 2_000_000:
        v = v[torch.randperm(v.numel(), device=v.device)[:2_000_000]]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if float((v ** mid).mean()) > target_mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def nova_target(img: torch.Tensor, gain: float, gamma: float, r: int = 16, eta: float = 1e-3,
                eps: Optional[float] = None) -> torch.Tensor:
    """NOVA-GS ratio brightening after LitaLite's exposure normalisation:
    x = clip(gain * img), L = guided(x), target = x * L**gamma / (L + eta), hue-safe clip."""
    L, x = nova_maps(img, gain, r, eps)
    t = x * (L ** gamma) / (L + eta)
    peak = t.max(dim=0, keepdim=True).values.clamp_min(1.0)
    return (t / peak).clamp(0, 1)


def gaussian_pixel_coords(pc, cam) -> torch.Tensor:
    """Projected centres of all Gaussians in NDC (x, y) in [-1, 1] (for grid_sample)."""
    xyz = pc.get_xyz
    ph = torch.cat([xyz, torch.ones_like(xyz[:, :1])], 1) @ cam.full_proj_transform
    return ph[:, :2] / ph[:, 3:4].clamp_min(1e-7)


def sh_noise_penalty(pc, cam, noise_map: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """NOVA-GS noise-guided SH regularisation: each visible Gaussian samples the noise map at its projected
    centre; the sample weights a penalty on its higher-order SH coefficients (so noise is not stored as
    view-dependent colour)."""
    ndc = gaussian_pixel_coords(pc, cam)[visible]
    if ndc.shape[0] == 0:
        return noise_map.sum() * 0
    with torch.no_grad():
        w = torch.nn.functional.grid_sample(noise_map[None, None], ndc.view(1, 1, -1, 2),
                                            mode="bilinear", align_corners=False).view(-1)
    rest = pc._features_rest[visible]
    return (w * rest.pow(2).flatten(1).sum(1)).mean()


def pseudo_weight(it: int, start: int, end: int) -> float:
    """Pseudo-target weight multiplier: 1 until ``start``, linear to 0 at ``end`` (end<=0: never fades)."""
    if end <= 0 or it <= start:
        return 1.0
    if it >= end:
        return 0.0
    return 1.0 - (it - start) / float(end - start)


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
                    tone: Optional["ToneCurve"] = None, illum_map: bool = False) -> dict:
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
    if illum_map:  # rendered illumination G(L) on black, for the illumination prior
        L = pc.get_illum
        L = L.expand(-1, 3) if L.shape[1] == 1 else L
        out["illum_map"] = render_fn(cam, pc, pipe, torch.zeros_like(bg_color), override_color=L, **kw)["render"]
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
