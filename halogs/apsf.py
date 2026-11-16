"""Physically-named glow kernels.

A photographic night-time halo has two physical origins that produce different
radial profiles:

1. **Atmospheric multiple scattering** (fog, mist, dust) — the Atmospheric Point
   Spread Function of Narasimhan & Nayar ("Shedding Light on the Weather",
   CVPR 2003).  Radially symmetric in scattering angle theta:

       APSF(T, mu) = sum_m [ g_m(T) + g_{m+1}(T) ] L_m(mu),   mu = cos(theta)
       g_m(T)      = exp( -beta_m T - alpha_m log T )
       alpha_m     = m + 1,   beta_m = (2m+1)/m * (1 - q^(m-1))   (m >= 1)

   q in (0,1) is the forward-scattering parameter, T > 0 the optical thickness.
   The Legendre series is only well-behaved for T >~ 0.8 (below that the
   truncated sum rings), so T is constrained to [0.8, inf) here; this is the
   regime the original paper reports (T ~ 1..4).  As in the public nighttime
   dehazing implementations, the sum starts at m = 1 and the result is
   normalised, so only the *shape* is used.

2. **Lens / sensor glare** (veiling glare, bloom) — a tight power-law glare
   spread function in pixel units (Talvala et al. 2007 style):

       GSF(r) = (1 + (r / gamma)^2)^(-p/2)

   gamma (px) sets the width, p the fall-off (fixed to 3 by default).

The full kernel is a normalised mixture  k = (1-w) APSF + w GSF.  Everything
is differentiable in (q, T, gamma, w) so it can be fitted jointly with the
scene.  Each component can be switched off for ablations, and a non-physical
monotone radial kernel (``RadialKernel``) is provided as a further ablation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

T_MIN = 0.8  # validity floor of the truncated Legendre expansion


def legendre_all(mu: torch.Tensor, n_terms: int) -> torch.Tensor:
    """Stacked Legendre polynomials L_0..L_{n_terms} of mu via Bonnet recursion.

    Returns tensor of shape (n_terms + 1, *mu.shape).
    """
    out = [torch.ones_like(mu), mu]
    for m in range(1, n_terms):
        out.append(((2 * m + 1) * mu * out[m] - m * out[m - 1]) / (m + 1))
    return torch.stack(out[: n_terms + 1], 0)


def apsf_profile(theta: torch.Tensor, q: torch.Tensor, T: torch.Tensor, n_terms: int = 24) -> torch.Tensor:
    """Un-normalised APSF value at scattering angle theta (radians); differentiable in q, T."""
    mu = torch.cos(theta)
    L = legendre_all(mu, n_terms + 1)
    m = torch.arange(1, n_terms + 2, dtype=theta.dtype, device=theta.device)
    alpha = m + 1.0
    beta = (2.0 * m + 1.0) / m * (1.0 - q.clamp(1e-4, 1 - 1e-4) ** (m - 1.0))
    logT = torch.log(T.clamp_min(1e-3))
    log_g = -beta * T - alpha * logT
    log_g = log_g - log_g.max()  # scale-free; kernel is normalised later
    g = torch.exp(log_g)
    coeff = g[:-1] + g[1:]  # g_m + g_{m+1}, m = 1..n_terms
    shape = (-1,) + (1,) * theta.dim()
    prof = (coeff.view(shape) * L[1 : n_terms + 1]).sum(0)
    return prof.clamp_min(0.0)


def gsf_profile(r: torch.Tensor, gamma: torch.Tensor, p: float = 3.0) -> torch.Tensor:
    """Lens glare spread function in pixel units."""
    return (1.0 + (r / gamma.clamp_min(0.3)) ** 2) ** (-0.5 * p)


def radial_grid(radius: int, device=None, dtype=torch.float32) -> torch.Tensor:
    ax = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    return torch.sqrt(xx * xx + yy * yy)


def taper(r: torch.Tensor, radius: int, width: float = 0.15) -> torch.Tensor:
    """Smooth cosine roll-off to zero at the kernel border (avoids ringing)."""
    start = radius * (1.0 - width)
    t = ((r - start) / (radius - start)).clamp(0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(math.pi * t))


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(x)) if x > 1e-6 else -12.0


def _logit(x: float) -> float:
    x = min(max(x, 1e-5), 1 - 1e-5)
    return math.log(x / (1 - x))


class GlowKernel(nn.Module):
    """Learnable physical glow kernel: mixture of atmospheric APSF and lens GSF.

    ``components``: "both" | "apsf" | "lens".
    ``focal_px`` converts pixel radius to scattering angle, theta = atan(r / f),
    so q and T keep their physical meaning regardless of image resolution.
    """

    def __init__(self, radius: int = 96, focal_px: float = 600.0, components: str = "both",
                 q_init: float = 0.9, T_init: float = 1.5, gamma_init: float = 6.0,
                 w_init: float = 0.5, p: float = 3.0, n_terms: int = 24, learn: bool = True,
                 gamma_min: float = 0.3):
        super().__init__()
        assert components in ("both", "apsf", "lens")
        self.gamma_min = float(gamma_min)  # px; raise (e.g. 3) to stop the lens term collapsing to a delta
        self.radius = int(radius)
        self.components = components
        self.focal_px = float(focal_px)
        self.n_terms = int(n_terms)
        self.p = float(p)
        self.q_logit = nn.Parameter(torch.tensor(_logit(q_init)), requires_grad=learn)
        self.T_raw = nn.Parameter(torch.tensor(_inv_softplus(T_init - T_MIN)), requires_grad=learn)
        self.gamma_raw = nn.Parameter(torch.tensor(_inv_softplus(max(gamma_init - gamma_min, 1e-3))), requires_grad=learn)
        self.w_logit = nn.Parameter(torch.tensor(_logit(w_init)), requires_grad=learn and components == "both")
        self.register_buffer("_r", radial_grid(self.radius))

    # --- constrained parameters -------------------------------------------------
    @property
    def q(self) -> torch.Tensor:
        return torch.sigmoid(self.q_logit)

    @property
    def T(self) -> torch.Tensor:
        return F.softplus(self.T_raw) + T_MIN

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.gamma_raw) + self.gamma_min

    @property
    def w(self) -> torch.Tensor:
        if self.components == "apsf":
            return torch.zeros((), device=self.w_logit.device)
        if self.components == "lens":
            return torch.ones((), device=self.w_logit.device)
        return torch.sigmoid(self.w_logit)

    def set_focal(self, focal_px: float) -> None:
        self.focal_px = float(focal_px)

    # --- kernel -------------------------------------------------------------------
    def component_profiles(self):
        r = self._r
        theta = torch.atan(r / self.focal_px)
        tap = taper(r, self.radius)
        k_apsf = apsf_profile(theta, self.q, self.T, self.n_terms) * tap
        k_lens = gsf_profile(r, self.gamma, self.p) * tap
        k_apsf = k_apsf / k_apsf.sum().clamp_min(1e-12)
        k_lens = k_lens / k_lens.sum().clamp_min(1e-12)
        return k_apsf, k_lens

    def kernel(self) -> torch.Tensor:
        """(K, K) kernel, non-negative, sums to 1."""
        k_apsf, k_lens = self.component_profiles()
        w = self.w
        k = (1.0 - w) * k_apsf + w * k_lens
        return k / k.sum().clamp_min(1e-12)

    def forward(self) -> torch.Tensor:  # pragma: no cover
        return self.kernel()

    def describe(self) -> dict:
        with torch.no_grad():
            return {"type": "glow", "components": self.components, "radius": self.radius,
                    "focal_px": self.focal_px, "q": self.q.item(), "T": self.T.item(),
                    "gamma_px": self.gamma.item(), "gamma_min": self.gamma_min, "w_lens": self.w.item(), "p": self.p}


class IdentityKernel(nn.Module):
    """Exact unit impulse for an additive-emission control."""

    def __init__(self):
        super().__init__()
        self.radius = 0
        self.register_buffer("_unit", torch.ones((1, 1)))

    def kernel(self) -> torch.Tensor:
        return self._unit

    def describe(self) -> dict:
        return {"type": "identity", "radius": self.radius}


class RadialKernel(nn.Module):
    """Ablation: non-physical learnable monotone-decreasing radial profile.

    Piecewise-linear on ``n_knots`` radii; monotone by construction (softplus
    increments of -log k).
    """

    def __init__(self, radius: int = 96, n_knots: int = 16, learn: bool = True):
        super().__init__()
        self.radius = int(radius)
        self.n_knots = int(n_knots)
        init = torch.linspace(0, 1, n_knots)
        steps = torch.log(torch.expm1(4.0 * (init[1:] - init[:-1]).clamp_min(1e-3)))
        self.steps_raw = nn.Parameter(steps, requires_grad=learn)
        self.register_buffer("_r", radial_grid(self.radius))

    def profile(self) -> torch.Tensor:
        dec = F.softplus(self.steps_raw)
        prof = torch.cat([torch.zeros(1, device=dec.device, dtype=dec.dtype), torch.cumsum(dec, 0)])
        return torch.exp(-prof)

    def kernel(self) -> torch.Tensor:
        prof = self.profile()
        pos = (self._r / self.radius) * (self.n_knots - 1)
        lo = pos.floor().long().clamp(0, self.n_knots - 1)
        hi = (lo + 1).clamp(max=self.n_knots - 1)
        wgt = (pos - lo.to(pos.dtype)).clamp(0, 1)
        k = prof[lo] * (1 - wgt) + prof[hi] * wgt
        k = k * taper(self._r, self.radius)
        return k / k.sum().clamp_min(1e-12)

    def forward(self) -> torch.Tensor:  # pragma: no cover
        return self.kernel()

    def set_focal(self, focal_px: float) -> None:  # no-op, pixel-space kernel
        pass

    def describe(self) -> dict:
        return {"type": "radial", "radius": self.radius, "n_knots": self.n_knots,
                "profile": [float(v) for v in self.profile().detach()]}


def make_kernel(kind: str, radius: int, focal_px: float, **kw) -> nn.Module:
    """kind: 'both' | 'apsf' | 'lens' | 'radial'."""
    if kind == "identity":
        return IdentityKernel()
    if kind in ("both", "apsf", "lens"):
        return GlowKernel(radius=radius, focal_px=focal_px, components=kind, **kw)  # kw may include gamma_min, gamma_init, w_init
    if kind == "radial":
        return RadialKernel(radius=radius, **{k: v for k, v in kw.items() if k in ("n_knots", "learn")})
    raise ValueError(f"unknown kernel kind {kind!r}")
