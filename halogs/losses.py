"""Losses: standard 3DGS photometric loss plus masked variants and glow priors."""
from __future__ import annotations


import torch
import torch.nn.functional as F


def gaussian_window(size: int = 11, sigma: float = 1.5, channels: int = 3, device=None, dtype=torch.float32):
    ax = torch.arange(size, dtype=dtype, device=device) - size // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    return w.expand(channels, 1, size, size).contiguous()


def ssim_map(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Per-pixel SSIM map (3,H,W) with the usual 3DGS constants."""
    C = img1.shape[0]
    w = gaussian_window(window_size, 1.5, C, img1.device, img1.dtype)
    pad = window_size // 2
    x = img1.unsqueeze(0)
    y = img2.unsqueeze(0)
    mu1 = F.conv2d(x, w, padding=pad, groups=C)
    mu2 = F.conv2d(y, w, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(x * x, w, padding=pad, groups=C) - mu1_sq
    s2 = F.conv2d(y * y, w, padding=pad, groups=C) - mu2_sq
    s12 = F.conv2d(x * y, w, padding=pad, groups=C) - mu12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return m.squeeze(0)


def photometric_loss(pred: torch.Tensor, gt: torch.Tensor, lambda_dssim: float = 0.2,
                     valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(1-l) L1 + l (1-SSIM).  ``valid`` (1,H,W) in {0,1} restricts both terms.

    Returns (loss, l1) so the caller can log L1 separately.  Without a mask it
    matches the official 3DGS objective up to SSIM edge handling.
    """
    if valid is None:
        l1 = (pred - gt).abs().mean()
        s = ssim_map(pred, gt).mean()
    else:
        v = valid.to(pred.dtype)
        denom = (v.sum() * pred.shape[0]).clamp_min(1.0)
        l1 = ((pred - gt).abs() * v).sum() / denom
        s = (ssim_map(pred, gt) * v).sum() / denom
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - s), l1


def emission_sparsity(emission: torch.Tensor) -> torch.Tensor:
    """L1 on the per-Gaussian emission (N,3): glow should be explained by few emitters."""
    return emission.mean()


def glow_tv(glow: torch.Tensor) -> torch.Tensor:
    """Total variation of the glow layer (3,H,W); glow is smooth by construction, this is a mild prior."""
    dx = (glow[:, :, 1:] - glow[:, :, :-1]).abs().mean()
    dy = (glow[:, 1:, :] - glow[:, :-1, :]).abs().mean()
    return dx + dy
