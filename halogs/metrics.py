"""Image metrics with optional region masks.

PSNR / SSIM are computed with explicit masks so that in-glow and out-of-glow
numbers come from the same pixels for every method.  LPIPS is computed on the
full image and on a crop around the largest glow region (a black-masked full
image is not a meaningful LPIPS input).
"""
from __future__ import annotations

from typing import Optional

import torch

from .losses import ssim_map

try:  # optional
    import lpips as _lpips  # type: ignore
except Exception:  # pragma: no cover
    _lpips = None

_LPIPS_NET = None


def psnr(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    err = (pred - gt) ** 2
    if mask is not None:
        n = (mask.sum() * pred.shape[0]).clamp_min(1.0)
        mse = (err * mask).sum() / n
        if mask.sum() < 1:
            return float("nan")
    else:
        mse = err.mean()
    return float(10.0 * torch.log10(1.0 / mse.clamp_min(1e-12)))


def ssim(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    m = ssim_map(pred, gt)
    if mask is not None:
        n = (mask.sum() * pred.shape[0]).clamp_min(1.0)
        if mask.sum() < 1:
            return float("nan")
        return float((m * mask).sum() / n)
    return float(m.mean())


def lpips(pred: torch.Tensor, gt: torch.Tensor, net: str = "vgg") -> float:
    global _LPIPS_NET
    if _lpips is None:
        return float("nan")
    if _LPIPS_NET is None:
        _LPIPS_NET = _lpips.LPIPS(net=net).to(pred.device).eval()
    with torch.no_grad():
        return float(_LPIPS_NET(pred.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1))


def mask_bbox(mask: torch.Tensor, pad: int = 32):
    """Bounding box (y0,y1,x0,x1) of a (1,H,W) mask padded by ``pad`` px, or None."""
    ys, xs = torch.nonzero(mask[0] > 0.5, as_tuple=True)
    if ys.numel() == 0:
        return None
    H, W = mask.shape[-2:]
    return (max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, H),
            max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, W))


def all_metrics(pred: torch.Tensor, gt: torch.Tensor, glow: torch.Tensor, with_lpips: bool = True) -> dict:
    """Full-image and region metrics.  pred, gt (3,H,W) in [0,1]; glow (1,H,W) {0,1}."""
    non = 1.0 - glow
    out = {
        "psnr": psnr(pred, gt), "ssim": ssim(pred, gt),
        "psnr_glow": psnr(pred, gt, glow), "ssim_glow": ssim(pred, gt, glow),
        "psnr_nonglow": psnr(pred, gt, non), "ssim_nonglow": ssim(pred, gt, non),
        "glow_frac": float(glow.mean()),
    }
    if with_lpips:
        out["lpips"] = lpips(pred, gt)
        bb = mask_bbox(glow)
        if bb is not None and (bb[1] - bb[0]) >= 64 and (bb[3] - bb[2]) >= 64:
            y0, y1, x0, x1 = bb
            out["lpips_glowcrop"] = lpips(pred[:, y0:y1, x0:x1], gt[:, y0:y1, x0:x1])
        else:
            out["lpips_glowcrop"] = float("nan")
    return out
