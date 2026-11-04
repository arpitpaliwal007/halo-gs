"""Method-independent glow / light-source masks derived from the *observed* image.

These are used (a) by the ``masked`` baseline, which simply drops glow pixels
from the loss, and (b) by evaluation, to report in-glow vs. out-of-glow metrics.
They never look at any render, so every method is scored on identical regions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def luminance(img: torch.Tensor) -> torch.Tensor:
    """img (3,H,W) -> (1,H,W) Rec.601 luma."""
    r, g, b = img[0:1], img[1:2], img[2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation by a (2r+1) square via max-pooling. mask (1,H,W) float {0,1}."""
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return F.max_pool2d(mask.unsqueeze(0), k, stride=1, padding=radius).squeeze(0)


def source_mask(img: torch.Tensor, thresh: float = 0.92) -> torch.Tensor:
    """Pixels that are (near-)saturated light sources."""
    return (luminance(img) >= thresh).to(img.dtype)


def glow_mask(img: torch.Tensor, thresh: float = 0.92, radius: int = 48, min_source_px: int = 4) -> torch.Tensor:
    """Light sources dilated by ``radius`` px.  Returns (1,H,W) float in {0,1}.

    Tiny isolated bright specks (< min_source_px pixels after a 3x3 opening) are
    ignored so noise does not create glow regions.
    """
    src = source_mask(img, thresh)
    # morphological opening to kill specks
    er = -F.max_pool2d(-src.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
    opened = dilate(er, 1)
    if opened.sum() < min_source_px:
        return torch.zeros_like(src)
    return dilate(opened, radius)


def split_masks(img: torch.Tensor, **kw):
    """Return (glow_mask, non_glow_mask, source_mask) for an observed image."""
    g = glow_mask(img, **kw)
    s = source_mask(img, kw.get("thresh", 0.92))
    return g, 1.0 - g, s
