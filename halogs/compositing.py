"""Glow compositing: linear convolution of the emission map with the APSF kernel.

Uses FFT convolution with zero padding so a 193x193 kernel on a 1-megapixel map
costs a couple of FFTs rather than a dense conv2d.  Fully differentiable with
respect to both the emission map and the kernel.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def fft_conv2d(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """'same'-size linear convolution of img (C,H,W) with a centred (K,K) kernel.

    True linear convolution ('same' window).  Equals F.conv2d(img, flip(kernel),
    padding=K//2); for the radially symmetric glow kernels used here the flip is
    immaterial.
    """
    C, H, W = img.shape
    K = kernel.shape[-1]
    assert kernel.shape[-2] == K, "kernel must be square"
    pad = K // 2
    Hp, Wp = H + K - 1, W + K - 1
    # next fast sizes
    Hf = _next_fast(Hp)
    Wf = _next_fast(Wp)
    img_f = torch.fft.rfft2(img, s=(Hf, Wf))
    ker_f = torch.fft.rfft2(kernel, s=(Hf, Wf))  # FFT product == true convolution
    out = torch.fft.irfft2(img_f * ker_f.unsqueeze(0), s=(Hf, Wf))
    # full convolution is Hp x Wp; the 'same' window starts at pad
    out = out[:, K - 1 - pad : K - 1 - pad + H, K - 1 - pad : K - 1 - pad + W]
    return out


def _next_fast(n: int) -> int:
    """Smallest 2^a 3^b 5^c >= n (keeps FFTs fast)."""
    best = None
    a = 1
    while a < 4 * n:
        b = a
        while b < 4 * n:
            c = b
            while c < n:
                c *= 5
            if best is None or c < best:
                best = c
            b *= 3
        a *= 2
    return int(best)


def direct_conv2d(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Reference implementation (slow) used by the tests."""
    C = img.shape[0]
    K = kernel.shape[-1]
    ker = torch.flip(kernel, dims=(-2, -1)).view(1, 1, K, K).repeat(C, 1, 1, 1)
    return F.conv2d(img.unsqueeze(0), ker, padding=K // 2, groups=C).squeeze(0)


def composite(scene: torch.Tensor, emission: torch.Tensor, kernel: torch.Tensor,
              gain: torch.Tensor | float = 1.0, clamp: bool = True):
    """Return (composite, glow_layer).

    scene, emission: (3,H,W).  kernel: (K,K) normalised.  gain: scalar exposure.
    composite = clip( gain * (scene + k * emission) ).
    """
    glow = fft_conv2d(emission, kernel)
    out = (scene + glow) * gain
    if clamp:
        out = out.clamp(0.0, 1.0)
    return out, glow
