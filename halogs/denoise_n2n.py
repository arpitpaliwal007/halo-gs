"""Self-supervised U-Net denoiser (Neighbor2Neighbor) trained on a scene's own dark images.

Neighbor2Neighbor (Huang et al., CVPR 2021) needs no clean images: each noisy image y is
split by a random neighbour sub-sampler into two half-resolution images g1(y), g2(y) whose
noise is independent but whose content is nearly identical, and the network f is trained so

    L = || f(g1(y)) - g2(y) ||^2  +  gamma * || f(g1(y)) - g2(y) - (g1(f(y)) - g2(f(y))) ||^2

where f(y) is computed without gradient (the second term corrects for the small content
difference between the two sub-images).  The network is a 4-level U-Net (width 32: ~3.7 M params; width 16: ~0.9 M).

In LitaLite it is trained once per scene on the *training* dark images only (never the test
views), then used to denoise each training image before it is brightened into the bright
target.  The dark reconstruction loss keeps the raw photo.
"""
from __future__ import annotations

import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- network
def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True))


class UNet(nn.Module):
    """4-level U-Net with skip connections, predicting a residual (output = input - noise)."""

    def __init__(self, ch: int = 3, width: int = 32):
        super().__init__()
        w = width
        self.e1, self.e2, self.e3, self.e4 = _block(ch, w), _block(w, 2 * w), _block(2 * w, 4 * w), _block(4 * w, 8 * w)
        self.mid = _block(8 * w, 8 * w)
        self.u4, self.d4 = nn.ConvTranspose2d(8 * w, 8 * w, 2, 2), _block(16 * w, 4 * w)
        self.u3, self.d3 = nn.ConvTranspose2d(4 * w, 4 * w, 2, 2), _block(8 * w, 2 * w)
        self.u2, self.d2 = nn.ConvTranspose2d(2 * w, 2 * w, 2, 2), _block(4 * w, w)
        self.u1, self.d1 = nn.ConvTranspose2d(w, w, 2, 2), _block(2 * w, w)
        self.out = nn.Conv2d(w, ch, 1)
        self.levels = 4

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        m = self.mid(F.max_pool2d(e4, 2))
        d4 = self.d4(torch.cat([self.u4(m), e4], 1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return x - self.out(d1)


def _pad_to(x, mult):
    H, W = x.shape[-2:]
    ph, pw = (-H) % mult, (-W) % mult
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, H, W


@torch.no_grad()
def apply(net: UNet, img: torch.Tensor) -> torch.Tensor:
    """Denoise a (3,H,W) image; pads to a multiple of 16 and crops back."""
    x, H, W = _pad_to(img.unsqueeze(0), 2 ** net.levels)
    return net(x)[0, :, :H, :W]


# ----------------------------------------------------------------------------- Neighbor2Neighbor
def neighbor_subsample(y: torch.Tensor):
    """Random neighbour sub-sampler: from every 2x2 cell pick two different, adjacent pixels.

    y: (B,C,H,W) with even H, W.  Returns g1(y), g2(y) of shape (B,C,H/2,W/2) and the index
    pair, so the same sampler can be applied to f(y).
    """
    B, C, H, W = y.shape
    h, w = H // 2, W // 2
    # the 4 positions of a 2x2 cell: 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1); adjacent pairs:
    pairs = torch.tensor([[0, 1], [0, 2], [1, 0], [1, 3], [2, 0], [2, 3], [3, 1], [3, 2]], device=y.device)
    choice = pairs[torch.randint(0, 8, (B, h, w), device=y.device)]  # (B,h,w,2)
    cells = F.unfold(y, 2, stride=2).view(B, C, 4, h, w)               # (B,C,4,h,w)

    def pick(k):
        idx = choice[..., k].unsqueeze(1).unsqueeze(2).expand(B, C, 1, h, w)
        return cells.gather(2, idx).squeeze(2)
    return pick(0), pick(1), choice


def _apply_sampler(x: torch.Tensor, choice: torch.Tensor):
    B, C, H, W = x.shape
    h, w = H // 2, W // 2
    cells = F.unfold(x, 2, stride=2).view(B, C, 4, h, w)
    out = []
    for k in (0, 1):
        idx = choice[..., k].unsqueeze(1).unsqueeze(2).expand(B, C, 1, h, w)
        out.append(cells.gather(2, idx).squeeze(2))
    return out


def train_n2n(images: List[torch.Tensor], iters: int = 2000, crop: int = 128, batch: int = 4,
              lr: float = 1e-3, gamma_max: float = 2.0, width: int = 32, device: str = "cuda",
              seed: int = 0, log_every: int = 250) -> UNet:
    """Train a U-Net with the Neighbor2Neighbor loss on a list of noisy (3,H,W) images in [0,1]."""
    g = torch.Generator().manual_seed(seed)
    rnd = random.Random(seed)
    net = UNet(images[0].shape[0], width).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(iters * 0.5), int(iters * 0.75)], gamma=0.5)
    crop = int(crop) - int(crop) % (2 ** (net.levels + 1))  # sub-images must still pass the U-Net
    crop = max(crop, 2 ** (net.levels + 1))
    imgs = []
    for im in images:  # tiny images: pad up to one crop so the sampler and the U-Net still apply
        im = im.to(device)
        ph, pw = max(crop - im.shape[1], 0), max(crop - im.shape[2], 0)
        if ph or pw:
            im = F.pad(im.unsqueeze(0), (0, pw, 0, ph), mode="replicate")[0]
        imgs.append(im)
    for it in range(1, iters + 1):
        batch_t = []
        for _ in range(batch):
            im = imgs[rnd.randrange(len(imgs))]
            _, H, W = im.shape
            m = 2 ** (net.levels + 1)
            c = min(crop, H - H % m, W - W % m)
            i = int(torch.randint(0, H - c + 1, (1,), generator=g))
            j = int(torch.randint(0, W - c + 1, (1,), generator=g))
            p = im[:, i:i + c, j:j + c]
            if rnd.random() < 0.5:
                p = p.flip(-1)
            batch_t.append(p)
        c = min(p.shape[-1] for p in batch_t)
        y = torch.stack([p[:, :c, :c] for p in batch_t])
        g1, g2, choice = neighbor_subsample(y)
        with torch.no_grad():
            fy = net(y)
            fg1, fg2 = _apply_sampler(fy, choice)
        out = net(g1)
        gamma = gamma_max * it / iters
        loss = F.mse_loss(out, g2) + gamma * F.mse_loss(out - g2, fg1 - fg2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if log_every and (it % log_every == 0 or it == iters):
            print(f"[n2n] {it}/{iters} loss {loss.item():.6f}")
    net.eval()
    return net


def n2n_scale(images: List[torch.Tensor], target_mean: float = 0.4) -> float:
    """One scene-wide gain that lifts dark images to a mid-grey level the U-Net trains well on."""
    m = float(torch.stack([im.mean() for im in images]).mean())
    return target_mean / max(m, 1e-4)


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


__all__ = ["UNet", "train_n2n", "apply", "neighbor_subsample", "n2n_scale", "count_params"]

