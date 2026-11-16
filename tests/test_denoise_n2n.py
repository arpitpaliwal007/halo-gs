"""The Neighbor2Neighbor U-Net must reduce noise without seeing a clean image."""
import torch

from halogs.denoise_n2n import UNet, apply, count_params, neighbor_subsample, train_n2n


def _scene(H=64, W=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
    clean = torch.stack([0.2 + 0.5 * xx, 0.3 + 0.4 * yy, 0.5 * (xx > 0.5).float() + 0.2])
    noisy = (clean + (0.02 + 0.08 * clean) * torch.randn(clean.shape, generator=g)).clamp(0, 1)  # signal-dependent
    return clean, noisy


def test_subsampler_shapes_and_independence():
    y = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).view(2, 3, 4, 4)
    g1, g2, _ = neighbor_subsample(y)
    assert g1.shape == g2.shape == (2, 3, 2, 2)
    assert (g1 != g2).all()  # always two different pixels of each 2x2 cell


def test_unet_shapes_and_size():
    net = UNet(3, 32)
    assert 2.0e6 < count_params(net) < 5.0e6
    out = apply(net, torch.rand(3, 37, 50))  # odd size: padded and cropped back
    assert out.shape == (3, 37, 50)


def test_n2n_reduces_noise():
    torch.manual_seed(0)
    pairs = [_scene(seed=s) for s in range(4)]
    net = train_n2n([n for _, n in pairs], iters=800, crop=64, batch=4, width=16, lr=1e-3, device="cpu", log_every=0)
    clean, noisy = _scene(seed=99)  # unseen noise realisation
    den = apply(net, noisy).clamp(0, 1)
    mse_in = (noisy - clean).pow(2).mean().item()
    mse_out = (den - clean).pow(2).mean().item()
    assert mse_out < 0.5 * mse_in, (mse_in, mse_out)
