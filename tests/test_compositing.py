import torch

from halogs.apsf import GlowKernel
from halogs.compositing import composite, direct_conv2d, fft_conv2d


def test_fft_matches_direct_small_and_large_kernels():
    torch.manual_seed(0)
    for (H, W, K) in [(37, 53, 9), (64, 80, 65), (50, 50, 51)]:
        img = torch.rand(3, H, W)
        ker = torch.rand(K, K)
        a = fft_conv2d(img, ker)
        b = direct_conv2d(img, ker)
        assert a.shape == (3, H, W)
        assert (a - b).abs().max().item() < 1e-3 * ker.sum().item()


def test_convolution_is_linear_and_energy_preserving_in_interior():
    torch.manual_seed(0)
    k = GlowKernel(radius=16, focal_px=400.0).kernel().detach()
    img = torch.zeros(3, 128, 128)
    img[:, 64, 64] = 1.0  # impulse well inside
    g = fft_conv2d(img, k)
    assert abs(g.sum().item() - 3.0) < 1e-4          # kernel sums to 1 per channel
    assert torch.allclose(g[0, 48:81, 48:81], k, atol=1e-5)  # impulse response == kernel


def test_composite_clamps_and_returns_glow():
    scene = torch.rand(3, 40, 40) * 0.5
    emission = torch.zeros(3, 40, 40)
    emission[:, 20, 20] = 50.0
    k = GlowKernel(radius=8, focal_px=400.0).kernel().detach()
    comp, glow = composite(scene, emission, k, gain=1.0, clamp=True)
    assert comp.max().item() <= 1.0 and comp.min().item() >= 0.0
    assert glow[0, 20, 20] > glow[0, 20, 27] > 0.0
    comp2, _ = composite(scene, emission, k, gain=0.5, clamp=False)
    assert torch.allclose(comp2, 0.5 * (scene + glow))


def test_gradients_reach_emission_and_kernel():
    kmod = GlowKernel(radius=8, focal_px=400.0)
    scene = torch.rand(3, 32, 32)
    emission = torch.rand(3, 32, 32, requires_grad=True)
    comp, _ = composite(scene, emission, kmod.kernel(), clamp=False)
    comp.mean().backward()
    assert emission.grad is not None and emission.grad.abs().sum() > 0
    assert kmod.T_raw.grad is not None
