"""The brightened pseudo-target must keep R:G:B ratios even where it saturates."""
import torch

from halogs.litalite import brighten_target


def _ratios(x):
    return x / x.sum(0, keepdim=True).clamp_min(1e-8)


def test_hue_safe_preserves_ratios_when_saturating():
    torch.manual_seed(0)
    img = torch.rand(3, 16, 16) * 0.2
    img[:, 4:8, 4:8] = torch.tensor([0.30, 0.15, 0.05])[:, None, None]  # bright, coloured lamp
    t = brighten_target(img, theta=0.40, hue_safe=True)
    assert t.max() <= 1.0 + 1e-6
    assert torch.allclose(_ratios(t), _ratios(img), atol=1e-5)


def test_legacy_clip_changes_hue_where_saturated():
    img = torch.zeros(3, 4, 4) + 0.02
    img[:, 0, 0] = torch.tensor([0.30, 0.15, 0.05])
    t = brighten_target(img, theta=0.40, hue_safe=False)
    assert not torch.allclose(_ratios(t)[:, 0, 0], _ratios(img)[:, 0, 0], atol=1e-3)


def test_unsaturated_pixels_identical_and_gain():
    img = torch.full((3, 8, 8), 0.1)
    a = brighten_target(img, theta=0.40, hue_safe=True)
    b = brighten_target(img, theta=0.40, hue_safe=False)
    assert torch.allclose(a, b) and torch.allclose(a.mean(), torch.tensor(0.40))
    c = brighten_target(img, gain=2.0)
    assert torch.allclose(c, img * 2.0)
