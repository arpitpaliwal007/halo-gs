import types
import torch
from halogs.litalite import apply_luma_curve, he_curve, illum_prior_loss, relative_photometric_loss, spatial_consistency_loss


def _cams(n=3, mean=0.05):
    torch.manual_seed(0)
    return [types.SimpleNamespace(original_image=(torch.rand(3, 32, 32) * 2 * mean).clamp(0, 1)) for _ in range(n)]


def test_he_curve_monotone_brightens_and_blend():
    cams = _cams()
    lut = he_curve(cams, theta=0.4, blend=0.5)
    assert lut.numel() == 257 and float(lut[0]) == 0
    assert (lut[1:] - lut[:-1] >= -1e-7).all()
    assert float(lut[13]) > 0.05 * 3   # a 0.05 input is lifted a lot
    lin = he_curve(cams, theta=0.4, blend=0.0)
    m = float(torch.cat([c.original_image.mean(0).flatten() for c in cams]).mean())
    assert abs(float(lin[64]) - min(0.25 * 0.4 / m, 1)) < 1e-4  # blend 0 == original linear gain


def test_luma_curve_keeps_hue():
    img = torch.tensor([0.02, 0.04, 0.06]).view(3, 1, 1).repeat(1, 4, 4)
    out = apply_luma_curve(img, he_curve(_cams(), 0.4, 0.5))
    r = out[:, 0, 0] / out[:, 0, 0].sum()
    assert torch.allclose(r, torch.tensor([1 / 6, 2 / 6, 3 / 6]), atol=1e-4) and float(out.mean()) > float(img.mean())


def test_spatial_consistency_zero_for_scaled_copy():
    inp = torch.rand(3, 32, 32) * 0.1
    out = inp * (0.5 / float(inp.mean()))
    assert float(spatial_consistency_loss(out, inp)) < 1e-10
    assert float(spatial_consistency_loss(torch.full_like(inp, 0.5), inp)) > 0


def test_illum_prior_and_relative_l1():
    dark = torch.rand(3, 16, 16) * 0.1
    target = (dark * 5).clamp(0, 1)
    ill = torch.full((3, 16, 16), 0.2, requires_grad=True)
    l = illum_prior_loss(ill, dark, target, w_smooth=0.001)
    l.backward()
    assert float(l) < 0.01 and ill.grad is not None   # max_c(dark)/max_c(target) == 0.2
    p = torch.rand(3, 16, 16, requires_grad=True)
    loss, l1 = relative_photometric_loss(p, torch.rand(3, 16, 16))
    loss.backward()
    assert p.grad is not None and float(loss) > 0
