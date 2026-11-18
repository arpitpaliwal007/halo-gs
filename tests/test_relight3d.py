import types
import torch
from halogs.litalite import Relight3D, illum_knn_smooth, illum_white_point, fit_relight_gamma


def _pc(n=200, c=3):
    torch.manual_seed(0)
    L = torch.rand(n, c, requires_grad=True)
    return types.SimpleNamespace(get_xyz=torch.randn(n, 3), get_illum=L)


def test_T_range_and_white_point(tmp_path):
    pc = _pc()
    k = illum_white_point(pc, 0.99)
    T = Relight3D(k, 0.7).T(pc.get_illum)
    assert T.min() > 0 and T.max() <= 1 + 1e-6
    assert float((T >= 0.999).float().mean()) > 0.005        # the best-lit ~1% become fully lit
    r = Relight3D(k, 0.7); r.save(tmp_path / "relight.json")
    r2 = Relight3D.load(tmp_path / "relight.json")
    assert abs(r2.k - k) < 1e-9 and abs(r2.gamma - 0.7) < 1e-9


def test_knn_smooth_grad_and_zero_when_constant():
    pc = _pc()
    loss = illum_knn_smooth(pc, 64, 128)
    loss.backward()
    assert float(loss) > 0 and pc.get_illum.grad is not None
    pc2 = types.SimpleNamespace(get_xyz=torch.randn(100, 3), get_illum=torch.full((100, 3), 0.3))
    assert float(illum_knn_smooth(pc2, 32, 64)) < 1e-9


def test_fit_gamma_hits_target(monkeypatch):
    import halogs.litalite as LL
    pc = _pc(); pc.get_illum = pc.get_illum.detach() * 0.2
    refl = torch.full((200, 3), 0.8)
    monkeypatch.setattr(LL, "reflectance_colours", lambda pc_, cam: refl)
    monkeypatch.setattr(LL, "_render_kwargs", lambda fn: {})
    fake_render = lambda cam, pc_, pipe, bg, override_color: {"render": override_color.mean(0)[:, None, None].expand(3, 4, 4)}
    k = illum_white_point(pc)
    g = fit_relight_gamma(fake_render, [None, None], pc, None, None, k, target_mean=0.5)
    m = float((refl * Relight3D(k, g).T(pc.get_illum)).mean())
    assert abs(m - 0.5) < 0.01
