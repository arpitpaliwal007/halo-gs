"""RGB illumination, per-view tone curve and target denoiser."""
import torch
import torch.nn as nn

from halogs.litalite import ToneCurve, bilateral


def test_tone_curve_starts_as_identity_and_is_monotone():
    t = ToneCurve(["a", "b"])
    x = torch.linspace(0.01, 1, 50)
    assert torch.allclose(t(x, "a"), x, atol=1e-6)
    with torch.no_grad():
        t.log_gain[0] = 0.3
        t.log_gamma[0] = -0.4
    y = t(x, "a")
    assert (y[1:] > y[:-1]).all()
    g, gm = t.params("unseen")  # test views use the mean curve
    assert torch.allclose(g, torch.exp(torch.tensor(0.15))) and torch.allclose(gm, torch.exp(torch.tensor(-0.2)))


def test_tone_curve_save_load(tmp_path):
    t = ToneCurve(["a", "b"])
    with torch.no_grad():
        t.log_gain[1] = 0.5
    t.save(str(tmp_path / "tone.pt"))
    t2 = ToneCurve.load(str(tmp_path / "tone.pt"), device="cpu")
    x = torch.rand(3, 4, 4)
    assert torch.allclose(t(x, "b"), t2(x, "b"))


def test_bilateral_denoises_but_keeps_edge():
    torch.manual_seed(0)
    clean = torch.zeros(3, 32, 32)
    clean[:, :, 16:] = 0.8  # vertical edge
    noisy = (clean + 0.03 * torch.randn_like(clean)).clamp(0, 1)
    out = bilateral(noisy, radius=2, sigma_s=1.5, sigma_r=0.08)
    assert (out - clean).pow(2).mean() < 0.5 * (noisy - clean).pow(2).mean()
    assert out[:, :, 14].mean() < 0.1 and out[:, :, 17].mean() > 0.7  # edge not smeared


def test_rgb_illum_lifecycle(gs_root, monkeypatch):
    import halogs.gaussians as G
    from test_lifecycle import _cpu_build_rotation
    monkeypatch.setattr(G, "build_rotation", _cpu_build_rotation)
    m = G.EmissiveGaussianModel(3, with_illum=True, illum_channels=3)
    n = 5
    m._xyz = nn.Parameter(torch.randn(n, 3)); m._features_dc = nn.Parameter(torch.randn(n, 1, 3))
    m._features_rest = nn.Parameter(torch.randn(n, 15, 3)); m._opacity = nn.Parameter(torch.randn(n, 1))
    m._scaling = nn.Parameter(torch.randn(n, 3) * 0.1 - 3)
    m._rotation = nn.Parameter(torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1))
    m._init_emission(n, "cpu"); m._init_illum(n, "cpu")
    assert m.get_illum.shape == (n, 3)
    with torch.no_grad():
        m._illum.copy_(torch.arange(n, dtype=torch.float32)[:, None].repeat(1, 3))
    m.max_radii2D = torch.zeros(n); m.xyz_gradient_accum = torch.zeros(n, 1); m.denom = torch.zeros(n, 1)
    m.percent_dense = 0.01; m.spatial_lr_scale = 1.0; m.tmp_radii = None
    groups = [{"params": [getattr(m, a)], "lr": 1e-3, "name": nm} for a, nm in
              [("_xyz", "xyz"), ("_features_dc", "f_dc"), ("_features_rest", "f_rest"),
               ("_opacity", "opacity"), ("_scaling", "scaling"), ("_rotation", "rotation")]]
    m.optimizer = torch.optim.Adam(groups, lr=0.0)
    m._add_emission_group()
    mask = torch.zeros(n, dtype=torch.bool); mask[1] = True
    m.prune_points(mask)
    grads = torch.zeros(4, 1); grads[2] = 1.0
    m.densify_and_clone(grads, 0.5, scene_extent=100.0)
    assert m._illum[:, 0].tolist() == [0, 2, 3, 4, 3] and m._illum.shape[1] == 3
