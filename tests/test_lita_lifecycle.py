"""Emission follows Gaussian identity through prune / clone / split on the LITA-GS backbone.

Runs on CPU against the real LITA-GS GaussianModel (CUDA extensions stubbed).
Skipped without a LITA-GS checkout ($LITA_ROOT).  Must run in a separate pytest
process from the vanilla-backend tests (module names clash): see tests/README.
"""
import pytest
import torch
import torch.nn as nn


def _cpu_build_rotation(r):
    norm = torch.sqrt((r ** 2).sum(1))
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device=r.device)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def make_model(lita_root, n, monkeypatch):
    import halogs.lita.model as M
    monkeypatch.setattr(M, "build_rotation", _cpu_build_rotation)
    # LITA's __init__ builds tonemapper/denoiser with .cuda(); patch to no-op on CPU
    monkeypatch.setattr(nn.Module, "cuda", lambda self, *a, **k: self)
    m = M.EmissiveLitaModel(sh_degree=3, depth_feat_dim=1, prior_feat_dim=1, noise_feat_dim=3, illu_feat_dim=3,
                            denoiser_stage=1, denoiser_layers=2, denoiser_channel=8, denoiser_mode=1, emission_lr=0.01)
    torch.manual_seed(0)
    m._xyz = nn.Parameter(torch.randn(n, 3))
    m._features_dc = nn.Parameter(torch.randn(n, 1, 3))
    m._features_rest = nn.Parameter(torch.randn(n, 15, 3))
    m._features_depth = nn.Parameter(torch.rand(n, 1))
    m._features_prior = nn.Parameter(torch.rand(n, 1))
    m._features_noise = nn.Parameter(torch.rand(n, 3))
    m._features_illu = nn.Parameter(torch.rand(n, 3))
    m._opacity = nn.Parameter(torch.randn(n, 1))
    m._scaling = nn.Parameter(torch.randn(n, 3) * 0.1 - 3)
    m._rotation = nn.Parameter(torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1))
    m._features_emission = nn.Parameter(torch.arange(n, dtype=torch.float32)[:, None].repeat(1, 3))
    m.max_radii2D = torch.zeros(n)
    m.min_radii2D = torch.zeros(n)
    m.xyz_gradient_accum = torch.zeros(n, 1)
    m.denom = torch.zeros(n, 1)
    m.percent_dense = 0.01
    m.spatial_lr_scale = 1.0
    groups = [{"params": [getattr(m, a)], "lr": 1e-3, "name": nme} for a, nme in [
        ("_xyz", "xyz"), ("_features_dc", "f_dc"), ("_features_rest", "f_rest"), ("_features_depth", "f_depth"),
        ("_features_prior", "f_prior"), ("_features_noise", "f_noise"), ("_features_illu", "f_illu"),
        ("_opacity", "opacity"), ("_scaling", "scaling"), ("_rotation", "rotation")]]
    groups.append({"params": list(m.tonemapper.parameters()), "lr": 1e-4, "name": "tonemapper"})
    groups.append({"params": list(m.denoiser.parameters()), "lr": 1e-4, "name": "denoiser"})
    m.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
    m.optimizer.add_param_group({"params": [m._features_emission], "lr": 0.01, "name": "f_emission"})
    for g in m.optimizer.param_groups:
        for p in g["params"]:
            p.grad = torch.randn_like(p)
    m.optimizer.step()
    m.optimizer.zero_grad(set_to_none=True)
    return m


def ids(m):
    return [int(round(v)) for v in m._features_emission[:, 0].tolist()]


def test_prune_keeps_identity(lita_root, monkeypatch):
    m = make_model(lita_root, 10, monkeypatch)
    mask = torch.zeros(10, dtype=torch.bool)
    mask[[1, 4, 7]] = True
    m.prune_points(mask)
    assert ids(m) == [0, 2, 3, 5, 6, 8, 9]
    assert m._features_illu.shape[0] == 7 and m.min_radii2D.shape[0] == 7
    assert m.optimizer.state[m._features_emission]["exp_avg"].shape[0] == 7
    # tonemapper / denoiser params untouched
    assert all(p.grad is None for p in m.tonemapper.parameters())


def test_clone_and_split_inherit(lita_root, monkeypatch):
    m = make_model(lita_root, 6, monkeypatch)
    grads = torch.zeros(6, 1)
    grads[[2, 5]] = 1.0
    m.densify_and_clone(grads, 0.5, scene_extent=100.0)
    assert ids(m) == [0, 1, 2, 3, 4, 5, 2, 5]
    grads = torch.zeros(8, 1)
    grads[3] = 1.0
    m.densify_and_split(grads, 0.5, scene_extent=1e-6, N=2)
    out = ids(m)
    assert sorted(out) == [0, 1, 2, 2, 3, 3, 4, 5, 5] and out.count(3) == 2
    assert m._features_emission.shape[0] == m._xyz.shape[0] == m._features_illu.shape[0]


def test_densify_and_prune_equal_count(lita_root, monkeypatch):
    m = make_model(lita_root, 6, monkeypatch)
    with torch.no_grad():
        m._opacity[0] = -20.0
        m._opacity[1:] = 10.0
    m.xyz_gradient_accum[5] = 1.0
    m.denom[:] = 1.0
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    m.densify_and_prune(max_grad=0.5, min_opacity=0.005, extent=1e-6, max_screen_size=None)
    out = ids(m)
    assert 0 not in out and out.count(5) == 2 and len(out) == 6


def test_emission_roundtrip(lita_root, monkeypatch, tmp_path):
    m = make_model(lita_root, 4, monkeypatch)
    p = tmp_path / "emission.pt"
    m.save_emission(str(p))
    m2 = make_model(lita_root, 4, monkeypatch)
    with torch.no_grad():
        m2._features_emission.zero_()
    m2.load_emission(str(p))
    assert ids(m2) == [0, 1, 2, 3]
    with pytest.raises(RuntimeError):
        make_model(lita_root, 5, monkeypatch).load_emission(str(p), strict=True)
