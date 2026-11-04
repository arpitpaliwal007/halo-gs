"""Emission must follow Gaussian *identity* through prune / clone / split.

Runs on CPU against the real GaussianModel from a gaussian-splatting checkout
(CUDA extensions stubbed).  Skipped when no checkout is available.
"""
import torch
import torch.nn as nn
import pytest


def _cpu_build_rotation(r):
    norm = torch.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2 + r[:, 2] ** 2 + r[:, 3] ** 2)
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device=r.device)
    r_, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - r_ * z); R[:, 0, 2] = 2 * (x * z + r_ * y)
    R[:, 1, 0] = 2 * (x * y + r_ * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - r_ * x)
    R[:, 2, 0] = 2 * (x * z - r_ * y); R[:, 2, 1] = 2 * (y * z + r_ * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def make_model(gs_root, n=10, monkeypatch=None):
    import halogs.gaussians as G
    if monkeypatch is not None:
        monkeypatch.setattr(G, "build_rotation", _cpu_build_rotation)
    m = G.EmissiveGaussianModel(3, emission_lr=0.01)
    torch.manual_seed(0)
    m._xyz = nn.Parameter(torch.randn(n, 3))
    m._features_dc = nn.Parameter(torch.randn(n, 1, 3))
    m._features_rest = nn.Parameter(torch.randn(n, 15, 3))
    m._opacity = nn.Parameter(torch.randn(n, 1))
    m._scaling = nn.Parameter(torch.randn(n, 3) * 0.1 - 3)
    m._rotation = nn.Parameter(torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1))
    m._emission = nn.Parameter(torch.arange(n, dtype=torch.float32)[:, None].repeat(1, 3))  # emission == identity
    m.max_radii2D = torch.zeros(n)
    m.xyz_gradient_accum = torch.zeros(n, 1)
    m.denom = torch.zeros(n, 1)
    m.percent_dense = 0.01
    m.spatial_lr_scale = 1.0
    groups = [{"params": [m._xyz], "lr": 1e-3, "name": "xyz"},
              {"params": [m._features_dc], "lr": 1e-3, "name": "f_dc"},
              {"params": [m._features_rest], "lr": 1e-3, "name": "f_rest"},
              {"params": [m._opacity], "lr": 1e-3, "name": "opacity"},
              {"params": [m._scaling], "lr": 1e-3, "name": "scaling"},
              {"params": [m._rotation], "lr": 1e-3, "name": "rotation"}]
    m.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
    m._add_emission_group()
    # one optimizer step so every group has Adam state
    for g in m.optimizer.param_groups:
        g["params"][0].grad = torch.randn_like(g["params"][0])
    m.optimizer.step()
    m.optimizer.zero_grad(set_to_none=True)
    return m


def emission_ids(m):
    return [int(round(v)) for v in m._emission[:, 0].tolist()]


def test_prune_keeps_identity(gs_root, monkeypatch):
    m = make_model(gs_root, 10, monkeypatch)
    mask = torch.zeros(10, dtype=torch.bool)
    mask[[1, 4, 7]] = True  # prune B, E, H
    m.tmp_radii = None
    m.prune_points(mask)
    assert emission_ids(m) == [0, 2, 3, 5, 6, 8, 9]
    assert m._xyz.shape[0] == 7 and m.max_radii2D.shape[0] == 7
    # optimizer moments pruned alongside
    st = m.optimizer.state[m._emission]
    assert st["exp_avg"].shape[0] == 7
    # same optimizer param object is registered
    assert any(g["params"][0] is m._emission for g in m.optimizer.param_groups if g["name"] == "emission")


def test_clone_children_inherit_emission(gs_root, monkeypatch):
    m = make_model(gs_root, 6, monkeypatch)
    m.tmp_radii = None
    grads = torch.zeros(6, 1)
    grads[[2, 5]] = 1.0
    m.densify_and_clone(grads, 0.5, scene_extent=100.0)  # all scales <= percent_dense*extent -> clone
    assert emission_ids(m) == [0, 1, 2, 3, 4, 5, 2, 5]
    assert m.optimizer.state[m._emission]["exp_avg"].shape[0] == 8
    assert m.optimizer.state[m._emission]["exp_avg"][6:].abs().sum() == 0  # fresh moments for children


def test_split_children_inherit_and_parent_removed(gs_root, monkeypatch):
    m = make_model(gs_root, 5, monkeypatch)
    m.tmp_radii = None
    grads = torch.zeros(5, 1)
    grads[3] = 1.0
    m.densify_and_split(grads, 0.5, scene_extent=1e-6, N=2)  # scale > percent_dense*extent -> split
    ids = emission_ids(m)
    assert sorted(ids) == [0, 1, 2, 3, 3, 4] and ids.count(3) == 2
    assert m._xyz.shape[0] == 6


def test_densify_and_prune_equal_count_replacement(gs_root, monkeypatch):
    """Count unchanged but identities change: the old count-based resync would silently misalign."""
    m = make_model(gs_root, 6, monkeypatch)
    # opacity -> prune row 0 (very negative), split row 5
    with torch.no_grad():
        m._opacity[0] = -20.0
        m._opacity[1:] = 10.0
    m.xyz_gradient_accum[5] = 1.0
    m.denom[:] = 1.0
    m.densify_and_prune(max_grad=0.5, min_opacity=0.005, extent=1e-6, max_screen_size=None, radii=torch.zeros(6))
    ids = emission_ids(m)
    assert 0 not in ids and ids.count(5) == 2 and len(ids) == 6
    assert m._emission.shape[0] == m._xyz.shape[0] == m.optimizer.state[m._emission]["exp_avg"].shape[0]


def test_save_load_emission_roundtrip(gs_root, monkeypatch, tmp_path):
    m = make_model(gs_root, 4, monkeypatch)
    p = tmp_path / "emission.pt"
    m.save_emission(str(p))
    m2 = make_model(gs_root, 4, monkeypatch)
    with torch.no_grad():
        m2._emission.zero_()
    m2.load_emission(str(p))
    assert emission_ids(m2) == [0, 1, 2, 3]
    m3 = make_model(gs_root, 5, monkeypatch)
    with pytest.raises(RuntimeError):
        m3.load_emission(str(p), strict=True)
