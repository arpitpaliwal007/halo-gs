"""LitaLite illumination must follow Gaussian *identity* through prune / clone / split.

This is the bug found in the original litalite.py: illumination lived in a side
tensor that was truncated or mean-padded by *count*, so after pruning a
surviving Gaussian could carry another Gaussian's illumination, children did
not inherit their parent's, and an equal-count replace went unnoticed.
"""
import torch
import torch.nn as nn
import pytest

from test_lifecycle import _cpu_build_rotation


def make_model(n, monkeypatch):
    import halogs.gaussians as G
    monkeypatch.setattr(G, "build_rotation", _cpu_build_rotation)
    m = G.EmissiveGaussianModel(3, emission_lr=0.01, with_illum=True, illum_lr=0.01)
    torch.manual_seed(0)
    m._xyz = nn.Parameter(torch.randn(n, 3))
    m._features_dc = nn.Parameter(torch.randn(n, 1, 3))
    m._features_rest = nn.Parameter(torch.randn(n, 15, 3))
    m._opacity = nn.Parameter(torch.randn(n, 1))
    m._scaling = nn.Parameter(torch.randn(n, 3) * 0.1 - 3)
    m._rotation = nn.Parameter(torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1))
    ids = torch.arange(n, dtype=torch.float32)[:, None]
    m._emission = nn.Parameter(ids.repeat(1, 3))
    m._illum = nn.Parameter(ids * 10.0)  # illumination raw value == 10 * identity
    m.max_radii2D = torch.zeros(n)
    m.xyz_gradient_accum = torch.zeros(n, 1)
    m.denom = torch.zeros(n, 1)
    m.percent_dense = 0.01
    m.spatial_lr_scale = 1.0
    groups = [{"params": [getattr(m, a)], "lr": 1e-3, "name": nm} for a, nm in
              [("_xyz", "xyz"), ("_features_dc", "f_dc"), ("_features_rest", "f_rest"),
               ("_opacity", "opacity"), ("_scaling", "scaling"), ("_rotation", "rotation")]]
    m.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
    m._add_emission_group()
    for g in m.optimizer.param_groups:
        g["params"][0].grad = torch.randn_like(g["params"][0])
    m.optimizer.step()
    m.optimizer.zero_grad(set_to_none=True)
    m.tmp_radii = None
    return m


def illum_ids(m):
    return [int(round(v / 10.0)) for v in m._illum[:, 0].tolist()]


def emission_ids(m):
    return [int(round(v)) for v in m._emission[:, 0].tolist()]


def test_illum_group_registered(gs_root, monkeypatch):
    m = make_model(4, monkeypatch)
    names = [g["name"] for g in m.optimizer.param_groups]
    assert names.count("illum") == 1 and names.count("emission") == 1
    assert m.get_illum.shape == (4, 1) and (m.get_illum > 0).all() and (m.get_illum <= 1).all()


def test_prune_keeps_illum_identity_and_moments(gs_root, monkeypatch):
    m = make_model(10, monkeypatch)
    before = m.optimizer.state[m._illum]["exp_avg"].clone()
    mask = torch.zeros(10, dtype=torch.bool)
    mask[[1, 4, 7]] = True
    m.prune_points(mask)
    keep = [0, 2, 3, 5, 6, 8, 9]
    assert illum_ids(m) == keep == emission_ids(m)
    # Adam moments are kept for surviving rows (the old resync rebuilt Adam from scratch)
    assert torch.equal(m.optimizer.state[m._illum]["exp_avg"], before[keep])


def test_clone_and_split_children_inherit_illum(gs_root, monkeypatch):
    m = make_model(6, monkeypatch)
    grads = torch.zeros(6, 1)
    grads[[2, 5]] = 1.0
    m.densify_and_clone(grads, 0.5, scene_extent=100.0)
    assert illum_ids(m) == [0, 1, 2, 3, 4, 5, 2, 5]
    assert m.optimizer.state[m._illum]["exp_avg"][6:].abs().sum() == 0

    m = make_model(5, monkeypatch)
    grads = torch.zeros(5, 1)
    grads[3] = 1.0
    m.densify_and_split(grads, 0.5, scene_extent=1e-6, N=2)
    ids = illum_ids(m)
    assert sorted(ids) == [0, 1, 2, 3, 3, 4] and ids == emission_ids(m)


def test_equal_count_replacement_detected(gs_root, monkeypatch):
    """Count stays 6 but identities change: the old count-based resync left illumination misaligned."""
    m = make_model(6, monkeypatch)
    with torch.no_grad():
        m._opacity[0] = -20.0
        m._opacity[1:] = 10.0
    m.xyz_gradient_accum[5] = 1.0
    m.denom[:] = 1.0
    m.densify_and_prune(max_grad=0.5, min_opacity=0.005, extent=1e-6, max_screen_size=None, radii=torch.zeros(6))
    ids = illum_ids(m)
    assert len(ids) == 6 and 0 not in ids and ids.count(5) == 2 and ids == emission_ids(m)
    assert m._illum.shape[0] == m._xyz.shape[0] == m.optimizer.state[m._illum]["exp_avg"].shape[0]


def test_save_load_roundtrip_with_illum(gs_root, monkeypatch, tmp_path):
    m = make_model(4, monkeypatch)
    p = tmp_path / "emission.pt"
    m.save_emission(str(p))
    m2 = make_model(4, monkeypatch)
    with torch.no_grad():
        m2._illum.zero_()
    m2.load_emission(str(p))
    assert illum_ids(m2) == [0, 1, 2, 3]


def test_capture_restore_keeps_illum(gs_root, monkeypatch):
    m = make_model(3, monkeypatch)
    blob = m.capture()
    assert len(blob) == 3 and torch.equal(blob[2], m._illum.detach())
