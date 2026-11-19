import types
import torch
from halogs.litalite import (guided_filter_self, noise_sigma, nova_gamma, nova_maps, nova_target,
                             gaussian_pixel_coords, sh_noise_penalty)


def test_guided_filter_smooths_noise_keeps_edge():
    torch.manual_seed(0)
    x = torch.zeros(1, 32, 32); x[:, :, 16:] = 0.8
    n = x + 0.02 * torch.randn_like(x)
    y = guided_filter_self(n, r=3, eps=1e-3)
    assert (y - x).abs()[:, 4:-4, 4:12].std() < (n - x).abs()[:, 4:-4, 4:12].std()
    assert float(y[:, :, 20:].mean() - y[:, :, :12].mean()) > 0.7


def test_noise_sigma_estimate():
    torch.manual_seed(0)
    img = 0.3 + 0.05 * torch.randn(3, 128, 128)
    assert 0.02 < noise_sigma(img) < 0.05   # channel mean of 3 -> sigma/sqrt(3) ~ 0.029


def test_gamma_search_and_target():
    torch.manual_seed(0)
    imgs = [torch.rand(3, 32, 32) * 0.1 for _ in range(3)]
    maps = [nova_maps(im, 0.4 / float(im.mean()), r=4)[0] for im in imgs]
    g = nova_gamma(maps, 0.5)
    v = torch.cat([m.flatten() for m in maps])
    assert abs(float((v ** g).mean()) - 0.5) < 1e-3 and 0.3 < g < 1.0
    t = nova_target(imgs[0], 0.4 / float(imgs[0].mean()), g, r=4)
    assert t.shape == imgs[0].shape and 0.35 < float(t.mean()) < 0.65 and float(t.max()) <= 1


def test_sh_noise_penalty_grad():
    N = 5
    pc = types.SimpleNamespace(get_xyz=torch.zeros(N, 3), _features_rest=torch.ones(N, 15, 3, requires_grad=True))
    cam = types.SimpleNamespace(full_proj_transform=torch.eye(4))
    assert torch.allclose(gaussian_pixel_coords(pc, cam), torch.zeros(N, 2))
    nm = torch.full((8, 8), 0.1)
    loss = sh_noise_penalty(pc, cam, nm, torch.ones(N, dtype=torch.bool))
    loss.backward()
    assert abs(float(loss) - 0.1 * 45) < 1e-4 and pc._features_rest.grad is not None


def test_reinhard_key_scale_invariant_and_ordering():
    from halogs.litalite import reinhard_key
    torch.manual_seed(0)
    even = [torch.rand(3, 64, 64) * 0.2 + 0.05]                              # evenly lit
    y = torch.full((3, 64, 64), 0.02); y[:, :8, :8] = 0.6                   # mostly dark, a few bright pixels
    dark_hc = [y + 0.002 * torch.rand_like(y)]
    d1, k1, _ = reinhard_key(even, clamp=(0, 1))
    d2, k2, _ = reinhard_key([im * 0.5 for im in even], clamp=(0, 1))     # capture exposure must not matter
    assert abs(k1 - k2) / k1 < 0.05
    d3, k3, _ = reinhard_key(dark_hc, clamp=(0, 1))
    assert k3 < k1                                                         # mostly-dark high-contrast -> lower key
    dd, _, _ = reinhard_key(dark_hc)
    assert 0.25 <= dd <= 0.6


def test_whitepatch_exposure_factor_hits_mean_and_keeps_shape():
    from halogs.litalite import white_patch_mean, exposure_factor_for_mean, nova_target
    torch.manual_seed(0)
    srcs = [(torch.rand(3, 48, 64) * 0.06).clamp(0, 1) for _ in range(3)]
    m = white_patch_mean(srcs, r=4)
    assert 0.0 < m < 1.0
    raws = [nova_target(im, 0.4 / float(im.mean()), 0.72, r=4, scale=1.0) for im in srcs]   # already clipped <= 1
    c = exposure_factor_for_mean(raws, 0.3)
    got = sum(float((c * r).clamp(0, 1).mean()) for r in raws) / 3
    assert abs(got - 0.3) < 0.01
    a = nova_target(srcs[0], 10.0, 0.72, r=4, scale=0.5); b = nova_target(srcs[0], 10.0, 0.72, r=4, scale=1.0)
    assert float(a.mean()) < float(b.mean())


def test_pearson_depth_and_robust_whitepatch():
    from halogs.litalite import pearson_depth_loss, white_patch_mean
    torch.manual_seed(0)
    d = torch.rand(1, 16, 16)
    assert float(pearson_depth_loss(d, 3 * d[0] + 2)) < 1e-5            # scale/shift invariant
    assert float(pearson_depth_loss(d, -d[0])) > 1.9
    r = torch.rand(1, 16, 16, requires_grad=True); pearson_depth_loss(r, d[0]).backward(); assert r.grad is not None
    base = [(torch.rand(3, 48, 64) * 0.06).clamp(0, 1) for _ in range(3)]
    lamp = [b.clone() for b in base]
    for b in lamp: b[:, :6, :6] = 0.98                                  # a light source in the dark photo
    m0, m1 = white_patch_mean(base, r=4), white_patch_mean(lamp, r=4)
    assert abs(m0 - m1) < 0.05                                           # the lamp does not change the white point much
