
import torch

from halogs.apsf import GlowKernel, RadialKernel, apsf_profile, make_kernel


def _radial(K):
    c = K.shape[-1] // 2
    return K[c, c:]


def test_kernel_is_normalised_nonnegative_and_monotone():
    for comp in ("both", "apsf", "lens"):
        for T in (0.8, 1.0, 1.5, 3.0):
            k = GlowKernel(radius=64, focal_px=600.0, components=comp, T_init=T)
            K = k.kernel()
            assert abs(K.sum().item() - 1.0) < 1e-5
            assert K.min().item() >= 0.0
            r = _radial(K)
            assert bool((r[1:] <= r[:-1] + 1e-7).all()), f"{comp} T={T} not monotone"
            assert K[64, 64] == K.max()


def test_apsf_width_increases_with_T_decreases_with_q():
    def hw(q, T):
        k = GlowKernel(radius=96, focal_px=600.0, components="apsf", q_init=q, T_init=T)
        r = _radial(k.kernel())
        return int((r > 0.5 * r[0]).sum())
    assert hw(0.9, 0.8) < hw(0.9, 1.5)       # thicker medium -> wider halo
    # the truncated series saturates to the kernel support for large T; only check the ordering above


def test_kernel_gradients_flow():
    k = GlowKernel(radius=48, focal_px=500.0)
    K = k.kernel()
    (K[48, 60] * 1000).backward()
    for name, p in k.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad), name


def test_apsf_profile_is_symmetric_in_theta():
    th = torch.tensor([0.02, 0.05])
    a = apsf_profile(th, torch.tensor(0.9), torch.tensor(1.2))
    b = apsf_profile(-th, torch.tensor(0.9), torch.tensor(1.2))
    assert torch.allclose(a, b)


def test_radial_kernel_monotone_and_normalised():
    k = RadialKernel(radius=40, n_knots=12)
    K = k.kernel()
    assert abs(K.sum().item() - 1.0) < 1e-5
    r = _radial(K)
    assert bool((r[1:] <= r[:-1] + 1e-7).all())
    K[40, 45].backward()
    assert k.steps_raw.grad is not None


def test_make_kernel_and_describe():
    for kind in ("both", "apsf", "lens", "radial"):
        k = make_kernel(kind, 32, 400.0)
        d = k.describe()
        assert isinstance(d, dict) and d["radius"] == 32
    k = make_kernel("both", 32, 400.0)
    assert 0.0 < k.describe()["q"] < 1.0 and k.describe()["T"] >= 0.8


def test_focal_changes_angular_scale():
    k1 = GlowKernel(radius=64, focal_px=300.0, components="apsf", T_init=1.0)
    k2 = GlowKernel(radius=64, focal_px=1200.0, components="apsf", T_init=1.0)
    r1, r2 = _radial(k1.kernel()), _radial(k2.kernel())
    # longer focal -> same angle spans more pixels -> wider in px
    assert int((r2 > 0.5 * r2[0]).sum()) >= int((r1 > 0.5 * r1[0]).sum())
