import torch
from halogs.litalite import ColourCurveCamera, colour_constancy_loss, exposure_loss, load_camera_model, pseudo_weight


def test_identity_init_and_monotone():
    m = ColourCurveCamera(["a", "b"], knots=8)
    x = torch.rand(3, 8, 8)
    assert torch.allclose(m(x, "a"), x, atol=1e-5) and torch.allclose(m(x, "zz"), x, atol=1e-5)
    with torch.no_grad():
        m.g_logits.normal_(); m.v_logits.normal_()
    r = torch.linspace(0, 1, 50).view(1, 1, 50).repeat(3, 1, 1)
    y = m(r, "a")[0, 0]
    assert (y[1:] - y[:-1] >= -1e-6).all() and abs(float(y[0])) < 1e-6


def test_tie_and_grads_and_save(tmp_path):
    m = ColourCurveCamera(["a"], knots=8)
    assert float(m.tie_loss()) == 0
    (m(torch.rand(3, 4, 4), "a").mean() + m.tie_loss()).backward()
    assert m.v_mat.grad is not None and m.g_logits.grad is not None
    p = tmp_path / "tone.pt"; m.save(p)
    m2 = load_camera_model(p, device="cpu")
    assert isinstance(m2, ColourCurveCamera)


def test_zero_dce_losses_and_fade():
    assert float(exposure_loss(torch.full((3, 32, 32), 0.6))) < 1e-10
    assert float(exposure_loss(torch.zeros(3, 32, 32))) > 0.3
    assert float(colour_constancy_loss(torch.full((3, 4, 4), 0.3))) == 0
    assert float(colour_constancy_loss(torch.stack([torch.ones(4, 4), torch.zeros(4, 4), torch.zeros(4, 4)]))) > 1
    assert pseudo_weight(5, 10, 20) == 1 and pseudo_weight(15, 10, 20) == 0.5 and pseudo_weight(30, 10, 20) == 0
    assert pseudo_weight(10**6, 0, 0) == 1
