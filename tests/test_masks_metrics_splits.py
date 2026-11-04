import math
import os
import tempfile
from types import SimpleNamespace

import torch

from halogs.losses import photometric_loss, ssim_map
from halogs.masks import glow_mask, source_mask
from halogs.metrics import all_metrics, psnr, ssim
from halogs.splits import load_split, make_split, save_split


def _lamp_image(H=96, W=128):
    img = torch.full((3, H, W), 0.1)
    img[:, 40:48, 60:68] = 1.0  # a saturated lamp
    return img


def test_source_and_glow_masks():
    img = _lamp_image()
    s = source_mask(img)
    assert s.sum().item() == 64
    g = glow_mask(img, radius=10)
    assert g.sum() > s.sum()
    assert g[0, 44, 64] == 1 and g[0, 44, 64 + 9] == 1 and g[0, 0, 0] == 0
    # isolated single bright pixel is ignored
    img2 = torch.full((3, 64, 64), 0.1)
    img2[:, 10, 10] = 1.0
    assert glow_mask(img2, radius=5).sum().item() == 0


def test_metrics_masked_vs_full():
    gt = _lamp_image()
    pred = gt.clone()
    pred[:, 40:48, 60:68] = 0.5  # error only inside the lamp
    g = glow_mask(gt, radius=4)
    m = all_metrics(pred, gt, g, with_lpips=False)
    assert m["psnr_nonglow"] > 60.0            # untouched outside
    assert m["psnr_glow"] < m["psnr"] < 100.0  # error concentrated in glow region
    assert 0.0 < m["glow_frac"] < 0.1
    assert math.isclose(psnr(gt, gt), 120.0, abs_tol=1e-3) or psnr(gt, gt) > 100


def test_ssim_map_and_masked_loss():
    a = torch.rand(3, 48, 48)
    assert abs(ssim(a, a) - 1.0) < 1e-5
    m = ssim_map(a, a)
    assert m.shape == (3, 48, 48)
    valid = torch.ones(1, 48, 48)
    valid[:, :24] = 0
    loss_full, l1_full = photometric_loss(a * 0.5, a, 0.2)
    loss_mask, l1_mask = photometric_loss(a * 0.5, a, 0.2, valid)
    assert torch.isfinite(loss_full) and torch.isfinite(loss_mask)
    # masked L1 equals the L1 over the valid half
    assert abs(l1_mask.item() - (a[:, 24:] * 0.5).abs().mean().item()) < 1e-5


def test_splits_every8_and_glowgs_roundtrip():
    cams = [SimpleNamespace(image_name=f"img_{i:03d}") for i in range(25)]
    tr, te = make_split(cams, "every8")
    assert len(te) == 4 and [c.image_name for c in te] == ["img_000", "img_008", "img_016", "img_024"]
    tr2, te2 = make_split(cams, "glowgs", n_train=6)
    assert len(tr2) == 6 and len(te2) == 19
    assert not set(c.image_name for c in tr2) & set(c.image_name for c in te2)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "split.json")
        save_split(p, tr2, te2, "glowgs")
        s = load_split(p)
        assert s["mode"] == "glowgs" and len(s["train"]) == 6 and len(s["test"]) == 19
