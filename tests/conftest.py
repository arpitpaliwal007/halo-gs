"""Test setup: make `halogs` importable and, when a gaussian-splatting checkout is
available ($GS_ROOT or ../gaussian-splatting), stub its CUDA extensions so the
pure-python parts (GaussianModel lifecycle) can be exercised on CPU."""
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _stub_cuda_modules():
    if "simple_knn" not in sys.modules:
        sk = types.ModuleType("simple_knn")
        skc = types.ModuleType("simple_knn._C")
        skc.distCUDA2 = lambda x: (x * 0 + 1e-3).sum(-1)
        sk._C = skc
        sys.modules["simple_knn"] = sk
        sys.modules["simple_knn._C"] = skc
    if "diff_gaussian_rasterization" not in sys.modules:
        dgr = types.ModuleType("diff_gaussian_rasterization")
        dgr.GaussianRasterizationSettings = object
        dgr.GaussianRasterizer = object
        sys.modules["diff_gaussian_rasterization"] = dgr


@pytest.fixture(scope="session")
def gs_root():
    from halogs.gs_path import find_gs_root
    try:
        root = find_gs_root(os.environ.get("GS_ROOT"))
    except FileNotFoundError:
        pytest.skip("gaussian-splatting checkout not found (set GS_ROOT)")
    _stub_cuda_modules()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
