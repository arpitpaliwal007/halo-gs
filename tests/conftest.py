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


def _stub_optional_modules():
    """LITA-GS's dataset_readers imports open3d at module level; stub it when absent."""
    for name in ("open3d",):
        if name not in sys.modules:
            try:
                __import__(name)
            except Exception:
                sys.modules[name] = types.ModuleType(name)


@pytest.fixture(scope="session")
def lita_root():
    from halogs.gs_path import find_lita_root
    try:
        root = find_lita_root(os.environ.get("LITA_ROOT"))
    except FileNotFoundError:
        pytest.skip("LITA-GS checkout not found (set LITA_ROOT)")
    for m in ("scene", "gaussian_renderer", "arguments", "utils"):
        if m in sys.modules:
            pytest.skip("another gaussian-splatting backend is already imported in this process")
    _stub_cuda_modules()
    _stub_optional_modules()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


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
