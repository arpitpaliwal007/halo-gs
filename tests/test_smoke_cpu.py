"""End-to-end pipeline smoke test on CPU (mock rasterizer). Needs a gaussian-splatting checkout."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.filterwarnings("ignore")
def test_pipeline_smoke(gs_root):
    env = dict(os.environ, GS_ROOT=gs_root)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "smoke_cpu.py"), "--iters", "40"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=1500)
    print(r.stdout[-4000:])
    print(r.stderr[-4000:])
    assert r.returncode == 0, "smoke pipeline failed"
    assert "[smoke] OK" in r.stdout
