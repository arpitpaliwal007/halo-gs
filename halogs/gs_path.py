"""Locate the gaussian-splatting checkout and put it on sys.path.

Resolution order: ``--gs_root`` CLI value, ``$GS_ROOT``, ``../gaussian-splatting``
next to this repo, ``./gaussian-splatting`` inside it.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_gs_root(explicit: str | None = None) -> str:
    cands = [explicit, os.environ.get("GS_ROOT"),
             os.path.join(os.path.dirname(_HERE), "gaussian-splatting"),
             os.path.join(_HERE, "gaussian-splatting")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "scene", "gaussian_model.py")):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "gaussian-splatting not found. Pass --gs_root /path/to/gaussian-splatting or set $GS_ROOT "
        "(run setup_gs.sh to clone and build it).")


def add_gs_to_path(explicit: str | None = None) -> str:
    root = find_gs_root(explicit)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
