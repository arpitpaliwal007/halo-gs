"""Locate a gaussian-splatting-style checkout and put it on sys.path.

Two backends are supported and they must NOT be on sys.path at the same time
(both define top-level ``scene``, ``gaussian_renderer``, ``arguments``, ``utils``):

* vanilla   : graphdeco-inria/gaussian-splatting (or the camenduru fork)
              -> ``--gs_root`` / ``$GS_ROOT`` / ../gaussian-splatting / ./gaussian-splatting
* lita      : LowLevelAI/LITA-GS (CVPR 2025, built on Gaussian-DK)
              -> ``--lita_root`` / ``$LITA_ROOT`` / ../LITA-GS / ./LITA-GS
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find(explicit, env, names, marker):
    cands = [explicit, os.environ.get(env)]
    for n in names:
        cands += [os.path.join(os.path.dirname(_HERE), n), os.path.join(_HERE, n)]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, *marker)):
            return os.path.abspath(c)
    return None


def find_gs_root(explicit: str | None = None) -> str:
    r = _find(explicit, "GS_ROOT", ["gaussian-splatting"], ("scene", "gaussian_model.py"))
    if r is None:
        raise FileNotFoundError(
            "gaussian-splatting not found. Pass --gs_root /path/to/gaussian-splatting or set $GS_ROOT "
            "(run setup_gs.sh to clone and build it).")
    return r


def find_lita_root(explicit: str | None = None) -> str:
    r = _find(explicit, "LITA_ROOT", ["LITA-GS", "lita-gs"], ("train_underexposed.py",))
    if r is None:
        raise FileNotFoundError(
            "LITA-GS not found. Pass --lita_root /path/to/LITA-GS or set $LITA_ROOT "
            "(run setup_lita.sh to clone and build it).")
    return r


def _check_clash():
    for m in ("scene", "gaussian_renderer", "arguments"):
        if m in sys.modules:
            raise RuntimeError(f"module '{m}' already imported from another backend; use one backend per process")


def add_gs_to_path(explicit: str | None = None) -> str:
    root = find_gs_root(explicit)
    _check_clash()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def add_lita_to_path(explicit: str | None = None) -> str:
    root = find_lita_root(explicit)
    _check_clash()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
