"""Train / test splits that are explicit and written to disk.

Every run writes ``split.json`` (camera image names per split) so evaluation
pairs renders with references by *name*, never by directory order.
"""
from __future__ import annotations

import json
import os
import re


def by_name(cams):
    return sorted(cams, key=lambda c: c.image_name)


def natural_key(name: str):
    """'img2' < 'img10' (numeric-aware), the order a notebook gets from numerically sorted files."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def by_natural_name(cams):
    return sorted(cams, key=lambda c: natural_key(c.image_name))


def _stem(n: str) -> str:
    return os.path.splitext(n)[0]


def make_split(all_cams, mode: str = "every8", n_train: int = 6, llffhold: int = 8,
               test_views=None, test_views_by: str = "index"):
    """Return (train_cams, test_cams).

    views   : explicit test views (``test_views``), given either as indices into the cameras
              sorted *numerically* by name (``test_views_by='index'``, e.g. LOM buu [1,17,24,9])
              or as image names with or without extension (``test_views_by='name'``).
              Everything else is training.
    every8  : standard 3DGS/LLFF protocol - every ``llffhold``-th image (by sorted
              name) is a test view.
    glowgs  : GlowGS protocol - ``n_train`` training views evenly spaced over the
              sorted sequence, everything else is test (few-shot).
    """
    cams = by_name(all_cams)
    n = len(cams)
    if mode == "views":
        if not test_views:
            raise ValueError("split 'views' needs --test_views")
        if test_views_by == "index":
            nat = by_natural_name(all_cams)
            idx = [int(v) for v in test_views]
            bad = [i for i in idx if not 0 <= i < n]
            if bad:
                raise ValueError(f"test view indices {bad} out of range for {n} cameras")
            chosen = [nat[i] for i in idx]
        else:
            want = {_stem(str(v)) for v in test_views}
            chosen = [c for c in cams if _stem(c.image_name) in want]
            missing = want - {_stem(c.image_name) for c in chosen}
            if missing:
                raise ValueError(f"test views not found: {sorted(missing)}")
        names = {c.image_name for c in chosen}
        test = chosen
        train = [c for c in cams if c.image_name not in names]
        return train, test
    if mode == "every8":
        test = [c for i, c in enumerate(cams) if i % llffhold == 0]
        train = [c for i, c in enumerate(cams) if i % llffhold != 0]
    elif mode == "glowgs":
        if n_train >= n:
            raise ValueError("n_train must be smaller than the number of images")
        idx = sorted({int(round(i * (n - 1) / (n_train - 1))) for i in range(n_train)})
        train = [cams[i] for i in idx]
        test = [c for i, c in enumerate(cams) if i not in idx]
    else:
        raise ValueError(f"unknown split mode {mode!r}")
    return train, test


def save_split(path: str, train, test, mode: str) -> None:
    with open(path, "w") as f:
        json.dump({"mode": mode,
                   "train": [c.image_name for c in train],
                   "test": [c.image_name for c in test]}, f, indent=2)


def load_split(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
