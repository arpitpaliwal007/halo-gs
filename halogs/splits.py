"""Train / test splits that are explicit and written to disk.

Every run writes ``split.json`` (camera image names per split) so evaluation
pairs renders with references by *name*, never by directory order.
"""
from __future__ import annotations

import json


def by_name(cams):
    return sorted(cams, key=lambda c: c.image_name)


def make_split(all_cams, mode: str = "every8", n_train: int = 6, llffhold: int = 8):
    """Return (train_cams, test_cams).

    every8  : standard 3DGS/LLFF protocol - every ``llffhold``-th image (by sorted
              name) is a test view.
    glowgs  : GlowGS protocol - ``n_train`` training views evenly spaced over the
              sorted sequence, everything else is test (few-shot).
    """
    cams = by_name(all_cams)
    n = len(cams)
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
