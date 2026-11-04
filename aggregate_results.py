#!/usr/bin/env python
"""Collect metrics.json files from out/<scene>/<method>/renders/iteration_N/test into one table.

  python aggregate_results.py out --iteration 30000 --md results.md --csv results.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

COLS = ["psnr", "ssim", "lpips", "psnr_glow", "ssim_glow", "lpips_glowcrop", "psnr_nonglow", "ssim_nonglow", "glow_frac"]

def _nanmean(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    return float(np.mean(xs)) if xs else float("nan")



def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--md", type=str, default=None)
    p.add_argument("--csv", type=str, default=None)
    args = p.parse_args()

    rows = []
    for mpath in sorted(glob.glob(os.path.join(args.root, "*", "*", "renders", "iteration_*", "test", "metrics.json"))):
        parts = mpath.split(os.sep)
        it = int(parts[-3].split("_")[1])
        if args.iteration > 0 and it != args.iteration:
            continue
        scene, method = parts[-6], parts[-5]
        with open(mpath) as f:
            mean = json.load(f)["mean"]
        rows.append({"scene": scene, "method": method, "iteration": it, **{c: mean.get(c, float("nan")) for c in COLS}})

    if not rows:
        print("no metrics found under", args.root)
        return

    # per-method average across scenes (scene-level first, as it should be)
    methods = sorted({r["method"] for r in rows})
    avg = []
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        avg.append({"scene": "MEAN", "method": m, "iteration": rs[0]["iteration"],
                    **{c: float(_nanmean([r[c] for r in rs])) for c in COLS}})

    def fmt(r):
        return [r["scene"], r["method"], str(r["iteration"])] + [f"{r[c]:.3f}" if not np.isnan(r[c]) else "-" for c in COLS]

    header = ["scene", "method", "iter"] + COLS
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows + avg:
        lines.append("| " + " | ".join(fmt(r)) + " |")
    table = "\n".join(lines)
    print(table)
    if args.md:
        with open(args.md, "w") as f:
            f.write(table + "\n")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header[:2] + ["iteration"] + COLS)
            w.writeheader()
            for r in rows + avg:
                w.writerow(r)


if __name__ == "__main__":
    main()
