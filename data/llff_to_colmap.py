#!/usr/bin/env python
"""Turn an LLFF scene (images/ + poses_bounds.npy) into a 3DGS-ready COLMAP model
WITHOUT re-estimating poses: known poses are written as a COLMAP text model and
COLMAP only triangulates points for the Gaussian initialisation.

  python data/llff_to_colmap.py --scene data/raw/bilarf/pondbike --out data/scenes/bilarf/pondbike

Requires the `colmap` binary (apt-get install colmap on Colab).
Output layout:  <out>/images/*.jpg  <out>/sparse/0/{cameras,images,points3D}.bin
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess

import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([[Rxx - Ryy - Rzz, 0, 0, 0],
                  [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                  [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                  [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def llff_poses(poses_bounds):
    """poses_bounds (N,17) -> list of (c2w 3x4 in OpenCV convention, (h,w,f))."""
    arr = poses_bounds[:, :15].reshape(-1, 3, 5)
    out = []
    for p in arr:
        hwf = p[:, 4]
        m = p[:, :4]  # columns: down, right, back, t   (LLFF convention)
        down, right, back, t = m[:, 0], m[:, 1], m[:, 2], m[:, 3]
        c2w = np.stack([right, down, -back, t], 1)  # OpenCV: right, down, forward
        out.append((c2w, hwf))
    return out


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, help="LLFF scene dir with images/ and poses_bounds.npy")
    p.add_argument("--out", required=True)
    p.add_argument("--images_dir", default="images", help="subfolder with the images poses_bounds refers to")
    p.add_argument("--colmap", default="colmap")
    p.add_argument("--no_gpu", action="store_true")
    args = p.parse_args()

    src_imgs = os.path.join(args.scene, args.images_dir)
    names = sorted(f for f in os.listdir(src_imgs) if f.endswith(IMG_EXT))
    pb = np.load(os.path.join(args.scene, "poses_bounds.npy"))
    assert len(names) == pb.shape[0], f"{len(names)} images vs {pb.shape[0]} poses"
    poses = llff_poses(pb)

    out_imgs = os.path.join(args.out, "images")
    os.makedirs(out_imgs, exist_ok=True)
    for n in names:
        dst = os.path.join(out_imgs, n)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(src_imgs, n), dst)

    h, w, f = poses[0][1]
    h, w = int(round(h)), int(round(w))
    # sanity: the images on disk must match the hwf resolution
    from PIL import Image
    W0, H0 = Image.open(os.path.join(out_imgs, names[0])).size
    if (H0, W0) != (h, w):
        s = W0 / w
        print(f"[llff_to_colmap] images are {W0}x{H0} but hwf says {w}x{h}; scaling focal by {s:.4f}")
        f, h, w = f * s, H0, W0

    db = os.path.join(args.out, "database.db")
    if os.path.exists(db):
        os.remove(db)
    cam_params = f"{f},{f},{w / 2.0},{h / 2.0}"
    run([args.colmap, "feature_extractor", "--database_path", db, "--image_path", out_imgs,
         "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera", "1",
         "--ImageReader.camera_params", cam_params, "--SiftExtraction.use_gpu", "0" if args.no_gpu else "1"])
    run([args.colmap, "exhaustive_matcher", "--database_path", db, "--SiftMatching.use_gpu", "0" if args.no_gpu else "1"])

    # image ids as COLMAP assigned them in the database
    con = sqlite3.connect(db)
    db_ids = {name: iid for iid, name in con.execute("SELECT image_id, name FROM images")}
    con.close()
    missing = [n for n in names if n not in db_ids]
    assert not missing, f"images missing from database: {missing[:5]}"

    known = os.path.join(args.out, "sparse_known")
    os.makedirs(known, exist_ok=True)
    with open(os.path.join(known, "cameras.txt"), "w") as fc:
        fc.write(f"1 PINHOLE {w} {h} {f} {f} {w / 2.0} {h / 2.0}\n")
    with open(os.path.join(known, "images.txt"), "w") as fi:
        for n, (c2w, _) in zip(names, poses):
            R = c2w[:, :3]
            t = c2w[:, 3]
            Rw2c = R.T
            tw2c = -Rw2c @ t
            q = rotmat2qvec(Rw2c)
            fi.write(f"{db_ids[n]} {q[0]} {q[1]} {q[2]} {q[3]} {tw2c[0]} {tw2c[1]} {tw2c[2]} 1 {n}\n\n")
    open(os.path.join(known, "points3D.txt"), "w").close()

    sparse0 = os.path.join(args.out, "sparse", "0")
    os.makedirs(sparse0, exist_ok=True)
    run([args.colmap, "point_triangulator", "--database_path", db, "--image_path", out_imgs,
         "--input_path", known, "--output_path", sparse0])
    print("[llff_to_colmap] done ->", args.out)


if __name__ == "__main__":
    main()
