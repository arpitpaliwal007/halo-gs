# Real glow scenes

All scenes are **real night captures with light sources**; none has glow-free
ground truth, so held-out views are scored against the *observed* image (the
composite render) and the scene/glow separation is judged on synthetic
controls and qualitatively. Train/test splits are written to `split.json`.

| Set | Scenes used | Images / scene | Poses | Source | Notes |
|---|---|---|---|---|---|
| Bilarf-night (GlowGS "Bilarf-Glow") | `building`, `pondbike`, `strat` (+ `chinesearch`, `nighttimepond`, `statue`, `lionpavilion`) | 20–93 | COLMAP, LLFF layout (`poses_bounds.npy`) | [HF Yuehao/bilarf_data](https://huggingface.co/datasets/Yuehao/bilarf_data), 7.8 GB zip | iPhone 13 / OnePlus 9; long exposure, **exposure/ISO varies per view** → consider `--per_view_exposure` |
| RawNeRF-Glow (sRGB) | `candlefiat`, `gardenlights`, `notchbush`, `parkstatue`, `stove`, `streetcorner` | ~100 | COLMAP `sparse/0` shipped | [raw.zip](http://storage.googleapis.com/gresearch/refraw360/raw.zip) | DNG + camera sRGB; we use sRGB like GlowGS |
| LLNeRF (optional) | any with lamps | 20–50 | run COLMAP | [GDrive](https://drive.google.com/drive/folders/1h-u8DkvuaIvcHZihYIWcqwpURiM32_u3) | low light, some glow |
| NightGlow (GlowGS) | 18 scenes | ~30 | — | **not released as of 2026-10** | add via `prepare_scene.py` when it appears |

Protocols:

* `--split every8` (default): standard LLFF hold-out, every 8th image by sorted name is test.
* `--split glowgs --n_train_views 6`: GlowGS few-shot protocol (6 training views, rest test).

Preparation:

```bash
bash data/download_bilarf.sh data/raw
for s in building pondbike strat; do
  python data/prepare_scene.py --scene data/raw/bilarf/$s --out data/scenes/bilarf/$s --max_side 1600
done
bash data/download_rawnerf.sh data/raw
for s in candlefiat gardenlights notchbush parkstatue stove streetcorner; do
  python data/prepare_scene.py --scene data/raw/rawnerf/scenes/$s --out data/scenes/rawnerf/$s
done
```

`prepare_scene.py` keeps shipped poses when present (COLMAP model, or LLFF
poses which are triangulated with `colmap point_triangulator`), and only runs
full SfM when there are none. Exact folder names inside the archives may
differ slightly from the table — the download scripts print what they found.
