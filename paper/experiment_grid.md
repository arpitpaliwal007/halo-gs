# Experiment grid and budget (T4, 1600 px, 30k iterations ≈ 25–35 min / run)

| Block | Runs | Purpose |
|---|---|---|
| A. Real, 3 methods × 9 scenes × 1 seed | 27 | main table |
| B. Real, 3 methods × 3 Bilarf scenes × 2 extra seeds | 18 | variance on headline rows |
| C. GlowGS protocol (6 views), halo + vanilla × 3 Bilarf scenes | 6 | comparison table vs. GlowGS numbers |
| D. Synthetic control, 3 methods × 3 T × 2 intensity | 18 | separation / parameter recovery |
| E. Ablations on pondbike (7 configs) | 7 | kernel family, sparsity, exposure |
| **Total** | **76** | ≈ 35–45 GPU-hours |

Order of execution: D (cheap, decides whether the method works) → A on Bilarf →
E → A on RawNeRF → B → C.

Commands:

```bash
# D
python data/make_synthetic_glow.py --scene data/scenes/lom/buu_bright --out data/scenes/synth/buu_T1.5_i3 --T 1.5 --intensity 3
bash scripts/run_scene.sh data/scenes/synth/buu_T1.5_i3 out/buu_T1.5_i3 30000 --data_device cpu
python eval_glow.py -m out/buu_T1.5_i3/halo --clean_gt_dir data/scenes/synth/buu_T1.5_i3/gt_clean
# A
for s in pondbike building strat; do bash scripts/run_scene.sh data/scenes/bilarf/$s out/$s 30000 --data_device cpu --per_view_exposure; done
# E
bash scripts/ablations.sh data/scenes/bilarf/pondbike out_abl/pondbike 30000
# B
for seed in 1 2; do python train_glow.py -s data/scenes/bilarf/pondbike -m out_seed$seed/pondbike/halo --mode halo --seed $seed ...; done
python aggregate_results.py out --iteration 30000 --md results.md --csv results.csv
```
