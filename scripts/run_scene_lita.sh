#!/usr/bin/env bash
# LITA-GS backbone: train + evaluate ours (halo) and plain LITA-GS on one LITA-layout scene.
#   bash scripts/run_scene_lita.sh data/lita/pondbike out_lita/pondbike [iterations] [extra args...]
# e.g. extra args: --no_depth --no_prior --split every8   (real night scenes without Marigold depth)
#                  --split lita --configs $LITA_ROOT/arguments/LOM/buu_low.py   (official LOM protocol)
set -euo pipefail
SCENE="$1"; OUT="$2"; ITERS="${3:-20000}"; shift 3 || shift $#
EXTRA=("$@")
for MODE in lita halo; do
  python train_lita_halo.py -s "$SCENE" -m "$OUT/$MODE" --mode $MODE --iterations "$ITERS" --quiet "${EXTRA[@]}"
  python eval_glow.py --renders "$OUT/$MODE/renders/iteration_$ITERS/test"       # bright pair (LITA protocol / GT)
  python eval_glow.py --renders "$OUT/$MODE/renders/iteration_$ITERS/test_low"   # composite vs observed dark photo
done
