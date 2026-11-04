#!/usr/bin/env bash
# Train + render + evaluate the three methods on one prepared scene with identical settings.
#   bash scripts/run_scene.sh data/scenes/bilarf/pondbike out/pondbike [iterations] [extra train args...]
set -euo pipefail
SCENE="$1"; OUT="$2"; ITERS="${3:-30000}"; shift 3 || shift $#
EXTRA=("$@")
for MODE in vanilla masked halo; do
  python train_glow.py -s "$SCENE" -m "$OUT/$MODE" --mode $MODE --iterations "$ITERS" \
      --save_iterations 7000 "$ITERS" --quiet "${EXTRA[@]}"
  python render_glow.py -m "$OUT/$MODE" --iteration "$ITERS" --skip_train
  python eval_glow.py -m "$OUT/$MODE" --iteration "$ITERS"
done
python aggregate_results.py "$(dirname "$OUT")" --iteration "$ITERS" --md "$OUT/results.md"
