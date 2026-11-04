#!/usr/bin/env bash
# Kernel / prior ablations on one scene (same split, schedule, init).
#   bash scripts/ablations.sh data/scenes/bilarf/pondbike out_abl/pondbike [iterations]
set -euo pipefail
SCENE="$1"; OUT="$2"; ITERS="${3:-30000}"
run () { NAME=$1; shift
  python train_glow.py -s "$SCENE" -m "$OUT/$NAME" --mode halo --iterations "$ITERS" --save_iterations "$ITERS" --quiet "$@"
  python render_glow.py -m "$OUT/$NAME" --iteration "$ITERS" --skip_train
  python eval_glow.py -m "$OUT/$NAME" --iteration "$ITERS"
}
run halo_both      --kernel both
run halo_apsf      --kernel apsf
run halo_lens      --kernel lens
run halo_radial    --kernel radial
run halo_frozen    --kernel both --freeze_kernel
run halo_nosparse  --kernel both --lambda_emission 0
run halo_exposure  --kernel both --per_view_exposure
python aggregate_results.py "$(dirname "$OUT")" --iteration "$ITERS" --md "$OUT/ablations.md"
