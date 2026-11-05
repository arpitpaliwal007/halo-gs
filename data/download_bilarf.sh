#!/usr/bin/env bash
# Bilarf nighttime captures (Wang et al., SIGGRAPH 2024) - 7 real night scenes
# with street lamps / building lights, COLMAP-posed, LLFF layout.
# GlowGS evaluates on three of them: building, pondbike, strat.
# Source: https://huggingface.co/datasets/Yuehao/bilarf_data  (bilarf_data.zip, ~7.8 GB, Apache-2.0)
set -euo pipefail
ROOT="${1:-data/raw}"
mkdir -p "$ROOT"
cd "$ROOT"
if [ ! -f bilarf_data.zip ]; then
  # huggingface_hub handles resume + LFS; falls back to wget
  python - <<'EOF' || wget -c "https://huggingface.co/datasets/Yuehao/bilarf_data/resolve/main/bilarf_data.zip"
from huggingface_hub import hf_hub_download
hf_hub_download("Yuehao/bilarf_data", "bilarf_data.zip", repo_type="dataset", local_dir=".")
EOF
fi
[ -d bilarf ] || { mkdir -p bilarf && unzip -q bilarf_data.zip -d bilarf; }
echo "scenes:"; find bilarf -maxdepth 3 -name poses_bounds.npy -printf '%h\n'
