#!/usr/bin/env bash
# Clone and build the official LITA-GS (CVPR 2025) backbone next to this repo.
#   bash setup_lita.sh [target_dir]
# Builds its Gaussian-DK rasteriser (NUM_CHANNELS=11) and simple-knn. CUDA required.
# Tested by the authors on RTX 5090 / CUDA 12.8 / PyTorch 2.8; on Colab (T4, CUDA 12.x) the
# build is expected to work but is unverified - run tests/test_lita_smoke_gpu.py after this.
set -euo pipefail
TARGET="${1:-$(dirname "$(cd "$(dirname "$0")" && pwd)")/LITA-GS}"
if [ ! -d "$TARGET" ]; then
  git clone --recursive https://github.com/LowLevelAI/LITA-GS "$TARGET"
fi
cd "$TARGET"
git submodule update --init --recursive
grep -q "cfloat" submodules/simple-knn/simple_knn.cu || sed -i '1i #include <cfloat>' submodules/simple-knn/simple_knn.cu
pip install -q trimesh tqdm scipy scikit-image plyfile einops opencv-python-headless open3d pandas  # dataset_readers imports open3d + pandas
pip install -q ./submodules/diff-gaussian-rasterization
pip install -q ./submodules/simple-knn
python - <<'EOF'
import diff_gaussian_rasterization, simple_knn
print("LITA-GS rasteriser + simple-knn built")
EOF
echo "export LITA_ROOT=$TARGET"
echo "NOTE: the Marigold depth checkpoint is not in the repo; depth priors need 'Depth Extraction/checkpoint/marigold-depth-lcm-v1-0'"
echo "      (prs-eth/marigold-depth-lcm-v1-0 on Hugging Face). Without it, train with --no_depth."
