#!/usr/bin/env bash
# Clone and build gaussian-splatting next to this repo (CUDA required for the rasterizer).
#   bash setup_gs.sh [target_dir] [fork]
# fork: "official" (graphdeco-inria, current API) or "camenduru" (2023 API used in the thesis notebooks).
set -euo pipefail
TARGET="${1:-$(dirname "$(cd "$(dirname "$0")" && pwd)")/gaussian-splatting}"
FORK="${2:-official}"
case "$FORK" in
  official)  URL=https://github.com/graphdeco-inria/gaussian-splatting ;;
  camenduru) URL=https://github.com/camenduru/gaussian-splatting ;;
  *) echo "unknown fork $FORK"; exit 1 ;;
esac
if [ ! -d "$TARGET" ]; then
  git clone --recursive "$URL" "$TARGET"
fi
cd "$TARGET"
# simple-knn needs <cfloat> on recent compilers
grep -q "cfloat" submodules/simple-knn/simple_knn.cu || sed -i '1i #include <cfloat>' submodules/simple-knn/simple_knn.cu
pip install -q plyfile tqdm lpips scikit-image
pip install -q ./submodules/diff-gaussian-rasterization
pip install -q ./submodules/simple-knn
python -c "import simple_knn, diff_gaussian_rasterization; print('gaussian-splatting ready at $TARGET')"
echo "export GS_ROOT=$TARGET"
