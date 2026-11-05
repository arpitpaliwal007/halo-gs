#!/usr/bin/env bash
# RawNeRF (Mildenhall et al., CVPR 2022) night scenes with light sources.
# GlowGS's "RawNeRF-Glow" uses the sRGB versions of:
#   candlefiat gardenlights notchbush parkstatue stove streetcorner
# Source: http://storage.googleapis.com/gresearch/refraw360/raw.zip  (one archive, all scenes, large)
# Each scene ships raw DNGs (+ EXIF json), camera sRGB images and a COLMAP sparse/ model.
set -euo pipefail
ROOT="${1:-data/raw}"
mkdir -p "$ROOT"
cd "$ROOT"
[ -f raw.zip ] || wget -c "http://storage.googleapis.com/gresearch/refraw360/raw.zip"
if [ ! -d rawnerf ]; then
  mkdir -p rawnerf
  # extract only the glow scenes to save disk (drop the pattern to get everything)
  unzip -q raw.zip -d rawnerf '*candlefiat*' '*gardenlights*' '*notchbush*' '*parkstatue*' '*stove*' '*streetcorner*' || unzip -q raw.zip -d rawnerf
fi
echo "scenes:"; find rawnerf -maxdepth 3 -type d \( -name images -o -name raw \) -printf '%h\n' | sort -u
