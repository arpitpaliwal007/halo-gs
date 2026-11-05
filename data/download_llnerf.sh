#!/usr/bin/env bash
# LLNeRF (Wang et al., ICCV 2023) real low-light scenes, camera-finished sRGB,
# several with visible light sources.  Optional extra real data.
# Source (Google Drive folder from the official repo https://github.com/onpix/LLNeRF):
set -euo pipefail
ROOT="${1:-data/raw}"
mkdir -p "$ROOT/llnerf"
pip install -q gdown
gdown --folder "https://drive.google.com/drive/folders/1h-u8DkvuaIvcHZihYIWcqwpURiM32_u3" -O "$ROOT/llnerf" --remaining-ok
find "$ROOT/llnerf" -maxdepth 3 -type d | head -50
