#!/usr/bin/env bash
# The two backends define the same top-level modules (scene, gaussian_renderer, arguments, utils),
# so their tests must run in separate interpreter processes.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q tests --ignore=tests/test_lita_lifecycle.py --ignore=tests/test_lifecycle.py --ignore=tests/test_smoke_cpu.py
[ -n "${GS_ROOT:-}" ]   && python -m pytest -q tests/test_lifecycle.py tests/test_smoke_cpu.py
[ -n "${LITA_ROOT:-}" ] && python -m pytest -q tests/test_lita_lifecycle.py
echo "all test groups passed"
