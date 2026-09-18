#!/usr/bin/env bash
set -euo pipefail
project_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec "$project_dir/.venv/bin/python" -B "$project_dir/realtime_rppg_v2_20260918/app.py" "$@"
