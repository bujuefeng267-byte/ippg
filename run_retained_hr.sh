#!/usr/bin/env bash
set -euo pipefail
project_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "$project_dir/.venv/bin/python" -B "$project_dir/motion_positive_refinement_20260917/analyze_retained.py" "$@"
