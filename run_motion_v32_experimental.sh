#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$PROJECT_DIR/.venv/bin/python" -B "$PROJECT_DIR/motion_upgrade_v32_20260914/analyze_v28_result_v32.py" "$@"
