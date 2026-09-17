#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$PROJECT_DIR/.venv/bin/python" -B "$PROJECT_DIR/motion_pos_ablation_20260916/replay_pos_experimental.py" "$@"
