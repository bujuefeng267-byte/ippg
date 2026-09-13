#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$PROJECT_DIR/.venv/bin/python" -B "$PROJECT_DIR/motion_upgrade_v27_20260912/analyze_video_v27.py" "$@"
