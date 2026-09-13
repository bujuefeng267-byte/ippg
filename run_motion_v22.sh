#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$project_dir/.venv/bin/python" "$project_dir/motion_upgrade_v22_20260909/analyze_motion_v2.py" "$@"
