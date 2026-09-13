#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$project_dir/.venv/bin/python" "$project_dir"/motion_upgrade_v23_20260910/analyze_motion_v2.py --pixel-mode tracking_screened --max-gap 0.10 "$@"
