#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$project_dir/.venv/bin/python" -B "$project_dir/motion_upgrade_v25_20260911/analyze_motion_v2.py" --pixel-mode tracking_screened --algorithm-mode guarded_fusion --max-gap 0.10 --motion-evidence reliable --hr-mode evidence --routing-mode evidence "$@"
