#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
exec "$ROOT/.venv/bin/python" -B "$ROOT/motion_upgrade_v26_20260911/analyze_components_v26.py" "$@"
