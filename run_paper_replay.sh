#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_SCRIPT="/mnt/c/Users/15011/Documents/ChatGPT/New project/replay_paper_methods.py"
ARCHIVED_SCRIPT="$PROJECT_DIR/research_archive_20260913/replay_paper_methods.py"
REPLAY_SCRIPT="${RPPG_PAPER_REPLAY_SCRIPT:-$ORIGINAL_SCRIPT}"
if [[ ! -f "$REPLAY_SCRIPT" && -z "${RPPG_PAPER_REPLAY_SCRIPT:-}" ]]; then
  REPLAY_SCRIPT="$ARCHIVED_SCRIPT"
fi
exec "$PROJECT_DIR/.venv/bin/python" -B "$REPLAY_SCRIPT" "$@"
