#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then
  echo "Usage: ./run.sh videos/example.mp4 [pos|chrom]" >&2
  exit 2
fi

video="$1"
method="${2:-pos}"
name="$(basename "${video%.*}")"
exec "$project_dir/.venv/bin/python" "$project_dir/analyze_rppg.py" \
  "$video" --method "$method" --output "$project_dir/results/$name/$method"
