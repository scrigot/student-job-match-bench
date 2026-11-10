#!/bin/sh
# Overnight run: keeps the Mac awake, grades a fresh batch, writes golden/batches/<date>-auto/.
# Usage: scripts/nightly.sh [bench-nightly options], e.g. scripts/nightly.sh --n 200 --max-spend 8
set -eu
cd "$(dirname "$0")/.."
mkdir -p results/nightly
log="results/nightly/$(date +%Y-%m-%d_%H%M).log"
echo "logging to $log"
caffeinate -i uv run bench-nightly "$@" 2>&1 | tee "$log"
