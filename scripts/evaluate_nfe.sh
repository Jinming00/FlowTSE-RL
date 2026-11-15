#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 CHECKPOINT OUTPUT_ROOT [evaluate.py options]" >&2
    exit 2
fi
checkpoint="$1"
output_root="$2"
shift 2
for steps in 4 8 16 32; do
    python evaluate.py --checkpoint "$checkpoint" --output-dir "$output_root/nfe-$steps" --steps "$steps" "$@"
done
