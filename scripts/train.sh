#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
num_gpus="${NUM_GPUS:-4}"
exec torchrun --standalone --nproc_per_node="$num_gpus" train.py --num-gpus "$num_gpus" "$@"
