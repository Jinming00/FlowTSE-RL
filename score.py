"""Score existing prediction/reference WAVs independently of model inference."""
import os

# Scope these defaults to this scoring process and its CPU workers.
for variable in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[variable] = os.environ.get('SCORING_CPU_THREADS', '1')

from flowtse_rl.scoring.cli import main

if __name__ == '__main__':
    main()
