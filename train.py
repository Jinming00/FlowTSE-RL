"""Train locally or with torchrun; checkpoints stay in the output directory."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch

from flowtse_rl.config import BASELINE_CONFIG, load_config, validate_config
from flowtse_rl.training import create_accelerator, run_training
from flowtse_rl.paths import check_data_paths


def prepare_config(args):
    config = load_config(args.config, data_root=args.data_root)
    if args.output_dir:
        config['checkpoint']['dir'] = str(args.output_dir)
    if args.num_gpus is not None:
        config['ddp']['num_gpus'] = args.num_gpus
    if args.resume:
        config['checkpoint']['resume'] = str(args.resume)
    elif config['checkpoint'].get('auto_resume', True) and not config['checkpoint'].get('resume'):
        last = Path(config['checkpoint']['dir']) / 'checkpoints' / 'last.ckpt'
        if last.is_file():
            config['checkpoint']['resume'] = str(last)
    validate_config(config)
    check_data_paths(config, 'train')
    for metric, dependency in [('pesq', 'pesq'), ('estoi', 'pystoi')]:
        if (not config['validation'].get('run_external_metrics', False)
                and metric in config['validation'].get('metrics', ['si_sdr', 'pesq', 'estoi'])
                and importlib.util.find_spec(dependency) is None):
            raise RuntimeError(f'Missing {dependency}: install the evaluation extra with pip install -e ".[eval]"')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASELINE_CONFIG)
    parser.add_argument('--data-root', type=Path, help='Override dataset root; FLOWTSE_DATA_ROOT is also supported')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', type=Path, help='Restore model, optimizer, scheduler and RNG state')
    parser.add_argument('--num-gpus', type=int, help='Override GPUs per node; match torchrun process count')
    parser.add_argument('--cpu', action='store_true', help='Small CPU checks; use precision: 32-true')
    args = parser.parse_args()
    if args.num_gpus is not None and args.num_gpus < 1:
        parser.error('--num-gpus must be positive')
    config = prepare_config(args)
    root = Path(config['checkpoint']['dir'])
    accelerator = create_accelerator(config, cpu=args.cpu)
    try:
        outcome = run_training(config, accelerator=accelerator)
        if accelerator.is_main_process:
            status = 'stopped' if outcome['stopped'] else 'complete'
            (root / 'training_status.json').write_text(json.dumps(dict(outcome, status=status), indent=2) + '\n')
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
