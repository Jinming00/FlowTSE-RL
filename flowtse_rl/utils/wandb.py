"""Optional W&B logging with a persistent run ID and fractional epoch axis."""
import os
from pathlib import Path
import tempfile
import uuid


def save_run_id(output_dir, run_id):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(run_id + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / 'wandb_run_id.txt')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def resume_options(config):
    output = Path(config['checkpoint']['dir'])
    identity = output / 'wandb_run_id.txt'
    run_id = os.environ.get('WANDB_RUN_ID')
    if not run_id and identity.is_file():
        run_id = identity.read_text().strip()
    resume = config['checkpoint'].get('resume')
    if not run_id and resume:
        import torch
        payload = torch.load(resume, map_location='cpu', mmap=True, weights_only=False)
        run_id = payload.get('meanflow_wandb', {}).get('id')
    existing = bool(run_id)
    run_id = run_id or uuid.uuid4().hex[:16]
    save_run_id(output, run_id)
    return {'id': run_id, 'resume': 'allow' if existing else None}


def define_epoch_axis(run):
    run.define_metric('epoch')
    run.define_metric('*', step_metric='epoch', step_sync=True)
    return run


def create_native_run(config):
    import wandb
    mode = os.environ.get('WANDB_MODE', 'online')
    options = resume_options(config)
    if mode in {'offline', 'disabled'}:
        options['resume'] = None
    settings = config['train']
    run = wandb.init(project=settings.get('wandb_project', 'FlowTSE-RL'),
                     name=settings.get('wandb_name', 'baseline'),
                     dir=str(Path(config['checkpoint']['dir']).resolve()),
                     config=config, mode=mode, **options)
    return define_epoch_axis(run)
