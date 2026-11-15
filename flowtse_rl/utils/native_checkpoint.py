"""Atomic local native checkpoints; import old Lightning payloads."""
import math
import os
import tempfile
from pathlib import Path

import torch


def available(path):
    path = Path(path)
    return path.is_file()


class NativeCheckpointIO:
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        if storage_options is not None or '://' in str(path):
            raise ValueError('Native checkpoints require a local filesystem')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        identity = path.parent.parent / 'wandb_run_id.txt'
        if identity.is_file():
            checkpoint = dict(checkpoint, meanflow_wandb={'id': identity.read_text().strip()})
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(checkpoint, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(0o644)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load_checkpoint(self, path, map_location='cpu'):
        return torch.load(path, map_location=map_location, mmap=True, weights_only=False)

    def remove_checkpoint(self, path):
        Path(path).unlink(missing_ok=True)



def resume_position(checkpoint, batches_per_epoch, accumulation_steps):
    state = checkpoint.get('flowtse_trainer')
    if state is not None:
        if state.get('format_version') != 1:
            raise ValueError('Unknown native trainer checkpoint version')
        epoch, batch = int(state['next_epoch']), int(state['next_batch'])
    else:
        # Lightning's processed-epoch counter can advance at a max_steps stop
        # inside an epoch. Inspect batch progress before deciding to skip it.
        epoch = int(checkpoint['epoch'])
        loop = checkpoint.get('loops', {}).get('fit_loop', {})
        epochs = loop.get('epoch_progress', {}).get('current', {})
        # Trainer.save_checkpoint() after fit has already advanced current_epoch;
        # the batch counters still describe the preceding epoch.
        if epoch > 0 and epochs.get('started') == epoch and epochs.get('completed') == epoch:
            epoch -= 1
        progress = loop.get('epoch_loop.batch_progress', {}).get('current', {})
        if not progress:
            progress = loop.get('epoch_loop', {}).get('batch_progress', {}).get('current', {})
        if not progress:
            raise ValueError('Old checkpoint has no batch progress; refusing an ambiguous full resume')
        batch = max(int(progress.get('completed', 0)), int(progress.get('processed', 0)))
        if batch >= batches_per_epoch:
            epoch, batch = epoch + 1, 0
    if not 0 <= batch < batches_per_epoch:
        raise ValueError('Resume batch lies outside the current epoch; workload changed')
    if batch % accumulation_steps:
        raise ValueError('Checkpoint is inside an unfinished accumulation group; gradient state is unavailable')
    return epoch, batch


class CheckpointManager:
    def __init__(self, config, io=None):
        self.config = config
        self.settings = config['checkpoint']
        self.directory = Path(self.settings['dir']) / 'checkpoints'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.io = io or NativeCheckpointIO()
        self.ranked = {}

    def restore(self, checkpoint):
        for callback in checkpoint.get('callbacks', {}).values():
            if isinstance(callback, dict) and callback.get('best_k_models'):
                self.ranked = {str(self.directory / Path(path).name): float(score)
                               for path, score in callback['best_k_models'].items()
                               if available(self.directory / Path(path).name)}
                break
        # Keep the bytes of a legacy single-best checkpoint before introducing
        # aliases. New FlowTSE snapshots already use immutable candidate names.
        legacy = self.directory / 'best.ckpt'
        if str(legacy) in self.ranked:
            replacement = self.directory / f'best-resumed-{int(checkpoint["global_step"]):012d}.ckpt'
            if legacy.is_file() and not legacy.is_symlink():
                os.replace(legacy, replacement)
            elif not replacement.is_file():
                raise FileNotFoundError('Cannot preserve the previous single-best checkpoint')
            self.ranked[str(replacement)] = self.ranked.pop(str(legacy))
        self.sync_aliases()

    def state_dict(self):
        return {'best_k_models': dict(self.ranked), 'monitor': self.settings.get('monitor', 'val/pesq'),
                'mode': self.settings.get('mode', 'max')}

    def sync_aliases(self):
        ranked = sorted(self.ranked, key=self.ranked.get, reverse=self.settings.get('mode', 'max') == 'max')
        for name, index in [('best.ckpt', 0), ('best1.ckpt', 0), ('best2.ckpt', 1), ('best3.ckpt', 2)]:
            alias = self.directory / name
            if index >= len(ranked):
                if alias.is_symlink():
                    alias.unlink()
                continue
            if alias.is_file() and not alias.is_symlink():
                raise RuntimeError('Refusing to replace a regular legacy checkpoint with an alias')
            temporary = self.directory / f'.{name}.{os.getpid()}.tmp'
            try:
                temporary.unlink(missing_ok=True)
                temporary.symlink_to(Path(ranked[index]).name)
                os.replace(temporary, alias)
            finally:
                temporary.unlink(missing_ok=True)

    def save(self, build, epoch, step, metrics, *, last):
        """All writers call this only on rank zero at optimizer boundaries."""
        monitor = self.settings.get('monitor', 'val/pesq')
        score = metrics.get(monitor)
        top_k = self.settings.get('save_top_k', 3)
        removed, candidate = [], None
        if score is not None and math.isfinite(float(score)) and top_k:
            candidate = self.directory / f'best-{epoch:04d}-{float(score):.6f}-s{step:09d}.ckpt'
            ranked = dict(self.ranked, **{str(candidate): float(score)})
            paths = sorted(ranked, key=ranked.get, reverse=self.settings.get('mode', 'max') == 'max')
            keep = paths if top_k < 0 else paths[:top_k]
            if str(candidate) in keep:
                removed = [path for path in self.ranked if path not in keep]
                self.ranked = {path: ranked[path] for path in keep}
            else:
                candidate = None
        if candidate is not None:
            self.io.save_checkpoint(build(self.state_dict()), candidate)
        if last:
            self.io.save_checkpoint(build(self.state_dict()), self.directory / 'last.ckpt')
        self.sync_aliases()
        for path in removed:
            self.io.remove_checkpoint(path)
