"""Framework-independent SFT computation, data protocol and full-utterance validation."""
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler, Subset

from .audio import BigVGANMel, ComplexISTFT, ComplexSTFT
from .config import resolve_warmup_steps
from .data import FrameBatchSampler, PrecomputedLibriMixDataset, collate_mixtures
from .flow import flow_matching_loss
from .inference import predict_chunked
from .metrics import audio_metrics, si_sdr
from .model import MelUDiT
from .quality import quality_size, save_quality_pair, score_quality, validation_size
from .representation import COMPLEX_STFT, feature_error_name, representation
from .vocoder import FrozenBigVGAN


class FlowSystem(torch.nn.Module):
    """State-dict names match the existing Lightning FlowModule (model.*)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = MelUDiT(**config['model'])
        if representation(config) == COMPLEX_STFT:
            self.frontend = ComplexSTFT(**config['audio'])
            self.vocoder = ComplexISTFT(**config['audio'])
        else:
            self.frontend = BigVGANMel(**config['audio'])
            self.vocoder = FrozenBigVGAN(**config['vocoder'])

    @property
    def device(self):
        return next(self.model.parameters()).device

    def features(self, batch):
        mixture, lengths = self.frontend(batch['mixture'], batch['mixture_lengths'])
        target, target_lengths = self.frontend(batch['target'], batch['target_lengths'])
        enrollment, enroll_lengths = self.frontend(batch['enrollment'], batch['enrollment_lengths'])
        if not torch.equal(lengths, target_lengths):
            raise ValueError('Target and mixture frame counts differ')
        return mixture, target, enrollment, lengths, enroll_lengths

    def forward(self, batch, *, noise=None, time=None):
        mixture, target, enrollment, lengths, enroll_lengths = self.features(batch)
        return flow_matching_loss(self.model, target, mixture, enrollment, lengths, enroll_lengths,
                                  noise=noise, time=time)


def seed_process(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    # Preserve Lightning 2.0.6's SeedSequence protocol for existing data loaders.
    # Adapted from Lightning Fabric utilities/seed.py (Apache-2.0).
    base_seed = torch.initial_seed() - worker_id
    sequence = np.random.SeedSequence([base_seed, worker_id, int(os.environ.get('RANK', 0))])
    np.random.seed(sequence.generate_state(4))
    torch_sequence, python_sequence = sequence.spawn(2)
    torch.manual_seed(int(torch_sequence.generate_state(1, dtype=np.uint64)[0]))
    words = python_sequence.generate_state(2, dtype=np.uint64).astype(object)
    random.seed(int(words[0] * (1 << 64) + words[1]))


class ExactEvalSampler(Sampler):
    def __init__(self, dataset, rank=0, world_size=1):
        self.indices = range(rank, len(dataset), world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def make_loaders(config, rank, world_size):
    data = dict(config['data'], seed=config['seed'])
    training = PrecomputedLibriMixDataset(data['train_dirs'], data['metadata_dir'], config['audio'], data, True)
    validation = PrecomputedLibriMixDataset([data['validation_dir']], data['metadata_dir'], config['audio'],
        dict(data, segment_seconds=data.get('validation_segment_seconds')), False)
    total = len(validation)
    validation = Subset(validation, range(validation_size(total, config['validation'])))
    sampler = FrameBatchSampler(training, data['max_frames_per_batch'], data['max_samples_per_batch'],
        config['audio']['hop_length'], True, config['seed'], rank, world_size,
        mixture_copies=2 if config['model'].get('conditioning_mode') == 'time_concat' else 1)
    common = dict(num_workers=data['num_workers'], pin_memory=True, collate_fn=collate_mixtures,
                  worker_init_fn=seed_worker)
    train_loader = DataLoader(training, batch_sampler=sampler, **common)
    val_loader = DataLoader(validation, batch_size=data['validation_batch_size'],
        sampler=ExactEvalSampler(validation, rank, world_size), **common)
    if rank == 0:
        print(f'Validation data prefix: {len(validation)}/{total} examples ({len(validation)/total:.1%})', flush=True)
    return train_loader, val_loader


def move_batch(batch, device):
    result = dict(batch)
    # Metadata retains IDs/paths and CPU reference tensors. It is never part of
    # the forward computation and must not be recursively copied a second time.
    for key in ('mixture', 'target', 'enrollment', 'mixture_lengths', 'target_lengths', 'enrollment_lengths'):
        result[key] = batch[key].to(device, non_blocking=True)
    return result


def estimated_updates(config, batches):
    total = math.ceil(batches / config['train']['accumulation_steps']) * config['train']['num_epochs']
    maximum = config['train']['max_steps']
    return min(total, maximum) if maximum > 0 else total


def make_optimizer_scheduler(system, config, total_steps):
    settings = config['optimizer']
    schedule = settings.get('schedule', 'cosine')
    if schedule not in ('cosine', 'constant'):
        raise ValueError(f'Unknown learning-rate schedule: {schedule}')
    optimizer = torch.optim.AdamW(system.model.parameters(), lr=settings['lr'], betas=settings['betas'],
        weight_decay=settings['weight_decay'], fused=settings.get('fused', None))
    warmup = resolve_warmup_steps(config, total_steps)
    epoch_warmup = 'warmup_epochs' in settings
    total = total_steps if epoch_warmup else max(warmup + 1, total_steps)

    def scale(step):
        if schedule == 'constant':
            return 1.0
        if step < warmup:
            return min((step + 1) / max(warmup, 1), 1.0) if epoch_warmup else max(step, 1) / max(warmup, 1)
        progress = min(1.0, (step - warmup) / max(total - warmup, 1))
        floor = settings.get('minimum_lr_ratio', .1)
        return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * progress))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def apply_resumed_lr_policy(optimizer, scheduler, config):
    """Override saved LR bases only for an explicitly requested constant stage."""
    if config['optimizer'].get('schedule', 'cosine') != 'constant':
        return
    rate = float(config['optimizer']['lr'])
    for group in optimizer.param_groups:
        group['lr'] = rate
        group['initial_lr'] = rate
    # LambdaLR.load_state_dict restores the old cosine base LR. Correct it too,
    # otherwise the first scheduler.step() silently raises the learning rate.
    scheduler.base_lrs = [rate] * len(optimizer.param_groups)
    scheduler._last_lr = [rate] * len(optimizer.param_groups)


@torch.no_grad()
def validate(system, loader, config, epoch, accelerator):
    """Same chunking, FM objective, coverage and external scorer as FlowModule."""
    system.eval()
    settings = config['validation']
    feature_error = feature_error_name(config)
    external = settings.get('run_external_metrics', False)
    selected = quality_size(len(loader.dataset), settings)
    directory = Path(config['checkpoint']['dir']) / 'quality_eval' / f'epoch_{epoch:04d}'
    totals, names = {}, []

    def collect(name, value):
        value = float(value)
        if math.isfinite(value):
            pair = totals.setdefault(name, [0.0, 0.0])
            pair[0] += value
            pair[1] += 1

    start = time.perf_counter()
    for batch in loader:
        batch = move_batch(batch, accelerator.device)
        with accelerator.autocast():
            for index, item in enumerate(batch['metadata']):
                samples = int(batch['mixture_lengths'][index])
                reference = batch['target'][index, :samples]
                enrollment = batch['enrollment'][index, :int(batch['enrollment_lengths'][index])]
                prediction = predict_chunked(system, batch['mixture'][index, :samples], enrollment,
                                             seed=config['seed'] + item['record_index'])
                target_features, target_lengths = system.frontend((reference * prediction.gain)[None])
                if int(target_lengths[0]) != prediction.generated_features.shape[-1]:
                    raise RuntimeError('Validation target and generated feature lengths differ')
                fm_sum, mel_sum, frames = 0.0, 0.0, 0
                for chunk in prediction.chunks:
                    clean = target_features[..., chunk.frame_start:chunk.frame_stop]
                    loss = flow_matching_loss(system.model, clean, chunk.mixture_features, chunk.enrollment_features,
                                              chunk.mixture_lengths, chunk.enrollment_lengths)
                    count = chunk.frame_stop - chunk.frame_start
                    fm_sum += float(loss) * count
                    mel_sum += float((chunk.generated_features - clean).abs().mean()) * count
                    frames += count
                waveform = prediction.waveform
                if waveform.numel() != reference.numel():
                    raise RuntimeError('Validation must retain the complete utterance')
                collect('fm_loss', fm_sum / frames)
                collect(feature_error, mel_sum / frames)
                collect('si_sdr', si_sdr(waveform, reference))
                if item['record_index'] < selected:
                    if external:
                        names.append(save_quality_pair(directory / 'audio', item, waveform, reference,
                                                       config['audio']['sample_rate']))
                    else:
                        for name, value in audio_metrics(waveform, reference,
                                sample_rate=config['audio']['sample_rate'],
                                names=settings.get('metrics', ['si_sdr', 'pesq', 'estoi'])).items():
                            if name != 'si_sdr':
                                collect(name, value)
    means, counts = {}, {}
    for name in ('fm_loss', feature_error, 'si_sdr', 'pesq', 'estoi'):
        pair = torch.tensor(totals.get(name, [0, 0]), device=accelerator.device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(pair)
        if pair[1] > 0:
            means[name], counts[name] = float(pair[0] / pair[1]), int(pair[1])
    scoring_seconds = 0.0
    if external:
        shards = [names]
        if dist.is_initialized():
            shards = [None] * dist.get_world_size()
            dist.all_gather_object(shards, names)
        result = None
        if accelerator.is_main_process:
            print(f'Validation epoch {epoch+1} | scoring {selected} full utterances on rank 0; other ranks wait', flush=True)
            try:
                if accelerator.device.type == 'cuda':
                    torch.cuda.empty_cache()
                scoring_start = time.perf_counter()
                result = score_quality(directory, settings, selected,
                    expected_filenames=[name for shard in shards for name in shard])
                scoring_seconds = time.perf_counter() - scoring_start
            except Exception as error:
                result = {'error': f'{type(error).__name__}: {error}'}
        if dist.is_initialized():
            payload = [result]
            dist.broadcast_object_list(payload, src=0)
            result = payload[0]
        if 'error' in result:
            raise RuntimeError('Validation quality scoring failed: ' + result['error'])
        means.update(result['metrics'])
        counts.update(result['finite_counts'])
    metrics = {f'val/{name}': value for name, value in means.items() if value is not None}
    metrics.update({f'diagnostic/val_{name}_count': float(count) for name, count in counts.items()})
    system.train()
    return metrics, time.perf_counter() - start, scoring_seconds
