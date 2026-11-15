"""Explicit PyTorch SFT loop with Accelerate providing BF16 and DDP only."""
import contextlib
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from .engine import (FlowSystem, apply_resumed_lr_policy, estimated_updates, make_loaders, make_optimizer_scheduler,
                     move_batch, seed_process, validate)
from .utils.native_checkpoint import CheckpointManager, NativeCheckpointIO, resume_position
from .utils.native_timing import NativeTiming
from .utils.wandb import create_native_run


def create_accelerator(config, *, cpu=False):
    precision = config['train']['precision']
    if precision not in ('bf16-mixed', '32-true', 32):
        raise ValueError('Native training currently supports bf16-mixed and 32-true')
    return Accelerator(cpu=cpu, mixed_precision='bf16' if precision == 'bf16-mixed' else 'no',
        gradient_accumulation_steps=1, step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False, find_unused_parameters=False)])


def rng_state(device):
    result = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if device.type == 'cuda':
        result['cuda'] = torch.cuda.get_rng_state(device)
    return result


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda' and 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'], device)


def gather_rng(device, world):
    value = rng_state(device)
    if world == 1:
        return [value]
    values = [None] * world
    dist.all_gather_object(values, value)
    return values


def run_training(config, *, accelerator=None, system=None, loaders=None, logger_factory=create_native_run,
                 save_checkpoints=True, observer=None):
    """Injectable components support matched CPU/GPU regression diagnostics.

    Neither the already sharded loader nor the scheduler is prepared
    by Accelerate: this prevents a second data shard or a world-size LR step.
    """
    accelerator = accelerator or create_accelerator(config)
    rank, world = accelerator.process_index, accelerator.num_processes
    expected = config['ddp']['num_nodes'] * config['ddp']['num_gpus']
    if world != expected:
        raise ValueError(f'WORLD_SIZE {world} differs from configured GPUs {expected}')
    if accelerator.gradient_accumulation_steps != 1:
        raise ValueError('The explicit loop owns accumulation; Accelerator must use accumulation_steps=1')
    seed_process(config['seed'])
    torch.set_float32_matmul_precision('high')
    if config['model'].get('compile_blocks'):
        import torch._inductor.select_algorithm as selection
        selection.PRINT_AUTOTUNE = False
    system = system or FlowSystem(config)
    train_loader, val_loader = loaders or make_loaders(config, rank, world)
    if len(train_loader) < 1:
        raise ValueError('Training loader is empty')
    total_steps = estimated_updates(config, len(train_loader))
    optimizer, scheduler = make_optimizer_scheduler(system, config, total_steps)
    io = NativeCheckpointIO()
    payload = None
    epoch_start = batch_start = global_step = 0
    resume = config['checkpoint'].get('resume')
    if resume:
        payload = io.load_checkpoint(resume)
        system.load_state_dict(payload['state_dict'], strict=True)
        guessed_epoch = payload.get('flowtse_trainer', {}).get('next_epoch', int(payload['epoch']))
        if hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(guessed_epoch)
        epoch_start, batch_start = resume_position(payload, len(train_loader), config['train']['accumulation_steps'])
        global_step = int(payload['global_step'])
    # Parameter order and names stay intact. Only model/optimizer are wrapped.
    prepared, optimizer = accelerator.prepare(system, optimizer)
    system = accelerator.unwrap_model(prepared)
    if payload:
        # Restore after prepare: old Accelerate versions otherwise move CPU Adam
        # step counters to CUDA, introducing a device sync per parameter/update.
        optimizer.load_state_dict(payload['optimizer_states'][0])
        scheduler.load_state_dict(payload['lr_schedulers'][0])
        apply_resumed_lr_policy(optimizer, scheduler, config)
    directory = Path(config['checkpoint']['dir'])
    directory.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        (directory / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        print(f'Trainer: Accelerate | world {world} | precision {config["train"]["precision"]} | '
              f'Separator parameters: {sum(p.numel() for p in system.model.parameters()):,}', flush=True)
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.8g} | '
              f'schedule {config["optimizer"].get("schedule", "cosine")} | '
              f'resume step {global_step} | target epochs {config["train"]["num_epochs"]}', flush=True)
    manager = CheckpointManager(config, io) if accelerator.is_main_process and save_checkpoints else None
    if manager and payload:
        manager.restore(payload)
    logger = logger_factory(config) if accelerator.is_main_process and config['train'].get('wandb') else None
    accelerator.wait_for_everyone()
    # Lightning DDP resets its process seed during distributed setup, after
    # model construction. Match that point before DataLoader worker seeds are
    # drawn; native resume RNG states below still take precedence.
    seed_process(config['seed'])
    previous_timing = None
    if payload:
        callbacks = payload.get('callbacks', {})
        previous_timing = callbacks.get('NativeTiming') or callbacks.get('TrainingTimeEstimator')
        states = payload.get('flowtse_trainer', {}).get('rng_states')
        if states:
            if len(states) != world:
                raise ValueError('RNG recovery requires the original DDP world size')
            restore_rng(states[rank], accelerator.device)
    timer = NativeTiming(previous_timing, known=not bool(resume))
    del payload
    accumulation = config['train']['accumulation_steps']
    interval = max(1, config['train']['log_interval'])
    budget = config['train']['max_steps']
    stopped = False
    requested = [False]
    handlers = {}
    if __import__('threading').current_thread() is __import__('threading').main_thread():
        def request_stop(signum, frame):
            requested[0] = True
        for signum in (signal.SIGTERM, signal.SIGINT):
            handlers[signum] = signal.signal(signum, request_stop)
    last_loss, last_norm = None, None
    last_metrics = {}
    epoch, next_epoch, next_batch = epoch_start, epoch_start, batch_start

    def log(metrics):
        if logger is not None and metrics:
            # Log completed epoch progress while keeping checkpoint counters
            # integral. Epoch-end validation is at 1.0, 2.0, ...; partial
            # validation and resumed training use their actual batch position.
            progress = next_epoch + next_batch / batches
            logger.log(dict(metrics, epoch=progress, **{'trainer/global_step': max(0, global_step - 1)}))

    def checkpoint(metrics, *, last):
        if not save_checkpoints:
            return
        timer.flush()
        started = time.perf_counter()
        states = gather_rng(accelerator.device, world)
        if manager:
            def build(ranking):
                return dict(state_dict=system.state_dict(), optimizer_states=[optimizer.state_dict()],
                    lr_schedulers=[scheduler.state_dict()], epoch=epoch, global_step=global_step,
                    hyper_parameters={'config': config},
                    callbacks={'NativeCheckpointManager': ranking, 'NativeTiming': timer.state_dict()},
                    loops={'fit_loop': {'epoch_progress': {'current': {'processed': next_epoch, 'completed': next_epoch}}}},
                    flowtse_trainer=dict(format_version=1, backend='accelerate', next_epoch=next_epoch,
                        next_batch=next_batch, world_size=world, rng_states=states))
            manager.save(build, epoch, global_step, metrics, last=last)
        accelerator.wait_for_everyone()
        timer.overhead_end(started)

    try:
        for epoch in range(epoch_start, config['train']['num_epochs']):
            if budget > 0 and global_step >= budget:
                break
            sampler = train_loader.batch_sampler
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)
            batches = len(train_loader)
            start_batch = batch_start if epoch == epoch_start else 0
            next_epoch, next_batch = epoch, start_batch
            system.train()
            optimizer.zero_grad(set_to_none=True)
            validation_done = False
            val_enabled = val_loader is not None and (epoch + 1) % config['validation'].get('every_n_epochs', 1) == 0
            val_interval = config['train'].get('val_check_interval', 1.)
            val_batches = val_interval if type(val_interval) is int else max(1, int(batches * val_interval))
            if val_batches < 1:
                raise ValueError('val_check_interval must be positive')
            for batch_index, batch in enumerate(train_loader):
                if batch_index < start_batch:
                    continue
                if observer:
                    observer('batch_start', dict(epoch=epoch, batch=batch_index, step=global_step))
                batch = move_batch(batch, accelerator.device)
                update = (batch_index + 1) % accumulation == 0 or batch_index + 1 == batches
                context = contextlib.nullcontext() if update else accelerator.no_sync(prepared)
                with context:
                    with accelerator.autocast():
                        # DDP also transfers forward arguments. CPU metadata is
                        # bookkeeping, so keep it outside the wrapped forward.
                        loss = prepared({key: value for key, value in batch.items() if key != 'metadata'})
                    accelerator.backward(loss / accumulation)
                next_epoch, next_batch = epoch, batch_index + 1
                if not update:
                    timer.batch_end()
                    continue
                clip = config['train']['gradient_clip_val']
                last_norm = accelerator.clip_grad_norm_(system.model.parameters(), clip) if clip > 0 else None
                rates = [group['lr'] for group in optimizer.param_groups]
                optimizer.step()
                scheduler.step()  # Exactly one step per completed optimizer update.
                global_step += 1
                last_loss = loss.detach()
                optimizer.zero_grad(set_to_none=True)
                timer.batch_end()
                at_end = batch_index + 1 == batches
                at_budget = budget > 0 and global_step >= budget
                if global_step % interval == 0 or at_end or at_budget:
                    if (not bool(torch.isfinite(last_loss))
                            or (last_norm is not None and not bool(torch.isfinite(last_norm)))):
                        raise FloatingPointError('Non-finite training loss or gradient norm')
                    if world > 1:
                        flag = torch.tensor(int(requested[0]), device=accelerator.device, dtype=torch.int32)
                        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                        stopped = bool(flag.item())
                    else:
                        stopped = requested[0]
                    if accelerator.is_main_process:
                        metrics = {'train/fm_loss': float(last_loss)}
                        metrics.update({('lr-AdamW' if len(rates) == 1 else f'lr-AdamW/pg{i+1}'): lr
                                        for i, lr in enumerate(rates)})
                        metrics.update(timer.report(config, epoch, next_batch, batches, global_step))
                        log(metrics)
                if observer:
                    observer('batch_end', dict(epoch=epoch, batch=batch_index, step=global_step,
                        loss=last_loss, gradient_norm=last_norm))
                if at_end:
                    next_epoch, next_batch = epoch + 1, 0
                should_validate = val_enabled and not stopped and ((batch_index + 1) % val_batches == 0 or at_end or at_budget)
                if should_validate:
                    timer.flush()
                    last_metrics, seconds, scoring = validate(system, val_loader, config, epoch, accelerator)
                    timer.validation_end(seconds, scoring)
                    validation_done = True
                    if accelerator.is_main_process:
                        print(f'Validation timing | epoch {epoch+1} | total {seconds:.1f}s | scoring {scoring:.1f}s', flush=True)
                        log(last_metrics)
                    checkpoint(last_metrics, last=False)
                if at_budget or stopped:
                    break
            timer.flush()
            save_last = ((epoch + 1) % config['checkpoint'].get('last_every_n_epochs', 1) == 0
                         or stopped or (budget > 0 and global_step >= budget))
            if save_last:
                checkpoint({}, last=True)
            if accelerator.is_main_process:
                pieces = [f'Epoch {epoch+1}/{config["train"]["num_epochs"]}',
                          f'train loss (last) {float(last_loss):.5f}' if last_loss is not None else 'no updates',
                          'LR ' + ','.join(f'{group["lr"]:.3g}' for group in optimizer.param_groups)]
                if validation_done:
                    pieces += [f'{key} {last_metrics[key]:.4f}' for key in
                        ('val/fm_loss', 'val/si_sdr', 'val/pesq', 'val/estoi', 'val/dnsmos_ovrl', 'val/dnsmos_p808')
                        if key in last_metrics]
                print(' | '.join(pieces), flush=True)
                log(timer.report(config, next_epoch, next_batch, batches, global_step, force=True))
            batch_start = 0
            if stopped or (budget > 0 and global_step >= budget):
                break
        accelerator.wait_for_everyone()
        return dict(global_step=global_step, next_epoch=next_epoch, next_batch=next_batch,
                    stopped=stopped, metrics=last_metrics, timing=timer.state_dict())
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        if logger is not None:
            logger.finish()
