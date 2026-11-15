"""RL building blocks for the no-CFG Gaussian-source baseline.

This module supplies LoRA, windowed SDE rollouts and a GRPO loss. Reward
selection and a production post-training orchestration loop are deliberately
separate from the supervised baseline. Nothing here claims a trained RL model.
SDE geometry follows FlowSE-GRPO Eq. (6), in the noise-to-data 0->1 direction.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch
from torch import nn

from .audio import lengths_to_mask


class LoRALinear(nn.Module):
    def __init__(self, base, rank=32, alpha=64):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale, self.enabled = alpha / rank, True

    def forward(self, value):
        result = self.base(value)
        if self.enabled:
            result = result + ((value @ self.lora_a.T) @ self.lora_b.T) * self.scale
        return result


def inject_lora(model, rank=32, alpha=64):
    targets = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.proj", "mlp.fc1", "mlp.fc2")
    selected = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, nn.Linear) and name.endswith(targets)]
    if not selected or any(isinstance(layer, LoRALinear) for layer in model.modules()):
        raise ValueError("Expected an unwrapped UDiT model")
    model.requires_grad_(False)
    for name, layer in selected:
        parent, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), child, LoRALinear(layer, rank, alpha))
    return [name for name, _ in selected]


@contextmanager
def reference_policy(model):
    layers = [layer for layer in model.modules() if isinstance(layer, LoRALinear)]
    enabled = [layer.enabled for layer in layers]
    try:
        for layer in layers:
            layer.enabled = False
        yield
    finally:
        for layer, previous in zip(layers, enabled):
            layer.enabled = previous


def sde_stats(state, velocity, time, dt, noise_level=0.4):
    time = torch.as_tensor(time, device=state.device, dtype=torch.float32)
    if time.ndim == 1:
        time = time[:, None, None]
    if dt <= 0 or noise_level <= 0 or bool(((time <= 0) | (time >= 1)).any()):
        raise ValueError("SDE requires dt>0, noise_level>0 and 0<t<1")
    sigma = noise_level * ((1 - time) / time).sqrt()
    state, velocity = state.float(), velocity.float()
    mean = state + (velocity + sigma.square() / (2 * (1 - time)) * (-state + time * velocity)) * dt
    return mean, sigma * math.sqrt(dt)


def masked_mean(values, lengths):
    mask = lengths_to_mask(lengths, values.shape[-1])[:, None]
    return values.masked_fill(~mask, 0).sum((1, 2)) / (lengths * values.shape[1])


def transition_log_density(action, mean, std, lengths):
    # Mean valid-element log density is an explicit high-dimensional surrogate,
    # not the exact joint log probability of a complete utterance.
    values = -0.5 * (((action.detach().float() - mean.float()) / std).square()
                     + 2 * std.log() + math.log(2 * math.pi))
    return masked_mean(values, lengths)


def group_advantages(rewards, epsilon=1e-8):
    if rewards.ndim != 2 or rewards.shape[1] < 2 or not bool(torch.isfinite(rewards).all()):
        raise ValueError("Require finite [prompts, group>=2] rewards")
    rewards = rewards.float()
    standard = rewards.std(dim=1, correction=0)
    eligible = standard > epsilon
    advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / standard.clamp_min(epsilon)[:, None]
    return advantages.masked_fill(~eligible[:, None], 0), eligible


@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    time: torch.Tensor
    dt: float
    old_log_density: torch.Tensor


@torch.no_grad()
def sample_window_sde(model, mixture, enrollment, mixture_lengths, enrollment_lengths,
                      steps=16, start_step=1, window_size=2, noise_level=0.4, generator=None):
    if not 1 <= start_step or window_size < 1 or start_step + window_size > steps:
        raise ValueError("SDE window must avoid step zero and fit inside the time grid")
    state = torch.randn(mixture.shape, device=mixture.device, dtype=torch.float32, generator=generator)
    mask = lengths_to_mask(mixture_lengths, mixture.shape[-1])[:, None]
    state = state.masked_fill(~mask, 0)
    records = []
    for index in range(steps):
        time = torch.full((len(state),), index / steps, device=state.device)
        velocity = model(state, time, enrollment, mixture, mixture_lengths, enrollment_lengths).float()
        if start_step <= index < start_step + window_size:
            mean, std = sde_stats(state, velocity, time, 1 / steps, noise_level)
            noise = torch.randn(state.shape, device=state.device, generator=generator)
            action = (mean + std * noise).masked_fill(~mask, 0)
            log_density = transition_log_density(action, mean, std, mixture_lengths)
            records.append(Transition(state.detach(), action.detach(), time, 1 / steps, log_density.detach()))
        else:
            action = (state + velocity / steps).masked_fill(~mask, 0)
        state = action
    return state, records


def grpo_loss(model, transitions, advantages, mixture, enrollment, mixture_lengths, enrollment_lengths,
               noise_level=0.4, clip_range=0.2, kl_weight=0.01):
    if not transitions:
        raise ValueError("At least one stochastic transition is required")
    if not any(isinstance(layer, LoRALinear) for layer in model.modules()):
        raise ValueError("This reference implementation requires LoRA on a frozen SFT backbone")
    losses, ratios, divergences = [], [], []
    for record in transitions:
        velocity = model(record.state, record.time, enrollment, mixture, mixture_lengths, enrollment_lengths)
        mean, std = sde_stats(record.state, velocity, record.time, record.dt, noise_level)
        new_density = transition_log_density(record.action, mean, std, mixture_lengths)
        ratio = (new_density - record.old_log_density).clamp(-20, 20).exp()
        with torch.no_grad(), reference_policy(model):
            reference_velocity = model(record.state, record.time, enrollment, mixture, mixture_lengths, enrollment_lengths)
            reference_mean, _ = sde_stats(record.state, reference_velocity, record.time, record.dt, noise_level)
        kl = masked_mean((mean - reference_mean).square() / (2 * std.square()), mixture_lengths)
        objective = torch.minimum(ratio * advantages, ratio.clamp(1 - clip_range, 1 + clip_range) * advantages)
        losses.append((-objective + kl_weight * kl).mean())
        ratios.append(ratio.detach().mean())
        divergences.append(kl.detach().mean())
    return torch.stack(losses).mean(), {"ratio": torch.stack(ratios).mean(), "reference_kl": torch.stack(divergences).mean()}

