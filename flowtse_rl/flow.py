"""Ordinary Gaussian-to-target conditional flow matching; no CFG or consistency loss."""
import torch

from .audio import lengths_to_mask


def masked_mse(error, lengths):
    mask = lengths_to_mask(lengths.to(error.device), error.shape[-1])[:, None]
    squared = error.float().masked_fill(~mask, 0).square()
    # Equal utterance weight, independent of its duration.
    return (squared.sum((1, 2)) / (lengths.to(error.device) * error.shape[1])).mean()


def flow_matching_loss(model, target, mixture, enrollment, mixture_lengths, enrollment_lengths,
                       noise=None, time=None):
    noise = torch.randn_like(target) if noise is None else noise
    time = torch.rand(target.shape[0], device=target.device) if time is None else time
    state = (1 - time[:, None, None]) * noise + time[:, None, None] * target
    velocity = model(state, time, enrollment, mixture, mixture_lengths, enrollment_lengths)
    return masked_mse(velocity - (target - noise), mixture_lengths)


@torch.no_grad()
def sample_ode(model, mixture, enrollment, mixture_lengths, enrollment_lengths,
               steps=16, solver="euler", noise=None, generator=None):
    if steps < 1 or solver not in {"euler", "midpoint"}:
        raise ValueError("Require positive steps and euler/midpoint solver")
    state = torch.randn(mixture.shape, device=mixture.device, dtype=torch.float32, generator=generator) if noise is None else noise.float().clone()
    mask = lengths_to_mask(mixture_lengths, mixture.shape[-1])[:, None]
    state = state.masked_fill(~mask, 0)
    for index in range(steps):
        time = torch.full((len(state),), index / steps, device=state.device)
        velocity = model(state, time, enrollment, mixture, mixture_lengths, enrollment_lengths).float()
        if solver == "midpoint":
            middle = state + velocity / (2 * steps)
            velocity = model(middle, time + 0.5 / steps, enrollment, mixture, mixture_lengths, enrollment_lengths).float()
        state = (state + velocity / steps).masked_fill(~mask, 0)
    return state

