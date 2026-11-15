"""Full-utterance feature analysis, bounded Flow chunks and one decoder pass."""
from dataclasses import dataclass
import math

import torch

from .audio import mixture_gain, rms
from .flow import sample_ode
from .representation import COMPLEX_STFT, frame_count, representation


INFERENCE_PROTOCOL = "full_mel_flow_chunks_single_bigvgan_v1"
COMPLEX_INFERENCE_PROTOCOL = "full_stft_flow_chunks_single_istft_v1"


def inference_protocol(config):
    return COMPLEX_INFERENCE_PROTOCOL if representation(config) == COMPLEX_STFT else INFERENCE_PROTOCOL


@dataclass
class ChunkPrediction:
    frame_start: int
    frame_stop: int
    mixture_features: torch.Tensor
    generated_features: torch.Tensor
    enrollment_features: torch.Tensor
    mixture_lengths: torch.Tensor
    enrollment_lengths: torch.Tensor
    gain: torch.Tensor

    # Preserve the existing Mel inference API for saved tools and regressions.
    mixture_mel = property(lambda self: self.mixture_features)
    generated_mel = property(lambda self: self.generated_features)
    enrollment_mel = property(lambda self: self.enrollment_features)


@dataclass
class UtterancePrediction:
    waveform: torch.Tensor
    generated_features: torch.Tensor
    chunks: list
    gain: torch.Tensor

    generated_mel = property(lambda self: self.generated_features)


def inference_settings(config, steps=None, chunk_seconds=None):
    if config.get('inference', {}).get('protocol', inference_protocol(config)) != inference_protocol(config):
        raise ValueError('Inference protocol does not match the audio representation')
    steps = config.get("flow", {}).get("steps", 16) if steps is None else steps
    seconds = config.get("inference", {}).get("chunk_seconds", 3.0) if chunk_seconds is None else chunk_seconds
    if type(steps) is not int or steps < 1:
        raise ValueError("Flow steps must be a positive integer")
    if not math.isfinite(float(seconds)) or seconds <= 0:
        raise ValueError("inference.chunk_seconds must be positive and finite")
    samples = round(float(seconds) * config["audio"]["sample_rate"])
    if samples < 1:
        raise ValueError("Inference chunk must contain at least one sample")
    return steps, float(seconds), samples


@torch.no_grad()
def iter_chunked(module, mixture, enrollment, *, seed=42, steps=None, chunk_seconds=None):
    """Analyze the whole mixture once, then yield generated feature chunks.

    Frame spans partition the utterance without gaps or repeated boundary
    padding. The decoder is outside this iterator. Each Flow input has at most
    the selected frontend's frame count for a waveform crop of that duration.
    """
    steps, _, chunk_samples = inference_settings(module.config, steps, chunk_seconds)
    if mixture.ndim != 1 or enrollment.ndim != 1 or min(mixture.numel(), enrollment.numel()) < 1:
        raise ValueError("Expected nonempty one-dimensional mixture and enrollment")
    mixture, enrollment = mixture.to(module.device), enrollment.to(module.device)
    audio = module.config["audio"]
    maximum = round(module.config["data"]["enrollment_seconds"][1] * audio["sample_rate"])
    enrollment = enrollment[:maximum]
    enrollment = enrollment * (audio["target_rms"] / rms(enrollment))
    enroll, elengths = module.frontend(enrollment[None])
    gain = mixture_gain(mixture, audio["target_rms"])
    observed, lengths = module.frontend((mixture * gain)[None])
    chunk_frames = frame_count(module.config, chunk_samples)
    total_frames = int(lengths[0])
    generator = torch.Generator(device=module.device).manual_seed(seed)
    for start in range(0, total_frames, chunk_frames):
        stop = min(start + chunk_frames, total_frames)
        segment = observed[..., start:stop]
        segment_lengths = lengths.new_tensor([stop - start])
        generated = sample_ode(module.model, segment, enroll, segment_lengths, elengths,
                               steps=steps, solver=module.config["flow"]["solver"], generator=generator)
        yield ChunkPrediction(start, stop, segment, generated, enroll, segment_lengths, elengths, gain)


@torch.no_grad()
def predict_chunked(module, mixture, enrollment, *, seed=42, steps=None, chunk_seconds=None):
    """Join feature chunks and decode the complete utterance exactly once."""
    chunks = list(iter_chunked(module, mixture, enrollment, seed=seed, steps=steps,
                               chunk_seconds=chunk_seconds))
    generated = torch.cat([chunk.generated_features for chunk in chunks], dim=-1)
    samples = torch.tensor([mixture.numel()], device=module.device)
    gain = chunks[0].gain
    waveform = module.vocoder(generated, samples)[0] / gain
    if waveform.numel() != mixture.numel():
        raise RuntimeError("Full-utterance decoder must preserve every input sample")
    return UtterancePrediction(waveform, generated, chunks, gain)


@torch.no_grad()
def extract_chunked(module, mixture, enrollment, *, seed=42, steps=None, chunk_seconds=None):
    return predict_chunked(module, mixture, enrollment, seed=seed, steps=steps,
                           chunk_seconds=chunk_seconds).waveform
