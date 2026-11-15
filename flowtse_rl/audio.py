"""Offline, length-aware BigVGAN v2 mel analysis.

All waveforms passed to the frontend are already at 24 kHz. Analysis precedes
batch padding. A final partial hop is extended at EOS so synthesis covers every
real sample. For hop-aligned inputs this equals the vendored official frontend.
"""
import math

import numpy as np
import soundfile as sf
import torch
import torchaudio
from librosa.filters import mel as mel_filter
from torch import nn
from torch.nn import functional as F


def load_audio(path, sample_rate):
    samples, source_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.mean(samples, axis=1))
    if waveform.numel() == 0:
        raise ValueError(f"Empty audio: {path}")
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return waveform


def rms(waveform, eps=1e-8):
    return waveform.square().mean().clamp_min(eps).sqrt()


def mixture_gain(mixture, target_rms, peak=0.99):
    gain = float(target_rms) / rms(mixture)
    return torch.minimum(gain, mixture.new_tensor(peak) / mixture.abs().max().clamp_min(1e-8))


def normalize_mixture(mixture, components, target_rms, peak=0.99):
    gain = mixture_gain(mixture, target_rms, peak)
    return mixture * gain, [item * gain for item in components]


def crop_or_repeat(waveform, length, start=0):
    if waveform.numel() == 0:
        raise ValueError("Cannot crop empty audio")
    if waveform.numel() < length:
        waveform = waveform.repeat(math.ceil(length / waveform.numel()))
    start = min(max(start, 0), waveform.numel() - length)
    return waveform[start:start + length]


def scale_to_snr(reference, signal, snr_db):
    return signal * rms(reference) / (rms(signal) * 10 ** (snr_db / 20))


def pad_waveforms(items):
    lengths = torch.tensor([item.numel() for item in items], dtype=torch.long)
    return torch.stack([F.pad(item, (0, int(lengths.max()) - item.numel())) for item in items]), lengths


def lengths_to_mask(lengths, maximum=None):
    maximum = int(lengths.max()) if maximum is None else maximum
    return torch.arange(maximum, device=lengths.device)[None] < lengths[:, None]


class BigVGANMel(nn.Module):
    def __init__(self, sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256,
                 n_mels=100, f_min=0.0, f_max=None, log_clip=1e-5, **_):
        super().__init__()
        self.sample_rate, self.n_fft = sample_rate, n_fft
        self.win_length, self.hop_length = win_length, hop_length
        self.n_mels, self.log_clip = n_mels, log_clip
        basis = mel_filter(sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
                           fmin=f_min, fmax=f_max, htk=False, norm="slaney")
        self.register_buffer("basis", torch.from_numpy(basis).float(), persistent=False)
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def analyze(self, waveform):
        if waveform.ndim != 2 or waveform.shape[-1] < 1:
            raise ValueError("Expected nonempty [batch, samples] audio")
        with torch.autocast(waveform.device.type, enabled=False):
            waveform = waveform.float()
            frames = math.ceil(waveform.shape[-1] / self.hop_length)
            waveform = F.pad(waveform[:, None], (0, frames * self.hop_length - waveform.shape[-1]), mode="replicate")
            side = (self.n_fft - self.hop_length) // 2
            mode = "reflect" if waveform.shape[-1] > side else "replicate"
            waveform = F.pad(waveform, (side, side), mode=mode)[:, 0]
            spec = torch.stft(waveform, self.n_fft, self.hop_length, self.win_length,
                              self.window.float(), center=False, return_complex=True)
            magnitude = (spec.real.square() + spec.imag.square() + 1e-9).sqrt()
            return (self.basis.float() @ magnitude).clamp_min(self.log_clip).log()

    def forward(self, waveform, lengths=None):
        if lengths is None:
            lengths = torch.full((waveform.shape[0],), waveform.shape[-1], device=waveform.device, dtype=torch.long)
        lengths = lengths.to(waveform.device, torch.long)
        if lengths.shape != (waveform.shape[0],) or bool(((lengths < 1) | (lengths > waveform.shape[-1])).any()):
            raise ValueError("Invalid audio lengths")
        counts = (lengths + self.hop_length - 1) // self.hop_length
        result = waveform.new_zeros((len(lengths), self.n_mels, int(counts.max())), dtype=torch.float32)
        for length in lengths.unique().tolist():
            indices = (lengths == length).nonzero().flatten()
            features = self.analyze(waveform[indices, :length])
            result[indices, :, :features.shape[-1]] = features
        return result, counts


class ComplexSTFT(nn.Module):
    """Uncompressed complex STFT: [real frequencies, imaginary frequencies].

    Analyze real lengths before batch padding. Normal utterances use a
    centered Hann/reflect STFT; very short inputs use replicate boundary padding.
    FFT and reconstruction stay FP32 even inside BF16 model autocast.
    """
    def __init__(self, sample_rate=16000, n_fft=510, win_length=510, hop_length=128, **_):
        super().__init__()
        if type(n_fft) is not int or n_fft < 2 or n_fft % 2:
            raise ValueError('Complex STFT requires an even n_fft >=2')
        if not 0 < hop_length < win_length <= n_fft:
            raise ValueError('Require 0 < hop_length < win_length <= n_fft')
        self.sample_rate, self.n_fft = sample_rate, n_fft
        self.win_length, self.hop_length = win_length, hop_length
        self.feature_dim = 2 * (n_fft // 2 + 1)
        self.register_buffer('window', torch.hann_window(win_length), persistent=False)

    def analyze(self, waveform):
        if waveform.ndim != 2 or waveform.shape[-1] < 1:
            raise ValueError('Expected nonempty [batch, samples] audio')
        with torch.autocast(waveform.device.type, enabled=False):
            mode = 'reflect' if waveform.shape[-1] > self.n_fft // 2 else 'replicate'
            spectrum = torch.stft(waveform.float(), self.n_fft, self.hop_length, self.win_length,
                self.window.to(waveform.device).float(), center=True, pad_mode=mode,
                normalized=False, onesided=True, return_complex=True)
            return torch.cat((spectrum.real, spectrum.imag), dim=1)

    def forward(self, waveform, lengths=None):
        if waveform.ndim != 2:
            raise ValueError('Expected [batch, samples] audio')
        if lengths is None:
            lengths = torch.full((len(waveform),), waveform.shape[-1], device=waveform.device, dtype=torch.long)
        lengths = lengths.to(waveform.device, torch.long)
        if lengths.shape != (len(waveform),) or bool(((lengths < 1) | (lengths > waveform.shape[-1])).any()):
            raise ValueError('Invalid audio lengths')
        counts = lengths // self.hop_length + 1
        result = waveform.new_zeros((len(waveform), self.feature_dim, int(counts.max())), dtype=torch.float32)
        for length in lengths.unique().tolist():
            indices = (lengths == length).nonzero().flatten()
            features = self.analyze(waveform[indices, :length])
            result[indices, :, :features.shape[-1]] = features
        return result, counts


class ComplexISTFT(nn.Module):
    """Exact-length whole-utterance inverse; no neural vocoder or model weights."""
    def __init__(self, n_fft=510, win_length=510, hop_length=128, **_):
        super().__init__()
        self.n_fft, self.win_length, self.hop_length = n_fft, win_length, hop_length
        self.register_buffer('window', torch.hann_window(win_length), persistent=False)

    def forward(self, features, samples):
        frequencies = self.n_fft // 2 + 1
        if features.ndim != 3 or features.shape[1] != 2 * frequencies:
            raise ValueError('Expected packed real/imaginary STFT [B,2F,T]')
        samples = torch.as_tensor(samples, device=features.device, dtype=torch.long)
        if samples.shape != (len(features),) or bool((samples < 1).any()):
            raise ValueError('Invalid synthesis sample counts')
        output = []
        with torch.autocast(features.device.type, enabled=False):
            for index, count in enumerate(samples.tolist()):
                frames = count // self.hop_length + 1
                if features.shape[-1] < frames:
                    raise ValueError('Not enough STFT frames for requested waveform length')
                packed = features[index, :, :frames].float()
                spectrum = torch.complex(packed[:frequencies], packed[frequencies:])
                waveform = torch.istft(spectrum, self.n_fft, self.hop_length, self.win_length,
                    self.window.to(features.device).float(), center=True, normalized=False,
                    onesided=True, length=count)
                output.append(waveform)
        return output

