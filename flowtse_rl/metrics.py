import math

import torch
import torchaudio


def si_sdr(estimate, target, eps=1e-8):
    estimate, target = estimate.float(), target.float()
    estimate = estimate - estimate.mean()
    target = target - target.mean()
    projected = torch.dot(estimate, target) * target / target.square().sum().clamp_min(eps)
    return 10 * torch.log10(projected.square().sum().clamp_min(eps) / (estimate - projected).square().sum().clamp_min(eps))


def audio_metrics(estimate, target, sample_rate=24000, names=("si_sdr", "pesq", "estoi")):
    if estimate.numel() != target.numel():
        raise ValueError("Metric waveforms must have identical full lengths; do not truncate silently")
    result = {}
    if "si_sdr" in names:
        result["si_sdr"] = float(si_sdr(estimate, target))
    if "pesq" in names or "estoi" in names:
        est = torchaudio.functional.resample(estimate.detach().cpu().float(), sample_rate, 16000).numpy()
        ref = torchaudio.functional.resample(target.detach().cpu().float(), sample_rate, 16000).numpy()
        if "pesq" in names:
            from pesq import pesq
            try:
                result["pesq"] = float(pesq(16000, ref, est, "wb"))
            except Exception:
                result["pesq"] = math.nan
        if "estoi" in names:
            from pystoi import stoi
            try:
                result["estoi"] = float(stoi(ref, est, 16000, extended=True))
            except Exception:
                result["estoi"] = math.nan
    return result

