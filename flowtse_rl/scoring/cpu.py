"""Production LibriMix metric preprocessing, isolated from CUDA model workers."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path):
    import librosa
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    audio = audio.mean(axis=1)
    if rate != 16000:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f'Expected nonempty finite audio: {path}')
    peak = np.max(np.abs(audio))
    return audio * .9 / peak if peak > 0 else audio


def si_sdr(estimate, reference):
    estimate = estimate - np.mean(estimate)
    reference = reference - np.mean(reference)
    energy = np.sum(reference ** 2)
    if energy == 0 or np.sum(estimate ** 2) == 0:
        return float('nan')
    projection = np.sum(reference * estimate) / energy * reference
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(10 * np.log10(np.sum(projection ** 2) / np.sum((estimate - projection) ** 2)))


def calculate_pair(job):
    prediction, reference, metrics = job
    clean, predicted = load_audio(reference), load_audio(prediction)
    if clean.shape != predicted.shape:
        raise ValueError(f'Full waveform lengths differ; refusing truncation: {prediction}')
    row = dict(filename=Path(reference).name, error='')
    if 'si_sdr' in metrics:
        row['sisdr'] = si_sdr(predicted, clean)
    if 'pesq' in metrics:
        from pesq import pesq, PesqError
        try:
            if not np.any(clean - clean.mean()) or not np.any(predicted - predicted.mean()):
                raise ValueError('PESQ is undefined for silent or constant audio')
            row['pesq'] = float(pesq(16000, clean, predicted, 'wb'))
        except (PesqError, ValueError) as error:
            row['pesq'] = float('nan')
            row['error'] = f'PESQ: {error}'
    if 'estoi' in metrics:
        from pystoi import stoi
        row['estoi'] = float(stoi(clean, predicted, 16000, extended=True))
    return row


def score_pairs(pairs, metrics, workers=1):
    jobs = [(str(prediction), str(reference), tuple(metrics)) for prediction, reference in pairs]
    if workers == 1:
        return [calculate_pair(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)),
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        return list(pool.map(calculate_pair, jobs))
