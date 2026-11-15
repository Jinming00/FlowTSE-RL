"""Portable quality scoring, with explicit coverage and per-item records."""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import shlex
import sys

import torch

from .metrics import audio_metrics

import soundfile as sf


QUALITY_COLUMNS = {"si_sdr": "sisdr", "pesq": "pesq", "estoi": "estoi",
                   "dnsmos_ovrl": "OVRL", "dnsmos_sig": "SIG", "dnsmos_bak": "BAK",
                   "dnsmos_p808": "P808_MOS"}


def validation_size(size, settings):
    if size < 1:
        raise ValueError("Validation dataset is empty")
    # Keep legacy snapshots readable; new configurations use the H1 ratio.
    if "loss_ratio" not in settings and "num_loss_samples" in settings:
        return min(size, int(settings["num_loss_samples"]) or size)
    ratio = float(settings.get("loss_ratio", 0.2))
    if not 0 < ratio <= 1:
        raise ValueError("validation.loss_ratio must be in (0, 1]")
    return max(1, math.ceil(size * ratio))


def quality_size(size, settings):
    ratio = float(settings.get("quality_metrics_ratio", 1.0))
    if not 0 < ratio <= 1:
        raise ValueError("validation.quality_metrics_ratio must be in (0, 1]")
    return min(size, max(1, round(size * ratio)))


def audio_stem(item):
    return f"{item['mixture_id']}__target-{item['utterance_id']}"


def save_quality_pair(directory, item, estimate, target, sample_rate):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if estimate.numel() != target.numel():
        raise ValueError("Quality waveforms must have identical complete lengths")
    stem = audio_stem(item)
    for suffix, wave in (("pred", estimate), ("ref", target)):
        sf.write(directory / f"{stem}_{suffix}.wav", wave.detach().float().cpu().numpy(), sample_rate, subtype="FLOAT")
    return stem + "_ref.wav"


def score_quality(directory, settings, expected_items, *, expected_filenames=None):
    directory = Path(directory)
    predictions = sorted((directory / 'audio').glob('*_pred.wav'))
    references = {path.name.replace('_pred.wav', '_ref.wav') for path in predictions}
    if expected_items < 1 or len(predictions) != expected_items or any(
            not (directory / 'audio' / name).is_file() for name in references):
        raise RuntimeError('Incomplete quality audio pairs; refusing partial scoring')
    if expected_filenames is not None and references != set(expected_filenames):
        raise RuntimeError('Quality audio IDs differ from the selected samples')
    metrics_dir = directory / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    columns = dict(QUALITY_COLUMNS)
    if settings.get('metrics_command'):
        command = settings['metrics_command']
        command = shlex.split(command) if isinstance(command, str) else list(command)
        env = dict(os.environ, TEST_DIR=str((directory / 'audio').resolve()),
                   OUT_DIR=str(metrics_dir.resolve()), MAX_ITEMS=str(expected_items))
        backend_log = metrics_dir / 'backend.log'
        try:
            with backend_log.open('w') as stream:
                subprocess.run(command, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT)
        except (subprocess.CalledProcessError, OSError) as error:
            raise RuntimeError(f'Quality backend failed; output: {backend_log}') from error
        if settings.get('quality_asr', False):
            columns.update(wer='wer', cer='cer')
        if settings.get('quality_speaker_similarity', False):
            columns['speaker_similarity'] = 'sim'
    elif settings.get('scoring_options'):
        from .scoring.runner import Options, COLUMNS
        from dataclasses import asdict
        options = Options(**settings['scoring_options'])
        columns = {name: COLUMNS[name] for name in options.metrics}
        request = metrics_dir / 'scoring_options.json'
        request.write_text(json.dumps(asdict(options), indent=2) + '\n')
        program = os.environ.get('FLOWTSE_SCORING_PYTHON', sys.executable)
        env = dict(os.environ)
        package_root = str(Path(__file__).resolve().parents[1])
        env['PYTHONPATH'] = package_root + os.pathsep + env.get('PYTHONPATH', '')
        threads = env.get('SCORING_CPU_THREADS', '1')
        env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)
        backend_log = metrics_dir / 'backend.log'
        try:
            with backend_log.open('w') as stream:
                subprocess.run([program, '-m', 'flowtse_rl.scoring.cli',
                    '--audio-dir', str((directory / 'audio').resolve()),
                    '--output-dir', str(metrics_dir.resolve()), '--options-json', str(request.resolve())],
                    env=env, check=True, stdout=stream, stderr=subprocess.STDOUT)
        except (subprocess.CalledProcessError, OSError) as error:
            raise RuntimeError(f'Bundled scorer failed; full output: {backend_log}') from error
    else:
        requested = settings.get('metrics', ['si_sdr', 'pesq', 'estoi'])
        if not requested or set(requested) - {'si_sdr', 'pesq', 'estoi'}:
            raise ValueError('Built-in scoring supports si_sdr, pesq and estoi; use metrics_command for other metrics')
        columns = {name: QUALITY_COLUMNS[name] for name in requested}
        rows = []
        for path in predictions:
            name = path.name.replace('_pred.wav', '_ref.wav')
            estimate, rate = sf.read(path, dtype='float32')
            target, target_rate = sf.read(directory / 'audio' / name, dtype='float32')
            if rate != target_rate or estimate.ndim != 1 or target.ndim != 1:
                raise ValueError(f'Invalid quality pair: {name}')
            values = audio_metrics(torch.from_numpy(estimate), torch.from_numpy(target), rate, requested)
            row = {'filename': name, **{column: values[key] for key, column in columns.items()}}
            row['error'] = ','.join(key for key, value in values.items() if not math.isfinite(value))
            rows.append(row)
        with (metrics_dir / '_results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['filename', *columns.values(), 'error'])
            writer.writeheader()
            writer.writerows(rows)
    with (metrics_dir / '_results.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    names = [row['filename'] for row in rows]
    if len(rows) != expected_items or len(set(names)) != expected_items or set(names) != references:
        raise RuntimeError(f'Quality coverage mismatch: {len(rows)}/{expected_items} rows or incorrect IDs')
    values, counts = {}, {}
    for metric, column in columns.items():
        finite = []
        for row in rows:
            try:
                value = float(row.get(column, ''))
            except (ValueError, TypeError):
                continue
            if math.isfinite(value):
                finite.append(value)
        values[metric] = sum(finite) / len(finite) if finite else None
        counts[metric] = len(finite)
    if 'pesq' in columns and values['pesq'] is None:
        raise RuntimeError('No finite PESQ; refusing to choose a best checkpoint')
    report = {'expected_items': expected_items, 'rows': len(rows), 'metrics': values, 'finite_counts': counts}
    if settings.get('scoring_options') and not settings.get('metrics_command'):
        bundled = json.loads((metrics_dir / 'summary.json').read_text())
        report.update(coverage=bundled['coverage'], protocol=bundled['protocol'])
    (metrics_dir / 'coverage.json').write_text(json.dumps(report, indent=2) + '\n')
    keep = max(0, int(settings.get('quality_audio_keep_items', 0)))
    for path in predictions[keep:]:
        path.unlink()
        path.with_name(path.name.replace('_pred.wav', '_ref.wav')).unlink()
    return report
