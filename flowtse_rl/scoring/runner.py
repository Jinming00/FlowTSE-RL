"""Independent, coverage-checked production scoring for paired full utterances."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import csv

import soundfile as sf

COLUMNS = dict(si_sdr='sisdr', pesq='pesq', estoi='estoi', dnsmos_ovrl='OVRL',
    dnsmos_sig='SIG', dnsmos_bak='BAK', dnsmos_p808='P808_MOS', wer='wer', cer='cer',
    speaker_similarity='sim', wespeaker_similarity='sim_wespeaker')
GROUPS = dict(core=['si_sdr', 'pesq', 'estoi'], dnsmos=list(COLUMNS)[3:7],
              asr=['wer', 'cer'], speakers=['speaker_similarity', 'wespeaker_similarity'], full=list(COLUMNS))


def expand_metrics(value):
    value = value.split(',') if isinstance(value, str) else value
    result = []
    for name in value:
        name = name.strip()
        for metric in GROUPS.get(name, [name]):
            if metric not in COLUMNS:
                raise ValueError(f'Unknown scoring metric: {metric}')
            if metric not in result:
                result.append(metric)
    if not result:
        raise ValueError('Select at least one scoring metric')
    return result


@dataclass
class Options:
    metrics: list = field(default_factory=lambda: list(GROUPS['core']))
    workers: int = 4
    device: str = 'cpu'
    dnsmos_workers: int = 1
    dnsmos_dir: str = ''
    qwen_model: str = ''
    wavlm_model: str = ''
    wespeaker_model: str = ''
    asr_batch_size: int = 16
    asr_max_new_tokens: int = 1024
    language: str = 'English'
    wer_reference: str = 'librispeech-shorter'
    librispeech_root: str = ''
    vctk_root: str = ''
    vctk_original_root: str = ''
    vctk_durations_csv: str = ''
    transcripts_json: str = ''
    wespeaker_workers: int = 4
    wespeaker_batch_size: int = 8
    wespeaker_max_batch_frames: int = 8000

    def __post_init__(self):
        self.metrics = expand_metrics(self.metrics)
        if self.device not in {'cpu', 'cuda'}:
            raise ValueError('Scoring device must be cpu or cuda')
        for key in ['workers', 'dnsmos_workers', 'asr_batch_size', 'asr_max_new_tokens',
                    'wespeaker_workers', 'wespeaker_batch_size', 'wespeaker_max_batch_frames']:
            if getattr(self, key) < 1:
                raise ValueError(f'{key} must be positive')
        environment = dict(dnsmos_dir='DNSMOS_DIR', qwen_model='QWEN_ASR_MODEL_ID',
                           wavlm_model='WAVLM_MODEL_PATH', wespeaker_model='WESPEAKER_MODEL_PATH',
                           librispeech_root='LIBRISPEECH_ROOT', vctk_root='VCTK_ROOT',
                           vctk_original_root='VCTK_ORIGINAL_ROOT', vctk_durations_csv='VCTK_DURATIONS_CSV')
        for key, variable in environment.items():
            if not getattr(self, key):
                setattr(self, key, os.environ.get(variable, ''))
        for key in ['dnsmos_dir', 'qwen_model', 'wavlm_model', 'wespeaker_model',
                    'librispeech_root', 'vctk_root', 'vctk_original_root', 'vctk_durations_csv', 'transcripts_json']:
            if getattr(self, key):
                setattr(self, key, str(Path(os.path.expandvars(getattr(self, key))).expanduser().resolve()))


def validate_options(options):
    required = set()
    if 'pesq' in options.metrics:
        required.add('pesq')
    if 'estoi' in options.metrics:
        required.add('pystoi')
    if any(x.startswith('dnsmos_') for x in options.metrics) or 'wespeaker_similarity' in options.metrics:
        required.add('onnxruntime')
    if 'speaker_similarity' in options.metrics:
        required.add('transformers')
    if {'wer', 'cer'} & set(options.metrics):
        required.update(['jiwer', 'whisper_normalizer', 'qwen_asr'])
    for package in sorted(required):
        if importlib.util.find_spec(package) is None:
            raise ImportError(f'Missing scoring dependency: {package}; install the appropriate scoring extra')
    def need(path, description, directory=False):
        if not path or not (Path(path).is_dir() if directory else Path(path).is_file()):
            raise FileNotFoundError(f'Missing {description}: {path or "path not configured"}')
    if any(x.startswith('dnsmos_') for x in options.metrics):
        need(options.dnsmos_dir, 'DNSMOS directory', True)
        for name in ['sig_bak_ovr.onnx', 'model_v8.onnx']:
            need(Path(options.dnsmos_dir) / name, name)
    if 'speaker_similarity' in options.metrics:
        need(options.wavlm_model, 'local WavLM model directory', True)
        need(Path(options.wavlm_model) / 'config.json', 'WavLM config.json')
    if 'wespeaker_similarity' in options.metrics:
        need(options.wespeaker_model, 'pinned WeSpeaker ONNX model')
    if {'wer', 'cer'} & set(options.metrics):
        need(options.qwen_model, 'local Qwen3-ASR model directory', True)
        need(Path(options.qwen_model) / 'config.json', 'Qwen3-ASR config.json')
        reference_provider(options)


def reference_provider(options):
    if options.wer_reference == 'librispeech-shorter':
        if not options.librispeech_root:
            raise ValueError('Set librispeech_root for complete-shorter-target true-text WER/CER')
        from .librispeech_reference import LibriSpeechReferences
        return LibriSpeechReferences(options.librispeech_root)
    if options.wer_reference == 'vctk-shorter':
        from .vctk_reference import VCTKReferences
        return VCTKReferences(options.vctk_root, options.vctk_original_root or None, options.vctk_durations_csv or None)
    if options.wer_reference == 'transcripts':
        if not options.transcripts_json:
            raise ValueError('Set transcripts_json for explicit true-text ASR references')
        records = json.loads(Path(options.transcripts_json).read_text())
        class ExplicitReferences:
            def reference(self, prediction, reference_audio):
                name = Path(reference_audio).name
                if name not in records:
                    raise ValueError(f'Missing true-text transcript entry: {name}')
                value = records[name]
                if isinstance(value, str):
                    value = dict(reference_text=value, wer_eligible=True)
                result = {**dict(wer_eligible=True, wer_skip_reason='', target_utt_id=''), **value}
                if not isinstance(result['wer_eligible'], bool):
                    raise ValueError(f'wer_eligible must be a JSON boolean: {name}')
                if result['wer_eligible'] and not result.get('reference_text', '').strip():
                    raise ValueError(f'Eligible transcript is empty: {name}')
                return result
        return ExplicitReferences()
    raise ValueError('Choose librispeech-shorter, vctk-shorter or transcripts for true-text ASR')


def find_pairs(directory, limit=0):
    directory = Path(directory).resolve()
    if limit < 0:
        raise ValueError('limit must be non-negative')
    predictions = sorted(directory.glob('*_pred.wav'))
    references = set(directory.glob('*_ref.wav'))
    expected = {p.with_name(p.name.replace('_pred.wav', '_ref.wav')) for p in predictions}
    if not predictions or expected != references:
        raise ValueError('Audio directory must contain complete, exactly paired *_pred.wav/*_ref.wav files')
    total = len(predictions)
    pairs = [(p, p.with_name(p.name.replace('_pred.wav', '_ref.wav'))) for p in predictions[:limit or None]]
    for prediction, reference in pairs:
        a, b = sf.info(prediction), sf.info(reference)
        if a.frames < 1 or b.frames < 1 or a.frames * b.samplerate != b.frames * a.samplerate:
            raise ValueError(f'Full waveform durations differ; refusing truncation: {prediction.name}')
    return pairs, total


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def release_models():
    gc.collect()
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_available():
        sys.modules['torch'].cuda.empty_cache()


def score_directory(audio_dir, output_dir, options=None, *, limit=0, expected_filenames=None):
    options = options or Options()
    validate_options(options)
    pairs, available = find_pairs(audio_dir, limit)
    filenames = [reference.name for _, reference in pairs]
    if expected_filenames is not None and set(filenames) != set(expected_filenames):
        raise ValueError('Scoring sample IDs differ from the requested samples')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'COMPLETE.json').exists():
        raise FileExistsError(f'Scoring already completed in {output}; use a new output directory')
    started = time.perf_counter()
    requested = set(options.metrics)
    rows = [dict(filename=name, error='') for name in filenames]
    protocol = dict(version='flowtse_full_utterance_scoring_v1', audio_pairs=len(pairs),
        available_pairs=available, subset=len(pairs) != available, requested_metrics=options.metrics,
        input='full utterances; no truncation', core='librosa 16k resample, mean mono, independent peak 0.9',
        pesq='wideband', estoi='extended', dnsmos='9.01s windows, 1s stride, repeat short audio, non-personalized calibration',
        wer_reference=options.wer_reference, error_rate='fraction; per-utterance rounding to 3 decimals then macro mean',
        wavlm='clean reference vs prediction, first channel, cosine', wespeaker='clean reference vs prediction, (cosine+1)/2',
        device=options.device, stages=[], assets={})
    def save(stage):
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (output / '_results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader(); writer.writerows(rows)
        protocol['stages'].append(stage)
        write_json(output / 'protocol.json', protocol)
        print(f'Scoring: {stage} | {len(pairs)} pairs', flush=True)
    def asset(name, path):
        path = Path(path)
        protocol['assets'][name] = dict(path=str(path), bytes=path.stat().st_size)
    core = [name for name in ['si_sdr', 'pesq', 'estoi'] if name in requested]
    if core:
        from .cpu import score_pairs
        scores = score_pairs(pairs, core, options.workers)
        for row, score in zip(rows, scores):
            row.update(score)
        save('core')
    if any(name.startswith('dnsmos_') for name in requested):
        from .dnsmos import DNSMOS
        primary = Path(options.dnsmos_dir) / 'sig_bak_ovr.onnx'
        p808 = Path(options.dnsmos_dir) / 'model_v8.onnx'
        asset('dnsmos_primary', primary); asset('dnsmos_p808', p808)
        model = DNSMOS(str(primary), str(p808), device=options.device)
        with ThreadPoolExecutor(max_workers=options.dnsmos_workers) as pool:
            scores = pool.map(lambda pair: model(str(pair[0]), 16000, False), pairs)
            for row, score in zip(rows, scores):
                row.update({COLUMNS[key]: float(score[COLUMNS[key]]) for key in requested if key.startswith('dnsmos_')})
        del model
        release_models(); save('dnsmos')
    if {'wer', 'cer'} & requested:
        from .asr import transcribe, error_rates
        references = reference_provider(options)
        metadata = [references.reference(str(prediction), str(reference)) for prediction, reference in pairs]
        paths = [str(pair[0]) for pair, info in zip(pairs, metadata) if info['wer_eligible']]
        asset('qwen_config', Path(options.qwen_model) / 'config.json')
        hypotheses = transcribe(paths, options.qwen_model, device=options.device, batch_size=options.asr_batch_size,
            max_new_tokens=options.asr_max_new_tokens, language=options.language)
        for row, pair, info in zip(rows, pairs, metadata):
            row.update(info)
            values = error_rates(info['reference_text'], hypotheses[str(pair[0])]) if info['wer_eligible'] else (float('nan'), float('nan'))
            row['hypothesis_text'] = hypotheses.get(str(pair[0]), '')
            for name, value in zip(['wer', 'cer'], values):
                if name in requested:
                    row[name] = value
        protocol['asr_eligible_pairs'] = len(paths)
        release_models(); save('asr')
    if 'speaker_similarity' in requested:
        from .wavlm import WavLMSimilarity
        asset('wavlm_config', Path(options.wavlm_model) / 'config.json')
        model = WavLMSimilarity(options.wavlm_model, options.device)
        for row, pair in zip(rows, pairs):
            row['sim'] = model(*pair)
        del model
        release_models(); save('wavlm')
    if 'wespeaker_similarity' in requested:
        from .wespeaker_similarity import WeSpeakerSimilarity
        model = WeSpeakerSimilarity(options.wespeaker_model, device=options.device,
            batch_size=options.wespeaker_batch_size, max_batch_frames=options.wespeaker_max_batch_frames)
        scores = model.score_pairs([str(pair[0]) for pair in pairs], [str(pair[1]) for pair in pairs], workers=options.wespeaker_workers)
        for row, score in zip(rows, scores):
            row['sim_wespeaker'] = score
        write_json(output / 'wespeaker_protocol.json', model.metadata())
        del model
        release_models(); save('wespeaker')
    report = dict(expected_items=len(pairs), rows=len(rows), available_pairs=available,
                  metrics={}, finite_counts={}, coverage={}, protocol=protocol['version'])
    for name in options.metrics:
        finite = [float(row[COLUMNS[name]]) for row in rows if math.isfinite(float(row[COLUMNS[name]]))]
        eligible = sum(row.get('wer_eligible', True) for row in rows) if name in {'wer', 'cer'} else len(rows)
        report['metrics'][name] = sum(finite) / len(finite) if finite else None
        report['finite_counts'][name] = len(finite)
        report['coverage'][name] = dict(eligible=eligible, finite=len(finite), failed=eligible-len(finite), skipped=len(rows)-eligible)
    report['elapsed_seconds'] = time.perf_counter() - started
    write_json(output / 'coverage.json', report)
    write_json(output / 'summary.json', report)
    write_json(output / 'COMPLETE.json', dict(processed_pairs=len(rows), available_pairs=available,
                                             all_available_pairs=len(rows)==available, metrics=options.metrics))
    return report
