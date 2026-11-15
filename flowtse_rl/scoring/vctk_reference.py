"""True transcripts for complete shorter VCTK2Mix targets.

Min-mode source WAVs are already cropped, so eligibility requires uncropped
VCTK audio, an utterance duration CSV, or the mixture duration/crop metadata.
"""

import csv
from functools import lru_cache
from pathlib import Path
import re

import soundfile as sf


_MIXTURE = re.compile(r'^(p\d+_\d+(?:_mic\d+)?)_(p\d+_\d+(?:_mic\d+)?)_\d+$')


def mixture_utterances(mixture_id):
    match = _MIXTURE.fullmatch(mixture_id)
    if match is None:
        raise ValueError(f'Unrecognized VCTK2Mix mixture ID: {mixture_id}')
    return match.groups()


def target_utterance(utterances, target):
    if target in ('s1', 's2'):
        return utterances[int(target[1]) - 1]
    matches = [utt for utt in utterances if target in (utt, utt.split('_')[0])]
    if len(matches) != 1:
        raise ValueError(f'Ambiguous or unknown VCTK target: {target}')
    return matches[0]


def prepare_vctk_manifest(mixture_csv, enrollment_csv, manifest, output_dir, include_noise=False):
    """Resolve either enrollment schema to the full target utterance ID."""
    with Path(mixture_csv).open(newline='', encoding='utf-8-sig') as handle:
        mixtures = {row['mixture_ID']: row for row in csv.DictReader(handle)}
    rows, outputs = [], set()
    with Path(enrollment_csv).open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            mixture_id = row.get('mixture_ID') or row['mixture_id']
            mixture = mixtures[mixture_id]
            utterances = mixture_utterances(mixture_id)
            target = row.get('target_source') or row.get('utterance_id') or row['target_speaker']
            utterance = target_utterance(utterances, target)
            speaker = row.get('target_speaker')
            if speaker and speaker != utterance.split('_')[0]:
                raise ValueError(f'Enrollment speaker does not match target for {mixture_id}')
            source = utterances.index(utterance) + 1
            output = str(Path(output_dir) / f'{mixture_id}__target-{utterance}_pred.wav')
            if output in outputs:
                raise ValueError(f'Duplicate VCTK target: {mixture_id} / {utterance}')
            outputs.add(output)
            sample = dict(mixture=mixture['mixture_path'],
                          enrollment=row.get('enrollment_path') or row['enr_path1'],
                          reference=mixture[f'source_{source}_path'], output=output)
            if include_noise:
                sample['noise'] = mixture['noise_path']
            rows.append(sample)
    fields = ['mixture', 'enrollment', 'reference', 'output']
    if include_noise:
        fields.insert(1, 'noise')
    with Path(manifest).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


class VCTKReferences:
    def __init__(self, root, original_root=None, durations_csv=None):
        self.root = Path(root)
        self.text_root = self.root / 'txt'
        if not self.text_root.is_dir():
            raise FileNotFoundError(f'Missing VCTK transcripts directory: {self.text_root}')
        if original_root:
            self.audio_roots = [Path(original_root)]
            if not self.audio_roots[0].is_dir():
                raise FileNotFoundError(f'Missing original VCTK audio directory: {original_root}')
        else:
            self.audio_roots = [self.root / name for name in
                                ('original', 'train_sr16k', 'wav48', 'wav48_silence_trimmed')
                                if (self.root / name).is_dir()]
        if not durations_csv:
            candidates = [self.root / 'wav16k/min/metadata/source_duration_and_crop_test.csv',
                          self.root / 'source_durations.csv',
                          self.root / 'wav16k/min/metadata/source_durations.csv']
            durations_csv = next((p for p in candidates if p.is_file()), None)
        self.durations = {}
        self.crop_ranges = {}
        self.durations_csv = Path(durations_csv) if durations_csv else None
        if durations_csv:
            self._load_durations(self.durations_csv)
        if not self.audio_roots and not self.durations:
            raise FileNotFoundError(
                'VCTK shorter-target WER needs uncropped original durations. '
                'Set VCTK_ORIGINAL_ROOT to the original VCTK audio directory, or '
                'VCTK_DURATIONS_CSV to source_duration_and_crop_test.csv or a CSV '
                'with utterance_id,frames,sample_rate. '
                'The cropped wav16k/min/test/s1 and s2 files cannot determine this.'
            )
        # A VCTK corpus root may contain the actual audio under these folders.
        self.audio_roots += [p / subdir for p in list(self.audio_roots)
                             for subdir in ('train_sr16k', 'wav48', 'wav48_silence_trimmed')
                             if (p / subdir).is_dir()]

    def _add_duration(self, utterance_id, frames, sample_rate):
        value = int(frames), int(sample_rate)
        if min(value) <= 0:
            raise ValueError(f'Invalid original duration for {utterance_id}')
        previous = self.durations.setdefault(utterance_id, value)
        if previous != value:
            raise ValueError(f'Conflicting original durations for {utterance_id}')

    def _load_durations(self, path):
        with path.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            if {'utterance_id', 'frames', 'sample_rate'} <= fields:
                for row in reader:
                    self._add_duration(row['utterance_id'], row['frames'], row['sample_rate'])
                return
            crop_fields = {'mixture_ID', 'mixture_sample_rate', 'mixture_samples'}
            crop_fields.update(f'source_{source}_{field}' for source in (1, 2) for field in
                               ('resampled_samples_16k', 'crop_start_sample_16k',
                                'crop_end_sample_16k_exclusive'))
            if not crop_fields <= fields:
                raise ValueError('VCTK duration CSV requires utterance_id, frames, sample_rate '
                                 'or the source_duration_and_crop_test.csv fields')
            for row in reader:
                mixture = row['mixture_ID']
                utterances = mixture_utterances(mixture)
                if (mixture, utterances[0]) in self.crop_ranges:
                    raise ValueError(f'Duplicate VCTK crop record: {mixture}')
                if int(row['mixture_sample_rate']) != 16000:
                    raise ValueError(f'VCTK crop metadata requires 16000 Hz: {mixture}')
                retained_frames = int(row['mixture_samples'])
                for source, utterance in enumerate(utterances, 1):
                    prefix = f'source_{source}_'
                    # Use the resampled originals that actually entered the mixer.
                    # Raw 48 kHz durations can differ by a fraction of a 16 kHz sample.
                    frames = int(row[prefix + 'resampled_samples_16k'])
                    start = int(row[prefix + 'crop_start_sample_16k'])
                    end = int(row[prefix + 'crop_end_sample_16k_exclusive'])
                    if not (0 <= start < end <= frames and end - start == retained_frames):
                        raise ValueError(f'Invalid VCTK crop range: {mixture} / {utterance}')
                    self._add_duration(utterance, frames, 16000)
                    self.crop_ranges[mixture, utterance] = start, end

    @lru_cache(maxsize=None)
    def duration(self, utterance_id):
        if utterance_id in self.durations:
            return self.durations[utterance_id]
        speaker = utterance_id.split('_')[0]
        names = [utterance_id]
        if '_mic' not in utterance_id:
            names.append(utterance_id + '_mic1')
        candidates = [folder / (name + suffix) for root in self.audio_roots
                      for folder in (root, root / speaker)
                      for name in names for suffix in ('.wav', '.flac')]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f'Missing original audio or duration for {utterance_id}')
        info = sf.info(path)
        return info.frames, info.samplerate

    @lru_cache(maxsize=None)
    def transcript(self, utterance_id):
        text_id = re.sub(r'_mic\d+$', '', utterance_id)
        path = self.text_root / text_id.split('_')[0] / (text_id + '.txt')
        text = path.read_text(encoding='utf-8-sig').strip()
        if not text:
            raise ValueError(f'Empty VCTK transcript: {path}')
        return text

    def reference(self, prediction, reference_audio):
        result = dict(wer_eligible=False, wer_skip_reason='', target_utt_id='', reference_text='')
        name = Path(prediction).name
        if not name.endswith('_pred.wav') or '__target-' not in name:
            result['wer_skip_reason'] = 'unrecognized_target_filename'
            return result
        mixture, target = name[:-len('_pred.wav')].rsplit('__target-', 1)
        try:
            utterances = mixture_utterances(mixture)
            target = target_utterance(utterances, target)
        except ValueError:
            result['wer_skip_reason'] = 'unrecognized_target_filename'
            return result
        result['target_utt_id'] = target
        try:
            target_frames, target_rate = self.duration(target)
            other_frames, other_rate = self.duration(utterances[1] if target == utterances[0] else utterances[0])
            if target_frames * other_rate > other_frames * target_rate:
                result['wer_skip_reason'] = 'longer_target'
                return result
            if self.crop_ranges:
                crop = self.crop_ranges.get((mixture, target))
                if crop is None:
                    raise FileNotFoundError(f'Missing VCTK crop record for {mixture}')
                if crop != (0, target_frames):
                    result['wer_skip_reason'] = 'truncated_target_reference'
                    return result
            info = sf.info(reference_audio)
            if info.frames * target_rate + info.samplerate < target_frames * info.samplerate:
                result['wer_skip_reason'] = 'truncated_target_reference'
                return result
            text = self.transcript(target)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            result['wer_skip_reason'] = f'missing_original_or_transcript: {exc}'
            return result
        result.update(wer_eligible=True, reference_text=text)
        return result
