"""True-text ASR references for complete shorter LibriMix target utterances."""
from functools import lru_cache
from pathlib import Path
import re
import soundfile as sf

_UTTERANCE = re.compile(r'^\d+-\d+-\d+$')


class LibriSpeechReferences:
    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f'Missing LibriSpeech root: {self.root}')
        self.splits = sorted(p for p in self.root.iterdir() if p.is_dir())

    @lru_cache(maxsize=None)
    def utterance(self, utterance_id):
        speaker, chapter, _ = utterance_id.split('-')
        candidates = [split / speaker / chapter / (utterance_id + suffix)
                      for split in self.splits for suffix in ('.flac', '.wav')]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f'Missing original audio for {utterance_id}')
        info = sf.info(path)
        text = self.chapter_texts(path.parent / f'{speaker}-{chapter}.trans.txt').get(utterance_id)
        if not text:
            raise ValueError(f'Missing transcript for {utterance_id}')
        return info.frames, info.samplerate, text

    @lru_cache(maxsize=None)
    def chapter_texts(self, path):
        result = {}
        with path.open(encoding='utf-8') as source:
            for line in source:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    result[parts[0]] = parts[1]
        return result

    def reference(self, prediction, reference_audio):
        result = dict(wer_eligible=False, wer_skip_reason='', target_utt_id='', reference_text='')
        name = Path(prediction).name
        if not name.endswith('_pred.wav') or '__target-' not in name:
            result['wer_skip_reason'] = 'unrecognized_target_filename'
            return result
        mixture, target = name[:-len('_pred.wav')].rsplit('__target-', 1)
        utterances = mixture.split('_')
        if len(utterances) != 2 or not all(_UTTERANCE.fullmatch(u) for u in utterances) or target not in utterances:
            result['wer_skip_reason'] = 'unrecognized_target_filename'
            return result
        result['target_utt_id'] = target
        try:
            first, second = [self.utterance(u) for u in utterances]
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            result['wer_skip_reason'] = f'missing_original_or_transcript: {exc}'
            return result
        target_data, other = (first, second) if target == utterances[0] else (second, first)
        frames, rate, text = target_data
        # Integer cross-products compare original durations without rounding.
        if frames * other[1] > other[0] * rate:
            result['wer_skip_reason'] = 'longer_target'
            return result
        # Eligibility depends only on the reference, never on model output.
        # A prediction that omits the end must incur ASR errors, not be excluded.
        info = sf.info(reference_audio)
        if info.frames * rate + info.samplerate < frames * info.samplerate:
            result['wer_skip_reason'] = 'truncated_target_reference'
            return result
        result.update(wer_eligible=True, reference_text=text)
        return result
