"""Scoring coverage, real signal metrics, true-text eligibility and backend IO."""
import csv
import json
import importlib.util
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock

import numpy as np
import soundfile as sf

from flowtse_rl.scoring.runner import Options, expand_metrics, score_directory, find_pairs
from flowtse_rl.scoring.librispeech_reference import LibriSpeechReferences


class ScoringTests(unittest.TestCase):
    def make_pairs(self, root, count=2):
        audio = root / 'audio'
        audio.mkdir()
        rng = np.random.default_rng(21)
        for i in range(count):
            wave = rng.normal(0, .05, 16000).astype('float32')
            stem = f'{i}__target-a'
            sf.write(audio / f'{stem}_ref.wav', wave, 16000, subtype='FLOAT')
            sf.write(audio / f'{stem}_pred.wav', wave + rng.normal(0, .005, len(wave)), 16000, subtype='FLOAT')
        return audio

    def test_core_coverage_and_selected_subset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = self.make_pairs(root)
            result = score_directory(audio, root / 'scores', Options(metrics=['si_sdr'], workers=1), limit=1)
            self.assertEqual(result['rows'], 1)
            self.assertEqual(result['available_pairs'], 2)
            self.assertEqual(result['finite_counts'], {'si_sdr': 1})
            self.assertEqual(result['coverage']['si_sdr'], dict(eligible=1, finite=1, failed=0, skipped=0))
            self.assertGreater(result['metrics']['si_sdr'], 15)
            complete = json.loads((root / 'scores/COMPLETE.json').read_text())
            self.assertFalse(complete['all_available_pairs'])
            with self.assertRaises(FileExistsError):
                score_directory(audio, root / 'scores', Options(metrics=['si_sdr'], workers=1))

    def test_orphan_and_length_mismatch_rejected_without_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = self.make_pairs(Path(folder), 1)
            prediction = next(audio.glob('*_pred.wav'))
            reference = next(audio.glob('*_ref.wav'))
            sf.write(prediction, np.zeros(15999), 16000)
            with self.assertRaisesRegex(ValueError, 'refusing truncation'):
                find_pairs(audio)
            reference.unlink()
            with self.assertRaisesRegex(ValueError, 'exactly paired'):
                find_pairs(audio)

    def test_full_selection_is_eleven_distinct_metrics(self):
        self.assertEqual(len(expand_metrics('full,core,dnsmos')), 11)
        self.assertEqual(expand_metrics(['asr']), ['wer', 'cer'])
        with self.assertRaises(ValueError):
            Options(metrics=['invented'])

    @unittest.skipUnless(all(importlib.util.find_spec(name) is not None for name in
                            ['jiwer', 'whisper_normalizer', 'qwen_asr']), 'Install scoring ASR extras')
    def test_asr_eligibility_counts_are_not_failures(self):
        from flowtse_rl.scoring import asr
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); audio = self.make_pairs(root)
            names = [path.name for path in sorted(audio.glob('*_ref.wav'))]
            text = root / 'transcripts.json'
            text.write_text(json.dumps({names[0]: 'Hello world', names[1]: {
                'reference_text': '', 'wer_eligible': False, 'wer_skip_reason': 'longer_target'}}))
            model = root / 'qwen'; model.mkdir(); (model / 'config.json').write_text('{}')
            options = Options(metrics=['asr'], qwen_model=str(model), wer_reference='transcripts', transcripts_json=str(text))
            expected = str(next(audio.glob('0*_pred.wav')))
            with patch.object(asr, 'transcribe', return_value={expected: 'hello there'}) as recognize:
                result = score_directory(audio, root / 'scores', options)
            self.assertEqual(recognize.call_args.args[0], [expected])
            self.assertEqual(result['metrics']['wer'], .5)
            self.assertEqual(result['coverage']['wer'], dict(eligible=1, finite=1, failed=0, skipped=1))
            with (root / 'scores/_results.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[1]['wer_skip_reason'], 'longer_target')
            self.assertEqual(rows[1]['wer'], 'nan')

    def test_true_librimix_reference_uses_clean_duration_not_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            chapter = root / 'test-clean/1/2'; chapter.mkdir(parents=True)
            for utterance, length in [('1-2-1', 1600), ('1-2-2', 3200)]:
                sf.write(chapter / f'{utterance}.wav', np.ones(length) * .1, 16000)
            (chapter / '1-2.trans.txt').write_text('1-2-1 HELLO WORLD\n1-2-2 LONGER SENTENCE\n')
            reference = root / 'reference.wav'
            sf.write(reference, np.ones(1600) * .1, 16000)
            provider = LibriSpeechReferences(root)
            info = provider.reference('1-2-1_1-2-2__target-1-2-1_pred.wav', reference)
            self.assertTrue(info['wer_eligible'])
            self.assertEqual(info['reference_text'], 'HELLO WORLD')
            longer = provider.reference('1-2-1_1-2-2__target-1-2-2_pred.wav', reference)
            self.assertEqual(longer['wer_skip_reason'], 'longer_target')
            sf.write(reference, np.ones(800) * .1, 16000)
            truncated = provider.reference('1-2-1_1-2-2__target-1-2-1_pred.wav', reference)
            self.assertEqual(truncated['wer_skip_reason'], 'truncated_target_reference')

    def test_bundled_adapter_uses_isolated_cpu_workers_and_preserves_coverage(self):
        from flowtse_rl.quality import score_quality
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); audio = self.make_pairs(root)
            names = [path.name for path in audio.glob('*_ref.wav')]
            settings = dict(scoring_options=dict(metrics=['si_sdr'], workers=2), quality_audio_keep_items=2)
            with patch.dict(os.environ, FLOWTSE_SCORING_PYTHON=sys.executable):
                report = score_quality(root, settings, 2, expected_filenames=names)
            self.assertEqual(report['finite_counts']['si_sdr'], 2)
            self.assertEqual(report['coverage']['si_sdr']['failed'], 0)
            self.assertEqual(report['protocol'], 'flowtse_full_utterance_scoring_v1')
            self.assertTrue((root / 'metrics/COMPLETE.json').exists())

    @unittest.skipUnless(importlib.util.find_spec('whisper_normalizer') is not None, 'Install scoring extras')
    def test_qwen_adapter_preserves_order_and_uses_local_weights(self):
        from flowtse_rl.scoring import asr
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'config.json').write_text('{}')
            package = Mock()
            factory = package.Qwen3ASRModel.from_pretrained
            factory.return_value.transcribe.side_effect = [
                [{'text': 'Hello world!'}, {'transcription': 'How are you?'}], ['Goodbye.']]
            with patch.dict('sys.modules', qwen_asr=package), patch.object(asr, 'configure_audio_only_transformers'):
                result = asr.transcribe(['first.wav', 'second.wav', 'third.wav'], folder,
                                        device='cpu', batch_size=2)
            self.assertEqual(list(result), ['first.wav', 'second.wav', 'third.wav'])
            self.assertEqual(list(result.values()), ['hello world', 'how are you', 'goodbye'])
            self.assertTrue(factory.call_args.kwargs['local_files_only'])


if __name__ == '__main__':
    unittest.main()
