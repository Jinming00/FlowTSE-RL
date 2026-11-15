"""Full Mel analysis, bounded Flow, whole-wave synthesis and metric coverage."""
import csv
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch.utils.data import DataLoader

from flowtse_rl.audio import BigVGANMel, mixture_gain
from flowtse_rl.engine import validate
from flowtse_rl.inference import extract_chunked, inference_settings, predict_chunked
from flowtse_rl.engine import ExactEvalSampler, FlowSystem
from flowtse_rl.data import collate_mixtures
from flowtse_rl import quality


class ToyFrontend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, waveform, lengths=None):
        self.inputs.append(waveform.clone())
        lengths = torch.full((len(waveform),), waveform.shape[-1], dtype=torch.long) if lengths is None else lengths
        return waveform[:, None], lengths


class ToyVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.extents = []

    def forward(self, state, time, enrollment, mixture, lengths, elengths):
        self.extents.append(state.shape[-1])
        return mixture + self.anchor * 0


class ToyVocoder:
    def __init__(self, hop_length=1):
        self.calls = []
        self.hop_length = hop_length

    def __call__(self, mel, samples):
        self.calls.append((mel.clone(), samples.clone()))
        # A noncausal decoder must see the neighboring Mel frame across each
        # Flow boundary. Per-chunk decoding would zero-pad at every join.
        wave = torch.nn.functional.conv1d(mel[:, :1], mel.new_ones(1, 1, 3), padding=1)[:, 0]
        wave = wave.repeat_interleave(self.hop_length, dim=-1)
        return [wave[index, :int(count)] for index, count in enumerate(samples)]


def module():
    result = FlowSystem.__new__(FlowSystem)
    torch.nn.Module.__init__(result)
    result.config = {"seed": 42, "audio": {"sample_rate": 10, "hop_length": 1, "target_rms": 0.1},
                     "data": {"enrollment_seconds": [3.0, 3.0]},
                     "flow": {"steps": 16, "solver": "euler"}, "inference": {"chunk_seconds": 3.0},
                     "validation": {"run_external_metrics": True}}
    result.model, result.frontend, result.vocoder = ToyVelocity(), ToyFrontend(), ToyVocoder()
    return result.eval()


class ChunkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.module = module()
        self.mix = torch.linspace(-0.2, 0.3, 67)
        self.enroll = torch.linspace(-0.1, 0.2, 30)

    def test_exact_three_second_chunks_sixteen_steps_and_partial_tail(self):
        result = extract_chunked(self.module, self.mix, self.enroll)
        self.assertEqual(result.shape, self.mix.shape)
        self.assertEqual(self.module.model.extents, [30] * 32 + [7] * 16)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual([value.shape[-1] for value in self.module.frontend.inputs], [30, 67])
        self.assertEqual(len(self.module.vocoder.calls), 1)
        self.assertEqual(self.module.vocoder.calls[0][0].shape[-1], 67)
        self.assertEqual(self.module.vocoder.calls[0][1].tolist(), [67])

    def test_global_gain_full_mel_slices_and_decoder_context_at_joins(self):
        prediction = predict_chunked(self.module, self.mix, self.enroll, seed=7)
        gain = mixture_gain(self.mix, 0.1)
        torch.testing.assert_close(prediction.gain, gain, rtol=0, atol=0)
        self.assertEqual([(chunk.frame_start, chunk.frame_stop) for chunk in prediction.chunks],
                         [(0, 30), (30, 60), (60, 67)])
        torch.testing.assert_close(torch.cat([chunk.mixture_mel for chunk in prediction.chunks], dim=-1),
                                   (self.mix * gain)[None, None], rtol=0, atol=0)
        generated = prediction.generated_mel
        expected = torch.nn.functional.conv1d(generated, generated.new_ones(1, 1, 3), padding=1)[0, 0] / gain
        torch.testing.assert_close(prediction.waveform, expected, rtol=0, atol=0)
        separated = torch.cat([torch.nn.functional.conv1d(chunk.generated_mel,
            generated.new_ones(1, 1, 3), padding=1)[0, 0] for chunk in prediction.chunks]) / gain
        self.assertFalse(torch.allclose(prediction.waveform[29:31], separated[29:31]))

    def test_real_frontend_282_frame_chunks_preserve_non_aligned_length(self):
        audio = dict(sample_rate=24000, n_fft=1024, win_length=1024,
                     hop_length=256, n_mels=100, target_rms=0.1)
        self.module.config['audio'] = audio
        self.module.frontend = BigVGANMel(**audio)
        self.module.vocoder = ToyVocoder(hop_length=256)
        torch.manual_seed(9)
        mixture, enrollment = torch.randn(24000 * 7 + 1) * .03, torch.randn(72000) * .03
        prediction = predict_chunked(self.module, mixture, enrollment)
        full_mel, lengths = self.module.frontend((mixture * prediction.gain)[None])
        assembled = torch.cat([chunk.mixture_mel for chunk in prediction.chunks], dim=-1)
        torch.testing.assert_close(assembled, full_mel, rtol=0, atol=0)
        self.assertEqual(self.module.model.extents, [282] * 32 + [93] * 16)
        self.assertEqual(prediction.generated_mel.shape[-1], int(lengths[0]))
        self.assertEqual(len(prediction.waveform), len(mixture))
        self.assertEqual(len(self.module.vocoder.calls), 1)

    def test_real_frontend_short_and_single_sample_tails_keep_all_samples(self):
        self.module.config['audio'].update(sample_rate=24000, hop_length=256)
        self.module.frontend = BigVGANMel()
        self.module.vocoder = ToyVocoder(hop_length=256)
        for samples in (1, 72000, 72192, 72193):
            with self.subTest(samples=samples):
                self.module.model.extents.clear()
                self.module.vocoder.calls.clear()
                result = extract_chunked(self.module, torch.zeros(samples), torch.zeros(72000))
                self.assertEqual(result.numel(), samples)
                self.assertTrue(torch.isfinite(result).all())
                self.assertLessEqual(max(self.module.model.extents), 282)
                self.assertEqual(len(self.module.vocoder.calls), 1)

    def test_one_sample_tail_and_silence_are_finite(self):
        result = extract_chunked(self.module, torch.zeros(31), self.enroll)
        self.assertEqual(len(result), 31)
        self.assertEqual(self.module.model.extents, [30] * 16 + [1] * 16)
        self.assertTrue(torch.isfinite(result).all())


    def test_native_validation_scores_one_full_wave_and_slices_one_full_target_mel(self):
        item = dict(mixture=self.mix, target=self.mix * .5, enrollment=self.enroll,
                    record_index=0, mixture_id='mix', utterance_id='target')
        loader = DataLoader([item], batch_size=1, collate_fn=collate_mixtures)
        accelerator = SimpleNamespace(device=torch.device('cpu'), autocast=nullcontext, is_main_process=True)
        expected = extract_chunked(self.module, self.mix, self.enroll)
        self.module.frontend.inputs.clear()
        self.module.vocoder.calls.clear()
        report = dict(metrics={'pesq': 2.8}, finite_counts={'pesq': 1})
        with tempfile.TemporaryDirectory() as folder:
            self.module.config['checkpoint'] = dict(dir=folder)
            with mock.patch('flowtse_rl.engine.save_quality_pair', return_value='full_ref.wav') as save, \
                 mock.patch('flowtse_rl.engine.score_quality', return_value=report) as scorer, \
                 mock.patch('flowtse_rl.engine.flow_matching_loss', return_value=torch.tensor(0.)) as loss:
                metrics, _, _ = validate(self.module, loader, self.module.config, 9, accelerator)
        torch.testing.assert_close(save.call_args.args[2], expected, rtol=0, atol=0)
        self.assertEqual([value.shape[-1] for value in self.module.frontend.inputs], [30, 67, 67])
        full_target = self.module.frontend.inputs[-1][:, None]
        for call, (start, stop) in zip(loss.call_args_list, [(0, 30), (30, 60), (60, 67)]):
            torch.testing.assert_close(call.args[1], full_target[..., start:stop], rtol=0, atol=0)
        self.assertEqual(loss.call_count, 3)
        self.assertEqual(len(self.module.vocoder.calls), 1)
        self.assertEqual(scorer.call_args.kwargs['expected_filenames'], ['full_ref.wav'])
        self.assertEqual(metrics['val/pesq'], 2.8)

    def test_invalid_chunk_and_step_settings_are_rejected(self):
        for seconds in [0, -1, float("nan")]:
            with self.assertRaises(ValueError):
                inference_settings(self.module.config, chunk_seconds=seconds)
        with self.assertRaises(ValueError):
            inference_settings(self.module.config, steps=0)

    def test_fixed_prefix_ratios_and_nonduplicated_ddp_shards(self):
        self.assertEqual(quality.validation_size(6000, {"loss_ratio": 0.2}), 1200)
        self.assertEqual(quality.validation_size(6000, {"loss_ratio": 0.2, "num_loss_samples": 64}), 1200)
        self.assertEqual(quality.quality_size(1200, {"quality_metrics_ratio": 1.0}), 1200)
        shards = [list(ExactEvalSampler(range(7), rank, 4)) for rank in range(4)]
        self.assertEqual(sorted(value for shard in shards for value in shard), list(range(7)))





class QualityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="flowtse-quality-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.names = []
        for index in range(2):
            item = {"mixture_id": f"mix{index}", "utterance_id": f"target{index}"}
            self.names.append(quality.save_quality_pair(self.root / "audio", item,
                                                       torch.zeros(100), torch.zeros(100), 24000))

    def backend(self, command, env, check, stdout=None, stderr=None):
        self.assertEqual(env["MAX_ITEMS"], "2")
        directory = Path(env["OUT_DIR"])
        directory.mkdir(exist_ok=True)
        if stdout is not None:
            stdout.write('Detailed backend output preserved\n')
        with (directory / "_results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["filename"] + list(quality.QUALITY_COLUMNS.values()))
            writer.writeheader()
            for index, name in enumerate(self.names):
                row = {column: 1.0 + index for column in quality.QUALITY_COLUMNS.values()}
                row.update(filename=name, pesq=float("nan") if index == 0 else 2.0)
                writer.writerow(row)

    def test_identical_backend_preserves_coverage_and_retains_requested_audio(self):
        with mock.patch.object(quality.subprocess, "run", side_effect=self.backend):
            report = quality.score_quality(self.root, {"quality_audio_keep_items": 1, "metrics_command": ["custom-scorer"]}, 2,
                                           expected_filenames=self.names)
        self.assertEqual(report["finite_counts"]["pesq"], 1)
        self.assertEqual(report["metrics"]["pesq"], 2.0)
        self.assertEqual(report["finite_counts"]["dnsmos_p808"], 2)
        self.assertTrue((self.root / "metrics/coverage.json").is_file())
        self.assertIn('Detailed backend', (self.root / 'metrics/backend.log').read_text())
        self.assertEqual(len(list((self.root / "audio").glob("*_pred.wav"))), 1)
        self.assertEqual(len(list((self.root / "audio").glob("*_ref.wav"))), 1)

    def test_missing_pairs_are_rejected_before_backend(self):
        (self.root / "audio" / self.names[0]).unlink()
        with mock.patch.object(quality.subprocess, "run") as scorer:
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                quality.score_quality(self.root, {"metrics_command": ["custom-scorer"]}, 2)
            scorer.assert_not_called()

    def test_wrong_sample_ids_are_rejected_before_backend(self):
        with mock.patch.object(quality.subprocess, "run") as scorer:
            with self.assertRaisesRegex(RuntimeError, "IDs"):
                quality.score_quality(self.root, {}, 2, expected_filenames=["wrong", "samples"])
            scorer.assert_not_called()

    def test_backend_failure_keeps_full_log_and_audio(self):
        def fail(command, **kwargs):
            kwargs['stdout'].write('Actual backend error detail\n')
            raise quality.subprocess.CalledProcessError(7, command)
        with mock.patch.object(quality.subprocess, 'run', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'Quality backend failed; output:'):
                quality.score_quality(self.root, {"metrics_command": ["custom-scorer"]}, 2)
        self.assertIn('Actual backend error', (self.root / 'metrics/backend.log').read_text())
        self.assertEqual(len(list((self.root / 'audio').glob('*_pred.wav'))), 2)

if __name__ == "__main__":
    unittest.main()
