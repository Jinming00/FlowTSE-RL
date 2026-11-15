"""Complex transform parity, exact phase/length reconstruction and native training IO."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from flowtse_rl.audio import ComplexISTFT, ComplexSTFT
from flowtse_rl.config import BASELINE_CONFIG, load_config, validate_config
from flowtse_rl.data import collate_mixtures
from flowtse_rl.engine import FlowSystem, validate
from flowtse_rl.inference import predict_chunked, inference_settings
from flowtse_rl.representation import frame_count
from flowtse_rl.training import run_training
from infer import load_module

ROOT = Path(__file__).resolve().parents[1]


def complex_fixture():
    """Small transform-test setup, independent of distributed training recipes."""
    config = load_config(BASELINE_CONFIG)
    config['audio'] = dict(representation='complex_stft', sample_rate=16000,
                           n_fft=510, win_length=510, hop_length=128, target_rms=.1)
    config['model'].pop('mel_dim', None)
    config['model']['feature_dim'] = 512
    config['vocoder'] = dict(type='istft')
    config['inference']['protocol'] = 'full_stft_flow_chunks_single_istft_v1'
    return config


def settings(root):
    config = complex_fixture()
    config['model'].update(hidden_size=32, depth=2, num_heads=4, mlp_ratio=2, compile_blocks=False)
    config['train'].update(num_epochs=1, max_steps=-1, precision='32-true', wandb=False)
    config['ddp'].update(num_gpus=1, use_ddp=False)
    config['checkpoint'].update(dir=str(root), resume=None)
    config['validation'].update(run_external_metrics=False, metrics=['si_sdr'])
    return config


class OracleVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, state, time, enrollment, mixture, lengths, elengths):
        return (mixture - state) / (1 - time[:, None, None]) + self.anchor * 0


class RecordingFrontend(ComplexSTFT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def forward(self, waveform, lengths=None):
        self.calls.append(waveform.shape[-1])
        return super().forward(waveform, lengths)


class RecordingDecoder(ComplexISTFT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def forward(self, features, samples):
        self.calls.append(features.shape[-1])
        return super().forward(features, samples)


class ComplexSTFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_v0_transform_and_three_second_roundtrip(self):
        wave = torch.randn(2, 48000) * .1
        actual, counts = ComplexSTFT()(wave)
        spectrum = torch.stft(wave, n_fft=510, win_length=510, hop_length=128,
                              window=torch.hann_window(510), return_complex=True)
        expected = torch.cat([spectrum.real, spectrum.imag], dim=1)
        self.assertEqual(actual.shape, (2, 512, 376))
        self.assertEqual(counts.tolist(), [376, 376])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        decoded = torch.stack(ComplexISTFT()(actual, [48000, 48000]))
        torch.testing.assert_close(decoded, wave, rtol=2e-5, atol=2e-6)

    def test_ragged_nan_padding_and_short_signal_lengths(self):
        samples = [1, 255, 48129]
        wave = torch.randn(3, max(samples)) * .1
        for i, count in enumerate(samples):
            wave[i, count:] = float('nan')
        frontend = ComplexSTFT()
        features, counts = frontend(wave, torch.tensor(samples))
        self.assertEqual(counts.tolist(), [n // 128 + 1 for n in samples])
        self.assertTrue(torch.isfinite(features).all())
        for i, count in enumerate(samples):
            torch.testing.assert_close(features[i:i+1, :, :counts[i]], frontend.analyze(wave[i:i+1, :count]))
            features[i, :, counts[i]:] = float('nan')
        decoded = ComplexISTFT()(features, samples)
        self.assertEqual([len(x) for x in decoded], samples)
        for i, count in enumerate(samples):
            torch.testing.assert_close(decoded[i], wave[i, :count], rtol=2e-5, atol=2e-6)

    def test_fft_and_inverse_stay_float32_under_bf16(self):
        wave = torch.randn(1, 4097) * .1
        with torch.autocast('cpu', dtype=torch.bfloat16):
            features, _ = ComplexSTFT()(wave)
            output = ComplexISTFT()(features, [4097])[0]
        self.assertEqual(features.dtype, torch.float32)
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, wave[0], rtol=2e-5, atol=2e-6)

    def test_frontend_config_dimensions_and_decoder_protocol(self):
        for tag, dimensions, frames in [('16k', 512, 376), ('24k_control', 1026, 282)]:
            config = complex_fixture()
            if tag == '24k_control':
                # Frontend compatibility remains tested without a retired recipe.
                config['audio'].update(sample_rate=24000,n_fft=1024,win_length=1024,hop_length=256)
                config['model']['feature_dim']=1026
            validate_config(config)
            self.assertEqual(config['model']['feature_dim'], dimensions)
            self.assertEqual(frame_count(config, 3 * config['audio']['sample_rate']), frames)
        config = settings('/tmp/unused')
        config['vocoder']['checkpoint_dir'] = '/tmp/unwanted-bigvgan'
        with self.assertRaisesRegex(ValueError, 'must not configure'):
            validate_config(config)
        config['vocoder'].clear()
        config['model']['feature_dim'] = 100
        with self.assertRaisesRegex(ValueError, 'dimension'):
            validate_config(config)
        config['model']['feature_dim'] = 512
        config['inference']['protocol'] = 'full_mel_flow_chunks_single_bigvgan_v1'
        with self.assertRaisesRegex(ValueError, 'protocol'):
            inference_settings(config)

    def test_all_interfaces_backward_without_bigvgan(self):
        for mode, attention in [('channel_concat', 'bidirectional'), ('time_concat', 'bidirectional'),
                                ('flowtse', 'bidirectional'), ('flowtse', 'enrollment_readonly')]:
            with self.subTest(mode=mode, attention=attention):
                config = settings('/tmp/unused')
                config['model'].update(conditioning_mode=mode, attention_mode=attention)
                with patch('flowtse_rl.engine.FrozenBigVGAN', side_effect=AssertionError('Unexpected BigVGAN')):
                    system = FlowSystem(config)
                with torch.no_grad():
                    system.model.final_layer.linear.weight.normal_(std=.02)
                wave, lengths = torch.randn(2, 1024) * .1, torch.tensor([1024, 901])
                batch = dict(mixture=wave, target=wave*.7, enrollment=wave[:, :601],
                    mixture_lengths=lengths, target_lengths=lengths, enrollment_lengths=torch.tensor([601, 601]))
                loss = system(batch)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in system.model.parameters()))
                self.assertTrue(all(key.startswith('model.') for key in system.state_dict()))

    def oracle_system(self, root):
        system = FlowSystem(settings(root))
        system.model = OracleVelocity()
        system.frontend = RecordingFrontend(**system.config['audio'])
        system.vocoder = RecordingDecoder(**system.config['audio'])
        return system

    def test_full_stft_chunks_single_istft_including_one_frame_tail(self):
        system = self.oracle_system('/tmp/unused').eval()
        wave, enroll = torch.randn(96257)*.05, torch.randn(48000)*.05
        prediction = predict_chunked(system, wave, enroll, seed=13, steps=4)
        self.assertEqual([(c.frame_start, c.frame_stop) for c in prediction.chunks], [(0,376), (376,752), (752,753)])
        self.assertEqual(system.frontend.calls, [48000, 96257])
        self.assertEqual(system.vocoder.calls, [753])
        self.assertEqual(len(prediction.waveform), len(wave))
        torch.testing.assert_close(prediction.waveform, wave, rtol=3e-5, atol=3e-6)

    def test_validation_whole_target_stft_and_correct_metric_names(self):
        with tempfile.TemporaryDirectory() as root:
            system = self.oracle_system(root)
            wave = torch.randn(96257)*.05
            item = dict(mixture=wave, target=wave, enrollment=torch.randn(48000)*.05,
                record_index=0, mixture_id='mix', utterance_id='target')
            loader = DataLoader([item], batch_size=1, collate_fn=collate_mixtures)
            accelerator = SimpleNamespace(device=torch.device('cpu'), is_main_process=True,
                autocast=lambda: __import__('contextlib').nullcontext())
            metrics, _, _ = validate(system, loader, system.config, 0, accelerator)
            self.assertEqual(system.frontend.calls, [48000, 96257, 96257])
            self.assertEqual(system.vocoder.calls, [753])
            self.assertIn('val/generated_stft_ri_l1', metrics)
            self.assertNotIn('val/generated_mel_l1', metrics)
            self.assertEqual(metrics['diagnostic/val_si_sdr_count'], 1)
            self.assertGreater(metrics['val/si_sdr'], 70)

    def test_native_train_checkpoint_reload_and_inference(self):
        with tempfile.TemporaryDirectory() as root:
            config, items = settings(root), []
            for i in range(4):
                wave = torch.randn(1024)*.05
                items.append(dict(mixture=wave, target=wave*.6, enrollment=torch.randn(601)*.05,
                    record_index=i, mixture_id=f'mix{i}', utterance_id=f'utt{i}'))
            loader = DataLoader(items, batch_size=2, collate_fn=collate_mixtures)
            accelerator = Accelerator(cpu=True, mixed_precision='no', step_scheduler_with_optimizer=False)
            result = run_training(config, system=FlowSystem(config), loaders=(loader,None), accelerator=accelerator)
            self.assertEqual(result['global_step'], 2)
            with patch('flowtse_rl.engine.FrozenBigVGAN', side_effect=AssertionError('Unexpected BigVGAN')):
                reloaded = load_module(str(Path(root)/'checkpoints/last.ckpt'), 'cpu')
            prediction = predict_chunked(reloaded, items[0]['mixture'], items[0]['enrollment'], steps=2)
            self.assertEqual(prediction.waveform.numel(), 1024)
            self.assertTrue(torch.isfinite(prediction.waveform).all())
