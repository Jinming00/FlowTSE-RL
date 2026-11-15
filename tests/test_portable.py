"""Exercise the distribution without a parent checkout or external storage tools."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
import torch
import yaml

from flowtse_rl.config import load_config
from flowtse_rl.data import resolve_audio_path
from flowtse_rl.quality import save_quality_pair, score_quality
from flowtse_rl.utils.wandb import create_native_run
from test_complex_stft import complex_fixture

ROOT = Path(__file__).resolve().parents[1]


class PortableTests(unittest.TestCase):
    def test_csv_paths_and_boundary_safe_prefix_relocation(self):
        self.assertEqual(resolve_audio_path('audio/a.wav', '/dataset'), '/dataset/audio/a.wav')
        self.assertEqual(resolve_audio_path('/old/data/a.wav', '/csv', {'/old/data': '/new'}), '/new/a.wav')
        self.assertEqual(resolve_audio_path('/old/database/a.wav', '/csv', {'/old/data': '/new'}), '/old/database/a.wav')

    def test_wandb_offline_identity_and_epoch_axis(self):
        with tempfile.TemporaryDirectory() as folder:
            sdk = Mock()
            config = {'checkpoint': {'dir': folder}, 'train': {'wandb_project': 'test'}}
            with patch.dict('sys.modules', {'wandb': sdk}), patch.dict(os.environ, WANDB_MODE='offline'):
                create_native_run(config)
                first = sdk.init.call_args.kwargs['id']
                create_native_run(config)
                self.assertEqual(first, sdk.init.call_args.kwargs['id'])
                self.assertEqual(sdk.init.call_args.kwargs['mode'], 'offline')
                self.assertIsNone(sdk.init.call_args.kwargs['resume'])
                sdk.init.return_value.define_metric.assert_any_call('*', step_metric='epoch', step_sync=True)

    def test_builtin_quality_pairs_coverage_and_retention(self):
        with tempfile.TemporaryDirectory() as folder:
            wave = torch.linspace(-.2, .2, 1600)
            filename = save_quality_pair(Path(folder) / 'audio', {'mixture_id': 'a', 'utterance_id': 'b'},
                                         wave, wave + torch.randn_like(wave) * .01, 16000)
            report = score_quality(folder, {'metrics': ['si_sdr'], 'quality_audio_keep_items': 1}, 1,
                                   expected_filenames=[filename])
            self.assertEqual(report['finite_counts'], {'si_sdr': 1})
            self.assertTrue((Path(folder) / 'audio' / filename).exists())
            with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
                score_quality(folder, {'metrics': ['si_sdr']}, 2)

    def test_real_cli_train_resume_infer_and_evaluate_from_another_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            split = root / 'train-100'
            metadata = root / 'metadata'
            split.mkdir(); metadata.mkdir()
            time = np.arange(3200, dtype=np.float32) / 16000
            target = np.sin(time * (2 * np.pi * 230)).astype(np.float32) * .1
            interferer = np.sin(time * (2 * np.pi * 370)).astype(np.float32) * .08
            for name, wave in [('mix', target + interferer), ('s1', target), ('s2', interferer), ('enroll', target)]:
                sf.write(root / f'{name}.wav', wave, 16000, subtype='FLOAT')
            with (metadata / 'mixture_train-100_mix_both.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['mixture_ID', 'mixture_path', 'source_1_path', 'source_2_path', 'length'])
                writer.writerow(['1-1-1_2-1-1', '../mix.wav', '../s1.wav', '../s2.wav', 3200])
            with (split / 'mixture2enrollment.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['mixture_id', 'utterance_id', 'path', 'length'])
                writer.writerow(['1-1-1_2-1-1', '1-1-1', '../enroll.wav', 3200])
            config = complex_fixture()
            config['model'].update(hidden_size=32, depth=2, num_heads=4, mlp_ratio=2, compile_blocks=False)
            config['data'].update(train_dirs=[str(split)], validation_dir=str(split), test_dir=str(split),
                                  metadata_dir=str(metadata), num_workers=0, segment_seconds=.06,
                                  enrollment_seconds=[.06, .06], max_samples_per_batch=1)
            config['train'].update(num_epochs=1, max_steps=-1, precision='32-true', wandb=False)
            config['optimizer']['warmup_epochs'] = 0
            config['validation'].update(every_n_epochs=1, metrics=['si_sdr'])
            config['checkpoint'].update(dir=str(root / 'run'), monitor='val/si_sdr')
            config['ddp']['num_gpus'] = 1
            config['flow']['steps'] = 1
            config['inference']['chunk_seconds'] = .06
            settings = root / 'config.yaml'
            settings.write_text(yaml.safe_dump(config))
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1')
            def run(script, *args):
                result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], cwd=root,
                                        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout[-8000:])
            run('train.py', '--cpu', '--config', settings)
            checkpoint = root / 'run/checkpoints/last.ckpt'
            before = torch.load(checkpoint, map_location='cpu', weights_only=False)
            self.assertEqual(before['global_step'], 1)
            config['train']['num_epochs'] = 2
            settings.write_text(yaml.safe_dump(config))
            run('train.py', '--cpu', '--config', settings)
            after = torch.load(checkpoint, map_location='cpu', weights_only=False)
            self.assertEqual(after['global_step'], 2)
            run('infer.py', '--checkpoint', checkpoint, '--device', 'cpu', '--mixture', root / 'mix.wav',
                '--enrollment', root / 'enroll.wav', '--output', root / 'pred.wav')
            predicted, rate = sf.read(root / 'pred.wav')
            self.assertEqual((len(predicted), rate), (3200, 16000))
            self.assertTrue(np.isfinite(predicted).all())
            run('evaluate.py', '--checkpoint', checkpoint, '--device', 'cpu', '--output-dir', root / 'eval',
                '--scoring-metrics', 'si_sdr', '--workers', '1')
            report = json.loads((root / 'eval/summary.json').read_text())
            self.assertEqual(report['processed_pairs'], 1)
            self.assertEqual(report['metrics']['si_sdr']['count'], 1)
            self.assertEqual(report['scoring_protocol'], 'flowtse_full_utterance_scoring_v1')
            self.assertTrue((root / 'eval/COMPLETE.json').is_file())


if __name__ == '__main__':
    unittest.main()
