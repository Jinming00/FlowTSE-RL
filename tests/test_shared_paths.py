"""Dataset relocation and an early failure for still-packaged training audio."""
import csv
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flowtse_rl.config import BASELINE_CONFIG, load_config
from flowtse_rl.paths import SHARED_DATA_ROOT, resolve_data_paths, check_data_paths


class SharedPathTests(unittest.TestCase):
    def test_baseline_uses_shared_root_and_can_relocate_loaded_paths(self):
        with patch.dict(os.environ, {}, clear=True):
            config = load_config(BASELINE_CONFIG)
            self.assertEqual(config['data']['root'], str(SHARED_DATA_ROOT))
            self.assertTrue(config['data']['train_dirs'][1].startswith(str(SHARED_DATA_ROOT)))
            expected = str(Path('/work/data/LibriMix/Libri2Mix/wav16k/min/train-360'))
            resolve_data_paths(config, '/work/data')
            self.assertEqual(config['data']['train_dirs'][1], expected)
            self.assertEqual(config['data']['asr_reference_dir'], '/work/data/asr_references')

    def test_cli_root_takes_precedence_over_environment(self):
        with patch.dict(os.environ, FLOWTSE_DATA_ROOT='/environment/data'):
            self.assertEqual(load_config(BASELINE_CONFIG)['data']['root'], '/environment/data')
            self.assertEqual(load_config(BASELINE_CONFIG, data_root='/command/data')['data']['root'], '/command/data')

    def test_legacy_checkpoint_paths_move_when_root_is_explicit(self):
        config = {'data': {'train_dirs': ['/old/data/LibriMix/Libri2Mix/wav16k/min/train-100']}}
        resolve_data_paths(config, '/new/data')
        self.assertEqual(config['data']['train_dirs'], ['/new/data/LibriMix/Libri2Mix/wav16k/min/train-100'])

    def test_no_root_preserves_legacy_custom_paths(self):
        config = {'data': {'train_dirs': ['/custom/train'], 'metadata_dir': 'metadata'}}
        before = deepcopy(config)
        with patch.dict(os.environ, {}, clear=True):
            resolve_data_paths(config)
        self.assertEqual(config, before)

    def test_missing_packaged_audio_fails_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); split = root/'train-360'; split.mkdir(); metadata = root/'metadata'; metadata.mkdir()
            with (metadata/'mixture_train-360_mix_both.csv').open('w', newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['mixture_ID','mixture_path','source_1_path','source_2_path'])
                writer.writerow(['mix','../train-360/mix_both/mix.wav','../train-360/s1/mix.wav','../train-360/s2/mix.wav'])
            (split/'mixture2enrollment.csv').write_text('mixture_id,utterance_id,path,length\nmix,target,s1/enroll.wav,100\n')
            config={'data':dict(train_dirs=[str(split)],validation_dir=str(split),metadata_dir=str(metadata))}
            with self.assertRaisesRegex(FileNotFoundError, 'unpack_shared_data.py'):
                check_data_paths(config, 'train')
            for path in [split/'mix_both/mix.wav',split/'s1/mix.wav',split/'s2/mix.wav',split/'s1/enroll.wav']:
                path.parent.mkdir(exist_ok=True);path.write_bytes(b'RIFF readable fixture')
            self.assertTrue(check_data_paths(config, 'train')[0]['representative_audio_readable'])


if __name__ == '__main__':
    unittest.main()
