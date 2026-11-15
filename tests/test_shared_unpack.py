"""Exercise extraction and resume using a tiny local archive."""
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


@unittest.skipUnless(shutil.which('zstd') and shutil.which('rsync'), 'zstd and rsync are required')
class SharedUnpackTests(unittest.TestCase):
    def test_extract_resume_and_archive_size_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'shared/data'
            packages = source / 'audio_archives'
            packages.mkdir(parents=True)
            (source.parent / 'DATA_READY.json').write_text(json.dumps({'status': 'ready'}))
            (source / 'pairings.csv').write_text('id,path\nfixture,relative.wav\n')
            payload = b'RIFF tiny extraction fixture'
            archive_name = 'train-360-s1.tar.zst'
            tar_path = root / 'fixture.tar'
            with tarfile.open(tar_path, 'w') as archive:
                info = tarfile.TarInfo('LibriMix/Libri2Mix/wav16k/min/train-360/s1/example.wav')
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            archive_path = packages / archive_name
            subprocess.run(['zstd', '-q', '-f', str(tar_path), '-o', str(archive_path)], check=True)
            manifest = {'status': 'complete', 'packages': [{
                'split': 'train-360', 'kind': 's1', 'files': 1, 'bytes': len(payload),
                'archive': archive_name, 'archive_bytes': archive_path.stat().st_size,
                'sha256': 'legacy information only; intentionally not a checksum',
            }]}
            (packages / 'manifest.json').write_text(json.dumps(manifest))
            script = Path(__file__).resolve().parents[1] / 'scripts/unpack_shared_data.py'
            command = [sys.executable, str(script), '--source-data-dir', str(source), '--workers', '1']
            target = root / 'working-data'
            first = subprocess.run(command + ['--data-dir', str(target)], text=True, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            waveform = target / 'LibriMix/Libri2Mix/wav16k/min/train-360/s1/example.wav'
            self.assertEqual(waveform.read_bytes(), payload)
            self.assertTrue((target / 'pairings.csv').is_file())
            self.assertFalse((target / 'audio_archives').exists())
            second = subprocess.run(command + ['--data-dir', str(target)], text=True, capture_output=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertNotIn('Unpacked:', second.stdout)
            with archive_path.open('ab') as stream:
                stream.write(b'changed length')
            failed = subprocess.run(command + ['--data-dir', str(root / 'other-data')],
                                    text=True, capture_output=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('Archive size differs', failed.stderr)


if __name__ == '__main__':
    unittest.main()
