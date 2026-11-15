"""Unpack the shared training WAV packages into a writable data directory."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess


def inventory(directory):
    count = size = 0
    if directory.is_dir():
        with os.scandir(directory) as files:
            for entry in files:
                if entry.name.endswith('.wav'):
                    count += 1
                    size += entry.stat().st_size
    return count, size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-data-dir', type=Path, default=os.environ.get('FLOWTSE_SHARED_DATA_ROOT', '/share/202611/FlowTSE-RL-share/data'), help='Shared source data directory')
    parser.add_argument('--data-dir', type=Path, required=True, help='Writable output data root with enough free space')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--overwrite', action='store_true', help='Replace incomplete existing training WAVs when no completion marker matches')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    source = args.source_data_dir.expanduser()
    target = args.data_dir.expanduser().resolve()
    source = source.resolve()
    if source in target.parents:
        parser.error('Output cannot be nested within the source data directory')
    executable = shutil.which('zstd')
    if not executable:
        parser.error('Install the zstd command-line tool before unpacking')
    packages = json.loads((source / 'audio_archives/manifest.json').read_text())
    if packages['status'] != 'complete':
        parser.error('Training audio packages are not ready yet')
    if not (source.parent / 'DATA_READY.json').is_file():
        parser.error('Wait for the shared DATA_READY.json completion marker')
    ready = json.loads((source.parent / 'DATA_READY.json').read_text())
    target.mkdir(parents=True, exist_ok=True)
    markers = target / '.flowtse_shared_audio'
    markers.mkdir(exist_ok=True)
    pending = []
    for item in packages['packages']:
        folder = target / 'LibriMix/Libri2Mix/wav16k/min' / item['split'] / item['kind']
        marker = markers / (item['archive'] + '.json')
        expected = item['files'], item['bytes']
        if (marker.is_file() and json.loads(marker.read_text()).get('archive') == item['archive']
                and inventory(folder) == expected):
            continue
        if inventory(folder)[0] and not args.overwrite:
            parser.error(f'{folder} already contains training WAVs; use an empty output directory or explicitly pass --overwrite')
        pending.append((item, folder, marker))
    required = sum(item['bytes'] for item, _, _ in pending)
    if target != source:
        for item in ready.get('unpacked_audio_directories', []):
            folder = target / 'LibriMix/Libri2Mix/wav16k/min' / item['split'] / item['kind']
            if inventory(folder) != (item['files'], item['bytes']):
                required += item['bytes']
        required += 1 << 30  # CSVs and reference metadata.
    if shutil.disk_usage(target).free < required + (2 << 30):
        parser.error(f'Insufficient free space: need about {(required + (2 << 30))/2**30:.1f} GiB for this preparation')
    if target != source:
        subprocess.run(['rsync', '-rlt', '--exclude=/audio_archives/', '--exclude=*.partial',
                        str(source) + '/', str(target) + '/'], check=True)

    def unpack(job):
        item, folder, marker = job
        archive = source / 'audio_archives' / item['archive']
        log = marker.with_suffix('.log')
        if archive.stat().st_size != item['archive_bytes']:
            raise RuntimeError(f'Archive size differs from the shared manifest: {archive}')
        with log.open('wb') as errors:
            decoder = subprocess.Popen([executable, '--quiet', '--decompress', '--stdout'],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
            extractor = subprocess.Popen(['tar', '--extract', '--file=-', '--directory', str(target),
                                          '--no-same-owner', '--no-same-permissions'],
                                         stdin=decoder.stdout, stderr=errors)
            decoder.stdout.close()
            try:
                with archive.open('rb') as stream:
                    while block := stream.read(8 << 20):
                        decoder.stdin.write(block)
            finally:
                decoder.stdin.close()
            decode_status, extract_status = decoder.wait(), extractor.wait()
        if decode_status or extract_status:
            raise RuntimeError(f'Archive integrity or extraction failure: {archive}; details: {log}')
        if inventory(folder) != (item['files'], item['bytes']):
            raise RuntimeError(f'Extracted WAV count/size mismatch: {folder}')
        temporary = marker.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(archive=item['archive'], archive_bytes=item['archive_bytes'],
                                             files=item['files'], bytes=item['bytes']), indent=2) + '\n')
        os.replace(temporary, marker)
        return item['archive']

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name in pool.map(unpack, pending):
            print('Unpacked:', name, flush=True)
    print('Data ready:', target, flush=True)


if __name__ == '__main__':
    main()
