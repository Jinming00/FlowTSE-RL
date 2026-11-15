"""Relocatable dataset paths and lightweight checks before model allocation."""
import csv
import os
from pathlib import Path


SHARED_DATA_ROOT = Path('/share/202611/FlowTSE-RL-share/data')
DATA_PATHS = ('train_dirs', 'validation_dir', 'test_dir', 'metadata_dir', 'asr_reference_dir')


def resolve_data_paths(config, data_root=None):
    data = config.get('data', {})
    previous = data.get('root')
    override = data_root if data_root is not None else os.environ.get('FLOWTSE_DATA_ROOT')
    selected = override if override is not None else previous
    if selected is None:
        return config
    root = Path(os.path.expandvars(str(selected))).expanduser().resolve()
    old = Path(previous).expanduser().resolve() if previous else None

    def relocate(value):
        path = Path(value).expanduser()
        if not path.is_absolute():
            return str(root / path)
        if override is not None and old is not None:
            try:
                return str(root / path.relative_to(old))
            except ValueError:
                pass
        if override is not None and old is None and 'LibriMix' in path.parts:
            return str(root.joinpath(*path.parts[path.parts.index('LibriMix'):]))
        return str(path)

    for key in DATA_PATHS:
        if key in data:
            data[key] = [relocate(item) for item in data[key]] if isinstance(data[key], list) else relocate(data[key])
    data['root'] = str(root)
    return config


def resolve_audio_path(value, csv_directory, prefix_map=None):
    path = str(value)
    for old, new in sorted((prefix_map or {}).items(), key=lambda pair: len(pair[0]), reverse=True):
        prefix = str(old).rstrip('/')
        if path == prefix or path.startswith(prefix + '/'):
            path = str(new).rstrip('/') + path[len(prefix):]
            break
    candidate = Path(path).expanduser()
    return str(candidate if candidate.is_absolute() else Path(csv_directory) / candidate)


def check_data_paths(config, split='train'):
    """Check CSVs and representative audio only; do not scan or hash audio files."""
    data = config['data']
    if split == 'train':
        directories = [*data['train_dirs'], data['validation_dir']]
    elif split in {'dev', 'test'}:
        directories = [data['validation_dir'] if split == 'dev' else data['test_dir']]
    else:
        raise ValueError('Expected train, dev or test')
    reports = []
    for directory in dict.fromkeys(directories):
        directory = Path(directory)
        mixture_csv = Path(data['metadata_dir']) / f'mixture_{directory.name}_mix_both.csv'
        enrollment_csv = directory / 'mixture2enrollment.csv'
        for path in [mixture_csv, enrollment_csv]:
            if not path.is_file():
                raise FileNotFoundError(f'Missing dataset pairing CSV: {path}. Set --data-root or FLOWTSE_DATA_ROOT.')
        with mixture_csv.open(newline='') as stream:
            mixture = next(csv.DictReader(stream), None)
        with enrollment_csv.open(newline='') as stream:
            reader = csv.reader(stream)
            next(reader, None)
            enrollment = next(reader, None)
        if mixture is None or enrollment is None or len(enrollment) < 4:
            raise ValueError(f'Empty or invalid dataset metadata: {directory}')
        paths = [resolve_audio_path(mixture[key], mixture_csv.parent, data.get('path_prefix_map'))
                 for key in ['mixture_path', 'source_1_path', 'source_2_path']]
        paths.append(resolve_audio_path(enrollment[2], directory, data.get('path_prefix_map')))
        for value in paths:
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(f'Audio is missing: {path}. The shared train-360 audio is packaged: '
                    'run scripts/unpack_shared_data.py --data-dir /your/writable/data, '
                    'then train with --data-root /your/writable/data.')
            with path.open('rb') as stream:
                if not stream.read(1):
                    raise ValueError(f'Empty audio file: {path}')
        reports.append(dict(split=directory.name, mixture_csv=str(mixture_csv),
                            enrollment_csv=str(enrollment_csv), representative_audio_readable=True))
    return reports
