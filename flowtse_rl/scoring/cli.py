"""CLI for standalone scoring and the bundled validation backend."""
import argparse
from dataclasses import asdict, fields
import json
import os
from pathlib import Path

from .runner import Options, score_directory, validate_options


def add_scoring_arguments(parser):
    parser.add_argument('--scoring-metrics', '--metrics', default=None,
                        help='Comma-separated core,dnsmos,asr,speakers,full or individual metric names')
    parser.add_argument('--scoring-device', choices=['cpu', 'cuda'], default=os.environ.get('SCORING_DEVICE', 'cpu'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dnsmos-workers', type=int, default=1)
    for flag, variable in [('dnsmos-dir', 'DNSMOS_DIR'), ('qwen-model', 'QWEN_ASR_MODEL_ID'),
                           ('wavlm-model', 'WAVLM_MODEL_PATH'), ('wespeaker-model', 'WESPEAKER_MODEL_PATH'),
                           ('librispeech-root', 'LIBRISPEECH_ROOT'), ('vctk-root', 'VCTK_ROOT'),
                           ('vctk-original-root', 'VCTK_ORIGINAL_ROOT'), ('vctk-durations-csv', 'VCTK_DURATIONS_CSV')]:
        parser.add_argument('--' + flag, default=os.environ.get(variable, ''))
    parser.add_argument('--asr-batch-size', type=int, default=16)
    parser.add_argument('--asr-max-new-tokens', type=int, default=1024)
    parser.add_argument('--language', default='English')
    parser.add_argument('--wer-reference', choices=['librispeech-shorter', 'vctk-shorter', 'transcripts'], default='librispeech-shorter')
    parser.add_argument('--transcripts-json', default='')
    parser.add_argument('--wespeaker-workers', type=int, default=4)
    parser.add_argument('--wespeaker-batch-size', type=int, default=8)
    parser.add_argument('--wespeaker-max-batch-frames', type=int, default=8000)


def options_from_args(args):
    values = {item.name: getattr(args, item.name) for item in fields(Options)
              if hasattr(args, item.name)}
    values['metrics'] = args.scoring_metrics or 'core'
    values['device'] = args.scoring_device
    return Options(**values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-dir', type=Path, default=os.environ.get('TEST_DIR'))
    parser.add_argument('--output-dir', type=Path, default=os.environ.get('OUT_DIR'))
    parser.add_argument('--limit', type=int, default=int(os.environ.get('MAX_ITEMS', '0')))
    parser.add_argument('--options-json', type=Path, help='Serialized Options used by the validation adapter')
    parser.add_argument('--check-assets', action='store_true', help='Check dependencies and asset paths without scoring')
    add_scoring_arguments(parser)
    args = parser.parse_args()
    options = Options(**json.loads(args.options_json.read_text())) if args.options_json else options_from_args(args)
    validate_options(options)
    if args.check_assets:
        print(json.dumps({'assets_ready': True, 'options': asdict(options)}, indent=2))
        return
    if args.audio_dir is None or args.output_dir is None:
        parser.error('--audio-dir and --output-dir are required for scoring')
    report = score_directory(args.audio_dir, args.output_dir, options, limit=args.limit)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
