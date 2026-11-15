"""Join Flow feature chunks, decode once, then score complete LibriMix utterances."""
import argparse
import csv
import json
from pathlib import Path

import soundfile as sf
import torch

from infer import extract, load_module
from flowtse_rl.data import PrecomputedLibriMixDataset
from flowtse_rl.inference import inference_protocol, inference_settings
from flowtse_rl.representation import COMPLEX_STFT, frame_count, representation
from flowtse_rl.quality import save_quality_pair, score_quality
from flowtse_rl.scoring.cli import add_scoring_arguments, options_from_args
from flowtse_rl.paths import check_data_paths


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument('--data-root', type=Path, help='Override dataset root; FLOWTSE_DATA_ROOT is also supported')
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--chunk-seconds", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="0 means all eligible target pairs")
    parser.add_argument("--save-audio", action="store_true")
    add_scoring_arguments(parser)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    module = load_module(args.checkpoint, args.device, args.config, data_root=args.data_root)
    check_data_paths(module.config, args.split)
    steps, chunk_seconds, chunk_samples = inference_settings(module.config, args.steps, args.chunk_seconds)
    sample_rate = module.config['audio']['sample_rate']
    data = dict(module.config["data"], segment_seconds=None, seed=args.seed)
    split_dir = data["validation_dir"] if args.split == "dev" else data["test_dir"]
    dataset = PrecomputedLibriMixDataset([split_dir], data["metadata_dir"], module.config["audio"], data, False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    expected = len(dataset)
    count = min(args.limit, expected) if args.limit else expected
    metadata = {}
    for index in range(count):
        item = dataset[index]
        waveform = extract(module, item["mixture"], item["enrollment"], seed=args.seed + index,
                           steps=steps, chunk_seconds=chunk_seconds)
        name = save_quality_pair(args.output_dir / "audio", item, waveform, item["target"], sample_rate)
        metadata[name] = {"mixture_id": item["mixture_id"], "utterance_id": item["utterance_id"],
                          "samples": len(item["target"]), "output_samples": waveform.numel(), "seed": args.seed + index}
        if args.save_audio:
            target = args.output_dir / item["utterance_id"] / item["mixture_id"]
            target.mkdir(parents=True, exist_ok=True)
            for field, wave in [("estimation", waveform), ("source", item["target"]),
                                ("mixture", item["mixture"]), ("enrollment", item["enrollment"])]:
                sf.write(target / f"{field}.wav", wave.numpy(), sample_rate, subtype="FLOAT")
        if (index + 1) % 10 == 0 or index + 1 == count:
            print(f"Inference: {index + 1}/{count} target utterances; {chunk_seconds:g}s chunks, {steps} steps", flush=True)
    settings = dict(module.config["validation"], quality_audio_keep_items=count if args.save_audio else 0)
    if args.scoring_metrics:
        from dataclasses import asdict
        references = Path(module.config['data'].get('asr_reference_dir', '')) / f'{args.split}.json'
        if (not args.transcripts_json and not args.librispeech_root
                and args.wer_reference == 'librispeech-shorter' and references.is_file()):
            args.wer_reference = 'transcripts'
            args.transcripts_json = str(references)
        settings['scoring_options'] = asdict(options_from_args(args))
    if module.device.type == "cuda":
        torch.cuda.empty_cache()
    report = score_quality(args.output_dir, settings, count, expected_filenames=metadata)
    with (args.output_dir / "metrics/_results.csv").open(newline="") as stream:
        scored = list(csv.DictReader(stream))
    rows = [{**metadata[row["filename"]], **row,
             **({'si_sdr': row['sisdr']} if 'sisdr' in row else {})} for row in scored]
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_bytes": Path(args.checkpoint).stat().st_size,
               "split": args.split, "steps": steps, "chunk_seconds": chunk_seconds,
               "inference_protocol": inference_protocol(module.config),
               "representation": representation(module.config), "sample_rate": sample_rate,
               "decoder": 'istft' if representation(module.config) == COMPLEX_STFT else 'bigvgan',
               "chunk_frames": frame_count(module.config, chunk_samples),
               "solver": module.config["flow"]["solver"], "cfg": False,
               "expected_pairs": expected, "processed_pairs": len(rows), "subset": count != expected,
               "metrics": {}, "unmeasured": []}
    summary['scoring_protocol'] = report.get('protocol', 'compact_torchaudio_core_v1')
    for key, value in report["metrics"].items():
        finite = report["finite_counts"][key]
        coverage = report.get('coverage', {}).get(key, {})
        summary["metrics"][key] = {"mean": value, "count": finite,
            "failed": coverage.get('failed', len(rows) - finite),
            "skipped": coverage.get('skipped', 0), "eligible": coverage.get('eligible', len(rows))}
    for key in ["dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak", "dnsmos_p808", "wer", "cer", "speaker_similarity", "wespeaker_similarity"]:
        if key not in summary["metrics"] or summary['metrics'][key]['count'] == 0:
            summary["unmeasured"].append(key)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "COMPLETE.json").write_text(json.dumps({"processed_pairs": len(rows), "all_eligible_pairs": len(rows) == expected}) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
