import argparse
from copy import deepcopy
from pathlib import Path

import soundfile as sf
import torch

from flowtse_rl.audio import load_audio
from flowtse_rl.config import load_config
from flowtse_rl.inference import extract_chunked, inference_settings
from flowtse_rl.engine import FlowSystem
from flowtse_rl.representation import COMPLEX_STFT, representation
from flowtse_rl.paths import resolve_data_paths


def load_module(checkpoint, device, config_path=None, data_root=None):
    # A checkpoint carries its exact architecture and frontend configuration.
    # Full locally produced Lightning checkpoints include callback Path objects.
    # Older Lightning does not pass weights_only=False to newer PyTorch; read
    # the trusted local format explicitly rather than changing global defaults.
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    config = deepcopy(payload["hyper_parameters"]["config"])
    if config_path:
        replacement = load_config(config_path, data_root=data_root)
        for key in ["model", "audio"]:
            if replacement[key] != config[key]:
                raise ValueError(f"Checkpoint/config mismatch: {key}")
        for key in ["flow", "inference", "validation", "data", "vocoder"]:
            if key in replacement:
                config[key] = deepcopy(replacement[key])
    resolve_data_paths(config, data_root)
    module = FlowSystem(config)
    module.load_state_dict(payload["state_dict"], strict=True)
    return module.eval().to(device)


@torch.no_grad()
def extract(module, mixture, enrollment, seed=42, steps=None, chunk_seconds=None):
    return extract_chunked(module, mixture, enrollment, seed=seed, steps=steps, chunk_seconds=chunk_seconds).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--enrollment", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--chunk-seconds", type=float)
    args = parser.parse_args()
    module = load_module(args.checkpoint, args.device, args.config)
    sample_rate = module.config['audio']['sample_rate']
    waveform = extract(module, load_audio(args.mixture, sample_rate), load_audio(args.enrollment, sample_rate),
                       args.seed, args.steps, args.chunk_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, waveform.numpy(), sample_rate, subtype="FLOAT")
    steps, seconds, _ = inference_settings(module.config, args.steps, args.chunk_seconds)
    decoder = 'iSTFT' if representation(module.config) == COMPLEX_STFT else 'BigVGAN'
    print(f"Saved {args.output}: {waveform.numel()} samples at {sample_rate} Hz; {seconds:g}s feature chunks, "
          f"{steps} Flow steps; one full-utterance {decoder} pass")


if __name__ == "__main__":
    main()
