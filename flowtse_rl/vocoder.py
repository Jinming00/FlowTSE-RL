"""Frozen, local-only BigVGAN. No automatic model download or CUDA compilation."""
import json
import math
from pathlib import Path

import torch


class FrozenBigVGAN:
    """Kept outside nn.Module registration: checkpoint contains the separator only."""
    def __init__(self, checkpoint_dir, **_):
        self.root = Path(checkpoint_dir).expanduser().resolve()
        self.model = None
        h = json.loads((self.root / "config.json").read_text())
        signature = tuple(h[key] for key in ["sampling_rate", "num_mels", "hop_size", "n_fft", "win_size", "fmin", "fmax"])
        if signature != (24000, 100, 256, 1024, 1024, 0, None) or math.prod(h["upsample_rates"]) != 256:
            raise ValueError(f"Incompatible BigVGAN configuration: {signature}")
        if not (self.root / "bigvgan_generator.pt").is_file():
            raise FileNotFoundError(self.root / "bigvgan_generator.pt")
        self.hparams = h

    def load(self, device):
        if self.model is None:
            from .vendor.bigvgan.bigvgan import BigVGAN
            from .vendor.bigvgan.env import AttrDict
            path = self.root / "bigvgan_generator.pt"
            model = BigVGAN(AttrDict(self.hparams), use_cuda_kernel=False)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["generator"], strict=True)
            model.remove_weight_norm()
            self.model = model.eval().requires_grad_(False)
        self.model.to(device)
        return self.model

    @torch.no_grad()
    def __call__(self, mel, sample_lengths=None):
        if mel.ndim != 3 or mel.shape[1] != 100:
            raise ValueError("BigVGAN expects [B,100,T] natural-log mel")
        if not bool(torch.isfinite(mel).all()):
            raise ValueError("Non-finite generated mel")
        model = self.load(mel.device)
        if sample_lengths is None:
            with torch.autocast(mel.device.type, enabled=False):
                return model(mel.float())[:, 0]
        sample_lengths = sample_lengths.to(mel.device, torch.long)
        if sample_lengths.shape != (len(mel),) or bool((sample_lengths < 1).any()):
            raise ValueError("Invalid synthesis lengths")
        frames = (sample_lengths + 255) // 256
        if bool((frames > mel.shape[-1]).any()):
            raise RuntimeError("Mel does not cover every requested sample")
        output = [None] * len(mel)
        # Decode true frame extents: padded mel must not change the real tail
        # through BigVGAN's noncausal convolution receptive field.
        with torch.autocast(mel.device.type, enabled=False):
            for count in frames.unique().tolist():
                indices = (frames == count).nonzero().flatten()
                waves = model(mel[indices, :, :count].float())[:, 0]
                for row, index in enumerate(indices.tolist()):
                    length = int(sample_lengths[index])
                    if length > waves.shape[-1]:
                        raise RuntimeError("Decoder did not cover every requested sample")
                    output[index] = waves[row, :length]
        return output

