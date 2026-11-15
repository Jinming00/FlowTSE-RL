from copy import deepcopy
import math
import os
from pathlib import Path

import yaml
from .representation import COMPLEX_STFT, feature_dim, representation
from .paths import resolve_data_paths

BASELINE_CONFIG = Path(__file__).resolve().parents[1] / "configs/baseline.yaml"

# Execution policy lives in code so experiment YAMLs describe the model and
# training recipe. Explicit values in old snapshots/diagnostics remain valid.
EXECUTION_DEFAULTS = {
    "model": {
        "use_checkpoint": False,
        "compile_blocks": True,
        "skip_redundant_mask": True,
        "vectorized_packing": True,
        "compile_mode": "max-autotune-no-cudagraphs",
    },
    "data": {"num_workers": 8},
    "optimizer": {"fused": False},
}


def merge(base, extra):
    result = deepcopy(base)
    for key, value in extra.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def _load_config(path):
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    parent = value.pop("extends", None)
    return merge(_load_config(path.parent / parent), value) if parent else value


def expand_environment(value):
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    return os.path.expandvars(os.path.expanduser(value)) if isinstance(value, str) else value


def load_config(path, data_root=None):
    value = expand_environment(_load_config(path))
    config = merge(EXECUTION_DEFAULTS, value)
    if config["model"]["use_checkpoint"] and "compile_blocks" not in value.get("model", {}):
        # Legacy recipes explicitly enabling checkpointing predate compilation.
        config["model"]["compile_blocks"] = False
    # Compile the full batch and retain the existing eager tail-batch path.
    config["model"].setdefault("compile_batch_size",
                             config.get("data", {}).get("max_samples_per_batch", 32))
    return resolve_data_paths(config, data_root)


def resolve_warmup_steps(config, total_steps):
    """Convert an epoch budget using the trainer's actual optimizer updates."""
    settings = config["optimizer"]
    if "warmup_epochs" in settings:
        if "warmup_steps" in settings:
            raise ValueError("Specify either optimizer.warmup_epochs or warmup_steps")
        return min(math.ceil(int(total_steps) * settings["warmup_epochs"] /
                             config["train"]["num_epochs"]), int(total_steps))
    return int(settings.get("warmup_steps", 0))


def validate_config(config):
    audio, model = config["audio"], config["model"]
    if representation(config) == COMPLEX_STFT:
        if (type(audio['sample_rate']) is not int or audio['sample_rate'] < 1
                or type(audio['n_fft']) is not int or audio['n_fft'] < 2 or audio['n_fft'] % 2
                or not 0 < audio['hop_length'] < audio['win_length'] <= audio['n_fft']):
            raise ValueError('Invalid centered complex STFT parameters')
        if config.get('vocoder', {}).get('checkpoint_dir'):
            raise ValueError('Complex STFT uses iSTFT and must not configure a BigVGAN checkpoint')
    elif (audio["sample_rate"], audio["n_mels"], audio["n_fft"], audio["hop_length"], audio["win_length"]) != (24000, 100, 1024, 256, 1024):
        raise ValueError("Mel baseline requires the 24k/100-band/256-hop BigVGAN frontend")
    if model.get('feature_dim', model.get('mel_dim', 100)) != feature_dim(config):
        raise ValueError('Model feature dimension must match the selected frontend')
    if 'feature_dim' in model and 'mel_dim' in model and model['feature_dim'] != model['mel_dim']:
        raise ValueError('Conflicting model.feature_dim and model.mel_dim')
    if model["hidden_size"] % model["num_heads"]:
        raise ValueError("Hidden size must be divisible by the head count")
    if model.get("conditioning_mode", "channel_concat") not in {"channel_concat", "time_concat", "flowtse"}:
        raise ValueError("Unknown model.conditioning_mode")
    if model.get("attention_mode", "bidirectional") not in {"bidirectional", "enrollment_readonly"}:
        raise ValueError("Unknown model.attention_mode")
    if model.get("compile_mode", "default") not in {"default", "max-autotune-no-cudagraphs"}:
        raise ValueError("Unknown model.compile_mode")
    if "fused" in config["optimizer"] and not isinstance(config["optimizer"]["fused"], bool):
        raise ValueError("optimizer.fused must be a boolean when specified")
    if "warmup_epochs" in config["optimizer"]:
        warmup = config["optimizer"]["warmup_epochs"]
        if (isinstance(warmup, bool) or not isinstance(warmup, (int, float))
                or not math.isfinite(warmup) or not 0 <= warmup <= config["train"]["num_epochs"]):
            raise ValueError("optimizer.warmup_epochs must be between 0 and train.num_epochs")
        if "warmup_steps" in config["optimizer"]:
            raise ValueError("Specify either optimizer.warmup_epochs or warmup_steps")
    if config.get("flow", {}).get("start_state", "gaussian") != "gaussian":
        raise ValueError("FlowTSE-RL uses Gaussian initial states")
    if config.get("flow", {}).get("cfg", False):
        raise ValueError("CFG is intentionally absent from this baseline")
    if config.get("checkpoint", {}).get("load_weights_only", False):
        raise ValueError("Use an explicit model import tool for weights-only initialization; full resume is separate")
    from .inference import inference_settings
    inference_settings(config)
    validation = config["validation"]
    for key in ("loss_ratio", "quality_metrics_ratio"):
        if not 0 < float(validation.get(key, 1.0)) <= 1:
            raise ValueError(f"validation.{key} must be in (0, 1]")
