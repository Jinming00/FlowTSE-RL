"""Qwen3-ASR and true-text macro WER/CER; models are local and loaded on demand."""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def normalizer():
    from whisper_normalizer.english import EnglishTextNormalizer
    return EnglishTextNormalizer()


def normalize_text(text):
    return normalizer()(str(text).strip()).strip()


def error_rates(reference, hypothesis):
    from jiwer import wer, cer
    reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
    if not reference:
        raise ValueError('True reference transcript is empty after normalization')
    # Preserve the production convention: round each utterance, then macro-average.
    return round(wer(reference, hypothesis), 3), round(cer(reference, hypothesis), 3)


def configure_audio_only_transformers():
    # Retain the production audio-only setup without loading optional vision ops.
    from transformers.utils import import_utils
    import transformers.utils as utils
    import_utils.is_torchvision_available = lambda: False
    import_utils.is_torchvision_v2_available = lambda: False
    utils.is_torchvision_available = lambda: False
    utils.is_torchvision_v2_available = lambda: False


def transcribe(paths, model_path, device='cuda', batch_size=16, max_new_tokens=1024, language='English'):
    if not paths:
        return {}
    import torch
    # Qwen's processor also resolves locally because the ID is a real directory.
    path = Path(model_path).expanduser().resolve()
    if not (path / 'config.json').is_file():
        raise FileNotFoundError(f'Missing local Qwen3-ASR directory: {path}')
    configure_audio_only_transformers()
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as error:
        raise ImportError('Install the scoring-asr extra in the scoring environment') from error
    model = Qwen3ASRModel.from_pretrained(str(path), local_files_only=True,
        dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
        device_map='cuda:0' if device == 'cuda' else 'cpu',
        max_inference_batch_size=batch_size, max_new_tokens=max_new_tokens)
    result = {}
    language = None if language.lower() in {'auto', 'none', ''} else language
    for start in range(0, len(paths), batch_size):
        batch = list(map(str, paths[start:start + batch_size]))
        outputs = model.transcribe(audio=batch, language=[language] * len(batch) if language else None)
        if isinstance(outputs, (str, dict)) or hasattr(outputs, 'text'):
            outputs = [outputs]
        if len(outputs) != len(batch):
            raise RuntimeError(f'ASR returned {len(outputs)} results for {len(batch)} files')
        for path, output in zip(batch, outputs):
            text = (output if isinstance(output, str) else
                    output.get('text', output.get('transcription', '')) if isinstance(output, dict) else output.text)
            result[path] = normalize_text(text)
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return result
