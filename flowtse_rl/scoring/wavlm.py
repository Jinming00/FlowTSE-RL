"""WavLM clean-reference similarity with the production first-channel frontend."""
from pathlib import Path


def load_audio(path):
    import librosa
    import numpy as np
    import soundfile as sf
    audio, rate = sf.read(path, always_2d=True)
    audio = audio.T.astype(np.float32, copy=False)
    if rate != 16000:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=16000, axis=-1)
    return audio[0]


class WavLMSimilarity:
    def __init__(self, model_path, device='cuda'):
        from .asr import configure_audio_only_transformers
        configure_audio_only_transformers()
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        path = Path(model_path).expanduser().resolve()
        self.device = device
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(path), local_files_only=True)
        self.model = WavLMForXVector.from_pretrained(str(path), local_files_only=True).to(device).eval()

    def __call__(self, prediction, reference):
        import torch
        inputs = self.extractor([load_audio(reference), load_audio(prediction)], sampling_rate=16000,
                                padding=True, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            embeddings = self.model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
        return float(torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=-1))
