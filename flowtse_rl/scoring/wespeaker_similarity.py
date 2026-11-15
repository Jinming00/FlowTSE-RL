"""Pinned ResNet221_LM evaluation, using the official WeSpeaker ONNX export.

Frontend follows wespeaker.cli.speaker.Speaker defaults (no VAD, hamming
fbank80, utterance CMN). Scores follow LauraTSE's compute_similarity API:
(cosine + 1) / 2. No AS-Norm, cohort scoring, or utterance truncation.
"""
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
from torchaudio.compliance.kaldi import fbank

MODEL_ID = "Wespeaker/wespeaker-voxceleb-resnet221-LM"
REVISION = "26732f43821f238bfb258c5ef54924a798e62118"
MODEL_FILE = "voxceleb_resnet221_LM.onnx"
DEFAULT_MODEL_PATH = "pretrained/wespeaker-resnet221-lm/" + MODEL_FILE


def compute_features(path):
    # normalize=False preserves PCM integer amplitude, as in the WeSpeaker API.
    waveform, rate = torchaudio.load(str(path), normalize=False)
    waveform = waveform[:1].float()
    if rate != 16000:
        waveform = torchaudio.functional.resample(waveform, rate, 16000)
    if waveform.shape[-1] < 400 or not torch.isfinite(waveform).all():
        raise ValueError(f"WeSpeaker requires finite audio of at least 25 ms: {path}")
    feats = fbank(waveform, num_mel_bins=80, frame_length=25, frame_shift=10,
                  sample_frequency=16000, window_type="hamming", dither=0.0,
                  use_energy=False)
    feats = feats - feats.mean(dim=0)
    return feats.unsqueeze(0).numpy()


def normalized_cosine(first, second):
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(first) * np.linalg.norm(second)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("Invalid WeSpeaker embedding norm")
    cosine = float(np.dot(first, second) / denom)
    if not np.isfinite(cosine):
        raise ValueError("Non-finite WeSpeaker similarity")
    return (float(np.clip(cosine, -1.0, 1.0)) + 1.0) / 2.0


class WeSpeakerSimilarity:
    def __init__(self, model_path=None, device="cuda", batch_size=8,
                 max_batch_frames=8000):
        if device not in ("cpu", "cuda"):
            raise ValueError("WeSpeaker device must be cpu or cuda")
        if batch_size < 1 or max_batch_frames < 1:
            raise ValueError("WeSpeaker batch limits must be positive")
        self.device = device
        self.batch_size = batch_size
        self.max_batch_frames = max_batch_frames
        self.batch_counts = defaultdict(int)
        self.undefined_pairs = []
        self.path = Path(model_path or os.environ.get("WESPEAKER_MODEL_PATH", DEFAULT_MODEL_PATH))
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Missing WeSpeaker model: {self.path}. Pass "
                "--wespeaker-model with the locally downloaded ONNX file.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("WeSpeaker CUDA requires onnxruntime-gpu; "
                                   "select --wespeaker-device cpu explicitly for CPU evaluation")
            # Do not silently turn a failed CUDA initialization into a slow CPU run.
            # ONNX shape/index operators need the CPU provider even when all
            # convolutions run on CUDA. Verify CUDA registration below instead
            # of forbidding these small host-side operations.
            providers = [("CUDAExecutionProvider", {
                "device_id": 0, "use_tf32": 0,
                "cudnn_conv_algo_search": "HEURISTIC",
            })]
        self.session = ort.InferenceSession(str(self.path), sess_options=options,
                                           providers=providers)
        expected = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        if self.session.get_providers()[0] != expected:
            raise RuntimeError(f"WeSpeaker requested {expected}, got {self.session.get_providers()}")
        self.session.disable_fallback()

    def _embed_batch(self, features):
        """All items have identical frame counts; no padding or cropping."""
        try:
            result = self.session.run(["embs"], {"feats": np.concatenate(features, axis=0)})[0]
        except Exception as error:
            # Preserve full utterances if a long batch exhausts device memory.
            message = str(error).lower()
            if self.device != "cuda" or len(features) == 1 or not any(
                word in message for word in ("out of memory", "failed to allocate memory", "cudaerrormemoryallocation")
            ):
                raise
            mid = len(features) // 2
            return np.concatenate((self._embed_batch(features[:mid]),
                                   self._embed_batch(features[mid:])), axis=0)
        if result.shape != (len(features), 256):
            raise ValueError(f"Unexpected WeSpeaker output shape: {result.shape}")
        self.batch_counts[len(features)] += 1
        return result

    def embedding(self, path):
        return self._embed_batch([compute_features(path)])[0]

    def __call__(self, prediction, reference):
        return self._score(self.embedding(prediction), self.embedding(reference), prediction, reference)

    def _score(self, first, second, prediction, reference):
        try:
            return normalized_cosine(first, second)
        except ValueError:
            short_inputs = []
            for embedding, path in ((first, prediction), (second, reference)):
                norm = np.linalg.norm(np.asarray(embedding, dtype=np.float64))
                if np.isfinite(norm) and norm > 0:
                    continue
                info = sf.info(path)
                # Extremely early EOS can leave only one pooled time point;
                # the exported unbiased variance then divides by n - 1 = 0.
                # Preserve an explicitly undefined metric for this short input.
                if info.frames / info.samplerate >= 0.25:
                    raise ValueError(f"Invalid WeSpeaker embedding for ordinary-length audio: {path}")
                short_inputs.append(dict(path=str(path), frames=info.frames,
                                         sample_rate=info.samplerate))
            if not short_inputs:
                raise
            self.undefined_pairs.append(dict(prediction=str(prediction), reference=str(reference),
                                             reason="insufficient_audio_for_embedding", inputs=short_inputs))
            return float("nan")

    def score_pairs(self, predictions, references, workers=4):
        """Yield scores in input order with bounded prefetch and exact-length batches.

        Workers compute CPU fbanks only. One caller owns GPU inference, so batches
        cannot overlap and multiply device-memory usage. Unequal-length tails are
        evaluated separately; this model has no length mask for its statistics pool.
        """
        if len(predictions) != len(references):
            raise ValueError("Prediction/reference counts differ")
        if workers < 1:
            raise ValueError("workers must be positive")
        window_pairs = max(1, self.batch_size * 4)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(predictions), window_pairs):
                pairs = list(zip(predictions[start:start + window_pairs],
                                 references[start:start + window_pairs]))
                paths = [path for pair in pairs for path in pair]
                features = list(pool.map(compute_features, paths))
                groups = defaultdict(list)
                for index, feats in enumerate(features):
                    groups[feats.shape[1]].append(index)
                embeddings = [None] * len(paths)
                for frames, indices in groups.items():
                    count = max(1, min(self.batch_size, self.max_batch_frames // frames))
                    for offset in range(0, len(indices), count):
                        selected = indices[offset:offset + count]
                        result = self._embed_batch([features[i] for i in selected])
                        for index, embedding in zip(selected, result):
                            embeddings[index] = embedding
                for index in range(len(pairs)):
                    yield self._score(embeddings[2 * index], embeddings[2 * index + 1],
                                      pairs[index][0], pairs[index][1])

    def metadata(self):
        return dict(metric="sim_wespeaker", model_id=MODEL_ID, revision=REVISION,
                    filename=MODEL_FILE, model_path=str(self.path), identity_check="not performed",
                    score="(cosine + 1) / 2", reference="paired clean *_ref.wav",
                    sample_rate=16000, fbank_bins=80, frame_length_ms=25,
                    frame_shift_ms=10, window="hamming", dither=0.0,
                    cmn="utterance", vad=False, as_norm=False, truncation=False,
                    channel="first", waveform_normalize=False,
                    provider=self.session.get_providers()[0], batch_size=self.batch_size,
                    max_batch_frames=self.max_batch_frames, batching="exact_fbank_length_no_padding",
                    batch_counts=dict(self.batch_counts), tf32=False,
                    undefined_pairs=self.undefined_pairs,
                    invalid_short_audio="NaN with explicit input paths; other invalid embeddings raise",
                    onnxruntime_version=ort.__version__,
                    torchaudio_version=torchaudio.__version__)
