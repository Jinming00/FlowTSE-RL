from __future__ import annotations

import json
import math
import random
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, Sampler

from .paths import resolve_audio_path

from .audio import crop_or_repeat, load_audio, normalize_mixture, pad_waveforms, rms, scale_to_snr


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


class OnlineMixtureDataset(Dataset):
    """Create two-speaker mixtures and optional WHAM! noise on demand."""

    def __init__(self, config: dict[str, Any], audio_config: dict[str, Any], training: bool) -> None:
        super().__init__()
        manifest = config["train_manifest"] if training else config["validation_manifest"]
        self.records = read_jsonl(manifest)
        self.noises = read_jsonl(config["noise_manifest"]) if config.get("noise_manifest") else []
        self.training = training
        self.seed = int(config.get("seed", 42))
        self.sample_rate = int(audio_config["sample_rate"])
        self.target_rms = float(audio_config["target_rms"])
        self.noise_probability = float(config["noise_probability"])
        self.speaker_snr_db = tuple(map(float, config["speaker_snr_db"]))
        self.noise_snr_db = tuple(map(float, config["noise_snr_db"]))
        self.enrollment_seconds = tuple(map(float, config["enrollment_seconds"]))
        self.mixture_seconds = tuple(map(float, config["mixture_seconds"]))
        self.min_active_rms = float(config["min_active_rms"])

        self.by_speaker: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            self.by_speaker[str(record["speaker"])].append(index)
        self.speakers = sorted(self.by_speaker)
        if len(self.speakers) < 2:
            raise ValueError("At least two speakers are required")
        self.indices = [
            index for index, record in enumerate(self.records) if len(self.by_speaker[str(record["speaker"])]) >= 2
        ]
        if not self.indices:
            raise ValueError("No speaker has two utterances for target plus enrollment")

    def __len__(self) -> int:
        return len(self.indices)

    def estimated_frames(self, index: int, hop_length: int, mixture_copies: int = 1) -> int:
        record = self.records[self.indices[index]]
        mix_seconds = min(float(record["duration"]), self.mixture_seconds[1])
        return math.ceil((mixture_copies * mix_seconds + self.enrollment_seconds[1]) * self.sample_rate / hop_length) + 1 + mixture_copies

    def _rng(self, index: int) -> random.Random:
        if self.training:
            return random
        return random.Random(self.seed + index)

    @staticmethod
    def _random_start(rng: random.Random, total: int, length: int) -> int:
        return rng.randint(0, max(0, total - length))

    def _load_active(self, record: dict[str, Any], length: int, rng: random.Random) -> torch.Tensor:
        waveform = load_audio(record["path"], self.sample_rate)
        for _ in range(4):
            start = self._random_start(rng, waveform.numel(), length)
            segment = crop_or_repeat(waveform, length, start)
            if rms(segment).item() >= self.min_active_rms:
                return segment
        return crop_or_repeat(waveform, length, 0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = self._rng(index)
        target_index = self.indices[index]
        target_record = self.records[target_index]
        target_speaker = str(target_record["speaker"])

        other_speaker = rng.choice([speaker for speaker in self.speakers if speaker != target_speaker])
        interferer_record = self.records[rng.choice(self.by_speaker[other_speaker])]
        enrollment_candidates = [item for item in self.by_speaker[target_speaker] if item != target_index]
        enrollment_record = self.records[rng.choice(enrollment_candidates)]

        available_seconds = min(float(target_record["duration"]), float(interferer_record["duration"]))
        upper_seconds = min(self.mixture_seconds[1], available_seconds)
        lower_seconds = min(self.mixture_seconds[0], upper_seconds)
        duration = rng.uniform(lower_seconds, upper_seconds)
        mix_length = max(1, round(duration * self.sample_rate))

        target = self._load_active(target_record, mix_length, rng)
        interferer = self._load_active(interferer_record, mix_length, rng)
        speaker_snr = rng.uniform(*self.speaker_snr_db)
        interferer = scale_to_snr(target, interferer, speaker_snr)
        mixture = target + interferer
        components = [target, interferer]

        noise_snr = None
        if self.noises and rng.random() < self.noise_probability:
            noise_record = rng.choice(self.noises)
            noise = self._load_active(noise_record, mix_length, rng)
            noise_snr = rng.uniform(*self.noise_snr_db)
            noise = scale_to_snr(mixture, noise, noise_snr)
            mixture = mixture + noise
            components.append(noise)

        mixture, components = normalize_mixture(mixture, components, self.target_rms)
        target = components[0]

        enrollment_duration = rng.uniform(*self.enrollment_seconds)
        enrollment_length = max(1, round(enrollment_duration * self.sample_rate))
        enrollment = self._load_active(enrollment_record, enrollment_length, rng)
        enrollment = enrollment * (self.target_rms / rms(enrollment))

        return {
            "mixture": mixture,
            "target": target,
            "enrollment": enrollment,
            "target_path": target_record["path"],
            "enrollment_path": enrollment_record["path"],
            "speaker_snr_db": speaker_snr,
            "noise_snr_db": noise_snr,
        }


def read_enrollment_csv(path: str | Path) -> dict[tuple[str, str], list[tuple[str, int]]]:
    mapping: dict[tuple[str, str], list[tuple[str, int]]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            mixture_id, utterance_id, *values = row
            candidates = []
            for candidate_path, candidate_length in zip(values[0::2], values[1::2]):
                if candidate_path and candidate_length:
                    candidates.append((candidate_path, int(float(candidate_length))))
            mapping[(mixture_id, utterance_id)] = candidates
    return mapping



class PrecomputedLibriMixDataset(Dataset):
    """Read the prepared LibriMix/SpeakerBeam mixture and enrollment layout."""

    def __init__(
        self,
        split_dirs: list[str | Path],
        metadata_dir: str | Path,
        audio_config: dict[str, Any],
        data_config: dict[str, Any],
        training: bool,
    ) -> None:
        super().__init__()
        self.training = training
        self.seed = int(data_config.get("seed", 42))
        self.input_sample_rate = int(data_config["input_sample_rate"])
        self.sample_rate = int(audio_config["sample_rate"])
        self.target_rms = float(audio_config["target_rms"])
        self.segment_seconds = data_config.get("segment_seconds")
        self.enrollment_seconds = tuple(map(float, data_config["enrollment_seconds"]))
        self.records: list[dict[str, Any]] = []

        metadata_dir = Path(metadata_dir)
        prefix_map = data_config.get("path_prefix_map", {})
        # Keep short but nonempty enrollment utterances; crop to the available
        # duration instead of silently losing target pairs from the evaluation.
        minimum_enrollment = 1
        for split_value in split_dirs:
            split_dir = Path(split_value)
            split_name = split_dir.name
            mixture_csv = metadata_dir / f"mixture_{split_name}_mix_both.csv"
            enrollment_csv = split_dir / "mixture2enrollment.csv"
            if not mixture_csv.is_file():
                raise FileNotFoundError(f"Missing mixture metadata: {mixture_csv}")
            if not enrollment_csv.is_file():
                raise FileNotFoundError(f"Missing enrollment mapping: {enrollment_csv}")

            with mixture_csv.open("r", encoding="utf-8") as handle:
                mixtures = {row["mixture_ID"]: row for row in csv.DictReader(handle)}
            enrollment = read_enrollment_csv(enrollment_csv)
            enrollment = {key: [(resolve_audio_path(path, split_dir, prefix_map), count)
                                 for path, count in candidates]
                          for key, candidates in enrollment.items()}
            for (mixture_id, utterance_id), candidates in enrollment.items():
                candidates = [item for item in candidates if item[1] >= minimum_enrollment]
                if not candidates or mixture_id not in mixtures:
                    continue
                row = mixtures[mixture_id]
                utterances = mixture_id.split("_")
                if utterance_id not in utterances:
                    continue
                source_index = utterances.index(utterance_id) + 1
                self.records.append(
                    {
                        "mixture_id": mixture_id,
                        "utterance_id": utterance_id,
                        "mixture_path": resolve_audio_path(row["mixture_path"], metadata_dir, prefix_map),
                        "source_path": resolve_audio_path(row[f"source_{source_index}_path"], metadata_dir, prefix_map),
                        "length": int(float(row["length"])),
                        "enrollment": candidates,
                    }
                )

        ratio = float(data_config.get("train_data_ratio", 1.0)) if training else 1.0
        if not 0 < ratio <= 1:
            raise ValueError("data.train_data_ratio must be in (0, 1]")
        if ratio < 1:
            rng = random.Random(self.seed)
            keep = max(1, math.ceil(len(self.records) * ratio))
            self.records = rng.sample(self.records, keep)
        if not self.records:
            raise ValueError(f"No usable examples found in {split_dirs}")

    def __len__(self) -> int:
        return len(self.records)

    def estimated_frames(self, index: int, hop_length: int, mixture_copies: int = 1) -> int:
        record = self.records[index]
        mixture_samples = record["length"]
        if self.segment_seconds is not None:
            mixture_samples = min(mixture_samples, round(float(self.segment_seconds) * self.input_sample_rate))
        seconds = mixture_copies * mixture_samples / self.input_sample_rate + self.enrollment_seconds[1]
        return math.ceil(seconds * self.sample_rate / hop_length) + 1 + mixture_copies

    def _rng(self, index: int) -> random.Random:
        return random if self.training else random.Random(self.seed + index)

    def _read_segment(self, path: str, start: int, stop: int | None) -> torch.Tensor:
        import soundfile as sf
        import torchaudio

        waveform, sample_rate = sf.read(path, dtype="float32", start=start, stop=stop)
        value = torch.from_numpy(waveform)
        if value.ndim == 2:
            value = value.mean(dim=1)
        if sample_rate != self.sample_rate:
            value = torchaudio.functional.resample(value, sample_rate, self.sample_rate)
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        rng = self._rng(index)
        segment_samples = None
        if self.segment_seconds is not None:
            segment_samples = round(float(self.segment_seconds) * self.input_sample_rate)
        if segment_samples is not None and record["length"] > segment_samples:
            start = rng.randint(0, record["length"] - segment_samples)
            stop = start + segment_samples
        else:
            start, stop = 0, None
        mixture = self._read_segment(record["mixture_path"], start, stop)
        target = self._read_segment(record["source_path"], start, stop)
        common_length = min(mixture.numel(), target.numel())
        mixture, target = mixture[:common_length], target[:common_length]
        mixture, components = normalize_mixture(mixture, [target], self.target_rms)
        target = components[0]

        enrollment_path, enrollment_source_length = rng.choice(record["enrollment"])
        maximum_seconds = min(self.enrollment_seconds[1], enrollment_source_length / self.input_sample_rate)
        enrollment_seconds = rng.uniform(min(self.enrollment_seconds[0], maximum_seconds), maximum_seconds)
        enrollment_source_samples = round(enrollment_seconds * self.input_sample_rate)
        enrollment_start = rng.randint(0, max(0, enrollment_source_length - enrollment_source_samples))
        enrollment = self._read_segment(
            enrollment_path, enrollment_start, enrollment_start + enrollment_source_samples
        )
        enrollment = enrollment * (self.target_rms / rms(enrollment))
        return {
            "mixture": mixture,
            "target": target,
            "enrollment": enrollment,
            "target_path": record["source_path"],
            "enrollment_path": enrollment_path,
            "mixture_id": record["mixture_id"],
            "utterance_id": record["utterance_id"],
            "record_index": index,
        }


def collate_mixtures(items: list[dict[str, Any]]) -> dict[str, Any]:
    mixture, mixture_lengths = pad_waveforms([item["mixture"] for item in items])
    target, target_lengths = pad_waveforms([item["target"] for item in items])
    enrollment, enrollment_lengths = pad_waveforms([item["enrollment"] for item in items])
    if not torch.equal(mixture_lengths, target_lengths):
        raise RuntimeError("Mixture and target lengths diverged")
    return {
        "mixture": mixture,
        "target": target,
        "enrollment": enrollment,
        "mixture_lengths": mixture_lengths,
        "target_lengths": target_lengths,
        "enrollment_lengths": enrollment_lengths,
        "metadata": items,
    }


class FrameBatchSampler(Sampler[list[int]]):
    """Pack examples under a padded-frame budget and shard complete batches by rank."""

    def __init__(
        self,
        dataset: Dataset,
        max_frames: int,
        max_samples: int,
        hop_length: int,
        shuffle: bool,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        mixture_copies: int = 1,
    ) -> None:
        self.dataset = dataset
        self.max_frames = max_frames
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self._length_cache = None
        self.frames = [dataset.estimated_frames(index, hop_length, mixture_copies=mixture_copies)
                       for index in range(len(dataset))]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _global_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng.shuffle(indices)
        batches: list[list[int]] = []
        batch: list[int] = []
        max_length = 0
        for index in indices:
            next_max = max(max_length, self.frames[index])
            exceeds = batch and (
                next_max * (len(batch) + 1) > self.max_frames or len(batch) >= self.max_samples
            )
            if exceeds:
                batches.append(batch)
                batch = []
                max_length = 0
            batch.append(index)
            max_length = max(max_length, self.frames[index])
        if batch:
            batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        usable = len(batches) - (len(batches) % self.world_size)
        if self.world_size > 1 and usable == 0:
            raise ValueError("Fewer training batches than ranks; reduce world size or batch budget")
        return batches[:usable] if self.world_size > 1 else batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._global_batches()[self.rank :: self.world_size]

    def __len__(self) -> int:
        # Lightning queries len(dataloader) after every batch. Rebuilding the
        # complete epoch plan here would repeatedly shuffle the full dataset.
        # Iteration still constructs exactly the original plan and uses the
        # original rank sharding; only this scalar length is cached.
        key = (self.epoch, self.world_size, self.max_frames, self.max_samples, self.shuffle, self.seed)
        if self._length_cache is None or self._length_cache[0] != key:
            self._length_cache = (key, math.ceil(len(self._global_batches()) / self.world_size))
        return self._length_cache[1]
