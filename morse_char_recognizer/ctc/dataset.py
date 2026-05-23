"""Synthetic multi-character sequence dataset for CTC training."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from morse_char_recognizer.ctc.frames import FrameConfig, audio_to_frames
from morse_char_recognizer.ctc.vocab import CtcVocab
from morse_synth.channel import ChannelConfig
from morse_synth.core import render
from morse_synth.keying import KeyingConfig
from morse_synth.operator import OperatorConfig

Range = tuple[float, float]


@dataclass(frozen=True)
class CtcDatasetConfig:
    """Parameter ranges for synthetic CTC training examples."""

    vocab: CtcVocab = field(default_factory=CtcVocab)
    frames: FrameConfig = field(default_factory=FrameConfig)
    num_examples: int = 4000
    sample_rate: int = 8000
    min_chars: int = 3
    max_chars: int = 10
    word_probability: float = 0.25  # chance to insert a word break per character
    text_pool: tuple[str, ...] | None = None
    seed: int = 0
    wpm_range: Range = (14.0, 28.0)
    freq_range: Range = (500.0, 750.0)
    amplitude_range: Range = (0.35, 0.9)
    rise_ms_range: Range = (2.0, 8.0)
    tail_ms_range: Range = (35.0, 90.0)
    element_jitter_range: Range = (0.0, 0.10)
    gap_jitter_range: Range = (0.0, 0.10)
    dash_dot_ratio_range: Range = (2.8, 3.3)
    gap_inflation_range: Range = (0.95, 1.15)
    word_gap_inflation_range: Range = (1.0, 2.0)
    snr_db_range: Range | None = None
    pad_ms_range: Range = (30.0, 120.0)


@dataclass
class CtcExample:
    text: str
    audio: np.ndarray
    frames: np.ndarray
    targets: list[int]
    sample_rate: int


class SyntheticCtcDataset(Dataset):
    """Lazy, deterministic synthetic sequence dataset."""

    def __init__(self, config: CtcDatasetConfig | None = None) -> None:
        self.config = config or CtcDatasetConfig()
        _validate_config(self.config)

    def __len__(self) -> int:
        return self.config.num_examples

    def __getitem__(self, index: int) -> CtcExample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, index]))
        text = _sample_text(rng, self.config)
        audio = _render_audio(rng, text, self.config)
        frames = audio_to_frames(audio, self.config.sample_rate, self.config.frames)
        targets = self.config.vocab.encode(text)
        return CtcExample(
            text=text,
            audio=audio,
            frames=frames,
            targets=targets,
            sample_rate=self.config.sample_rate,
        )


class CachedSyntheticCtcDataset(Dataset):
    """Materialise the synthetic dataset once for fast multi-epoch training."""

    def __init__(self, config: CtcDatasetConfig | None = None) -> None:
        self.base = SyntheticCtcDataset(config)
        self.examples: list[CtcExample] = list(self.base)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> CtcExample:
        return self.examples[index]


def collate_ctc(batch: list[CtcExample]) -> dict[str, torch.Tensor]:
    """Pad a list of CTC examples into a batched tensor dict."""
    frame_tensors = [torch.from_numpy(item.frames).float() for item in batch]
    frame_lengths = torch.tensor([t.numel() for t in frame_tensors], dtype=torch.long)
    padded = pad_sequence(frame_tensors, batch_first=True)
    inputs = padded.unsqueeze(1)  # [B, 1, T]
    target_lengths = torch.tensor([len(item.targets) for item in batch], dtype=torch.long)
    targets = torch.tensor([idx for item in batch for idx in item.targets], dtype=torch.long)
    return {
        "inputs": inputs,
        "input_lengths": frame_lengths,
        "targets": targets,
        "target_lengths": target_lengths,
        "texts": [item.text for item in batch],
    }


def _sample_text(rng: np.random.Generator, config: CtcDatasetConfig) -> str:
    if config.text_pool:
        return _sample_from_pool(rng, config)
    length = int(rng.integers(config.min_chars, config.max_chars + 1))
    chars = [c for c in config.vocab.chars if c != " "]
    out: list[str] = []
    space_index = -1
    for position in range(length):
        out.append(chars[int(rng.integers(0, len(chars)))])
        if (
            position < length - 1
            and position - space_index > 1
            and rng.random() < config.word_probability
        ):
            out.append(" ")
            space_index = position + 1
    return "".join(out)


def _sample_from_pool(rng: np.random.Generator, config: CtcDatasetConfig) -> str:
    text = config.text_pool[int(rng.integers(0, len(config.text_pool)))]
    encoded = config.vocab.encode(text)
    return config.vocab.decode(encoded)


def _render_audio(
    rng: np.random.Generator,
    text: str,
    config: CtcDatasetConfig,
) -> np.ndarray:
    operator_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    channel_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    wpm = _uniform(rng, config.wpm_range)
    freq = _uniform(rng, config.freq_range)
    amplitude = _uniform(rng, config.amplitude_range)
    rise_ms = _uniform(rng, config.rise_ms_range)
    tail_ms = _uniform(rng, config.tail_ms_range)
    element_jitter = _uniform(rng, config.element_jitter_range)
    gap_jitter = _uniform(rng, config.gap_jitter_range)
    dash_dot_ratio = _uniform(rng, config.dash_dot_ratio_range)
    gap_inflation = _uniform(rng, config.gap_inflation_range)
    word_gap_inflation = _uniform(rng, config.word_gap_inflation_range)
    if config.snr_db_range is None:
        channel = ChannelConfig(seed=channel_seed)
    else:
        channel = ChannelConfig(
            snr_db=_uniform(rng, config.snr_db_range),
            seed=channel_seed,
        )
    audio = render(
        text,
        operator=OperatorConfig(
            wpm=wpm,
            element_jitter=element_jitter,
            gap_jitter=gap_jitter,
            dash_dot_ratio=dash_dot_ratio,
            gap_inflation=gap_inflation,
            word_gap_inflation=word_gap_inflation,
            seed=operator_seed,
        ),
        keying=KeyingConfig(rise_ms=rise_ms),
        channel=channel,
        freq=freq,
        sample_rate=config.sample_rate,
        amplitude=amplitude,
        tail_ms=tail_ms,
    )
    pad_lead = int(round(_uniform(rng, config.pad_ms_range) / 1000.0 * config.sample_rate))
    pad_trail = int(round(_uniform(rng, config.pad_ms_range) / 1000.0 * config.sample_rate))
    return np.concatenate(
        [
            np.zeros(pad_lead, dtype=np.float32),
            audio.astype(np.float32, copy=False),
            np.zeros(pad_trail, dtype=np.float32),
        ]
    )


def _validate_config(config: CtcDatasetConfig) -> None:
    if config.num_examples <= 0:
        raise ValueError("num_examples must be positive")
    if config.min_chars < 1:
        raise ValueError("min_chars must be >= 1")
    if config.max_chars < config.min_chars:
        raise ValueError("max_chars must be >= min_chars")
    if not 0.0 <= config.word_probability <= 1.0:
        raise ValueError("word_probability must lie in [0, 1]")
    for name in (
        "wpm_range",
        "freq_range",
        "amplitude_range",
        "rise_ms_range",
        "tail_ms_range",
        "element_jitter_range",
        "gap_jitter_range",
        "dash_dot_ratio_range",
        "gap_inflation_range",
        "word_gap_inflation_range",
        "pad_ms_range",
    ):
        low, high = getattr(config, name)
        if low > high:
            raise ValueError(f"{name} lower bound must be <= upper bound")
    if config.snr_db_range is not None:
        low, high = config.snr_db_range
        if low > high:
            raise ValueError("snr_db_range lower bound must be <= upper bound")


def _uniform(rng: np.random.Generator, value_range: Range) -> float:
    low, high = value_range
    if low == high:
        return float(low)
    return float(rng.uniform(low, high))
