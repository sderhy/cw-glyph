"""Synthetic isolated-character dataset helpers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from morse_char_recognizer.classes import DEFAULT_CLASSES, morse_code_for_class
from morse_char_recognizer.features import (
    EnvelopeConfig,
    extract_envelope,
    extract_unit_scaled_envelope,
)
from morse_synth.channel import ChannelConfig
from morse_synth.core import render
from morse_synth.keying import KeyingConfig
from morse_synth.operator import OperatorConfig

Range = tuple[float, float]


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    """Parameter ranges for synthetic isolated Morse-character examples."""

    classes: tuple[str, ...] = DEFAULT_CLASSES
    samples_per_class: int = 100
    sample_rate: int = 8000
    envelope: EnvelopeConfig = field(default_factory=EnvelopeConfig)
    seed: int | None = 0
    wpm_range: Range = (12.0, 30.0)
    freq_range: Range = (500.0, 750.0)
    amplitude_range: Range = (0.35, 0.9)
    rise_ms_range: Range = (2.0, 8.0)
    tail_ms_range: Range = (35.0, 90.0)
    element_jitter_range: Range = (0.0, 0.12)
    gap_jitter_range: Range = (0.0, 0.12)
    dash_dot_ratio_range: Range = (2.7, 3.4)
    gap_inflation_range: Range = (0.9, 1.25)
    leading_silence_ms_range: Range = (0.0, 80.0)
    trailing_silence_ms_range: Range = (0.0, 80.0)
    snr_db_range: Range | None = None


@dataclass(frozen=True)
class CharacterExample:
    """One synthesized character example and its extracted feature vector."""

    char: str
    label: int
    audio: np.ndarray
    envelope: np.ndarray
    sample_rate: int
    params: dict[str, float | int | None]


class SyntheticMorseCharacterDataset(Sequence[CharacterExample]):
    """Deterministic, lazy synthetic dataset for isolated Morse characters.

    ``__getitem__`` synthesizes the requested example on demand. Nothing is
    generated or written during construction.
    """

    def __init__(self, config: SyntheticDatasetConfig | None = None) -> None:
        self.config = config or SyntheticDatasetConfig()
        _validate_config(self.config)
        self.classes = tuple(self.config.classes)
        self.class_to_index = {char: i for i, char in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.classes) * self.config.samples_per_class

    def __getitem__(self, index: int) -> CharacterExample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        class_index = index % len(self.classes)
        char = self.classes[class_index]
        rng = _rng_for_index(self.config.seed, index)
        return synthesize_character_example(char, class_index, rng, self.config)

    def __iter__(self) -> Iterator[CharacterExample]:
        for index in range(len(self)):
            yield self[index]


def synthesize_character_example(
    char: str,
    label: int,
    rng: np.random.Generator,
    config: SyntheticDatasetConfig | None = None,
) -> CharacterExample:
    """Synthesize one in-memory character example."""
    config = config or SyntheticDatasetConfig()
    _validate_config(config)

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

    channel = ChannelConfig(seed=channel_seed)
    snr_db: float | None = None
    if config.snr_db_range is not None:
        snr_db = _uniform(rng, config.snr_db_range)
        channel = ChannelConfig(snr_db=snr_db, seed=channel_seed)

    audio = render(
        char,
        operator=OperatorConfig(
            wpm=wpm,
            element_jitter=element_jitter,
            gap_jitter=gap_jitter,
            dash_dot_ratio=dash_dot_ratio,
            gap_inflation=gap_inflation,
            seed=operator_seed,
        ),
        keying=KeyingConfig(rise_ms=rise_ms),
        channel=channel,
        freq=freq,
        sample_rate=config.sample_rate,
        amplitude=amplitude,
        tail_ms=tail_ms,
    )
    audio = _add_edge_silence(audio, config.sample_rate, rng, config)
    if config.envelope.scale_mode == "unit":
        unit_samples = int(round(1.2 / wpm * config.sample_rate))
        envelope = extract_unit_scaled_envelope(
            audio,
            config.sample_rate,
            unit_samples,
            config.envelope,
        )
    else:
        envelope = extract_envelope(audio, config.sample_rate, config.envelope)

    return CharacterExample(
        char=char,
        label=label,
        audio=audio.astype(np.float32, copy=False),
        envelope=envelope,
        sample_rate=config.sample_rate,
        params={
            "wpm": wpm,
            "freq": freq,
            "amplitude": amplitude,
            "rise_ms": rise_ms,
            "tail_ms": tail_ms,
            "element_jitter": element_jitter,
            "gap_jitter": gap_jitter,
            "dash_dot_ratio": dash_dot_ratio,
            "gap_inflation": gap_inflation,
            "snr_db": snr_db,
            "operator_seed": operator_seed,
            "channel_seed": channel_seed,
        },
    )


def _validate_config(config: SyntheticDatasetConfig) -> None:
    if not config.classes:
        raise ValueError("classes must not be empty")
    unknown = [token for token in config.classes if not morse_code_for_class(token)]
    if unknown:
        raise ValueError(f"classes contain unsupported Morse tokens: {unknown}")
    if config.samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    if config.sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
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
        "leading_silence_ms_range",
        "trailing_silence_ms_range",
    ):
        _validate_range(name, getattr(config, name))
    if config.snr_db_range is not None:
        _validate_range("snr_db_range", config.snr_db_range)


def _validate_range(name: str, value: Range) -> None:
    low, high = value
    if low > high:
        raise ValueError(f"{name} lower bound must be <= upper bound")


def _rng_for_index(seed: int | None, index: int) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence([seed, index]))


def _uniform(rng: np.random.Generator, value_range: Range) -> float:
    low, high = value_range
    if low == high:
        return float(low)
    return float(rng.uniform(low, high))


def _add_edge_silence(
    audio: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    config: SyntheticDatasetConfig,
) -> np.ndarray:
    lead_ms = _uniform(rng, config.leading_silence_ms_range)
    trail_ms = _uniform(rng, config.trailing_silence_ms_range)
    lead = np.zeros(int(round(lead_ms / 1000.0 * sample_rate)), dtype=np.float32)
    trail = np.zeros(int(round(trail_ms / 1000.0 * sample_rate)), dtype=np.float32)
    return np.concatenate([lead, audio.astype(np.float32, copy=False), trail])
