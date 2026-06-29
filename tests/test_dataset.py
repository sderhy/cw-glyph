"""Tests for lazy synthetic Morse-character datasets."""

from __future__ import annotations

import numpy as np
import pytest

from morse_char_recognizer.classes import CLASS_PRESETS, REAL_CLASSES, parse_class_spec
from morse_char_recognizer.dataset import (
    SyntheticDatasetConfig,
    SyntheticMorseCharacterDataset,
)
from morse_char_recognizer.features import EnvelopeConfig


def test_dataset_is_lazy_and_indexable() -> None:
    cfg = SyntheticDatasetConfig(
        classes=("A", "B"),
        samples_per_class=2,
        envelope=EnvelopeConfig(length=64),
        seed=123,
    )
    dataset = SyntheticMorseCharacterDataset(cfg)

    assert len(dataset) == 4

    example = dataset[0]
    assert example.char == "A"
    assert example.label == 0
    assert example.audio.dtype == np.float32
    assert example.envelope.shape == (64,)
    assert example.envelope.dtype == np.float32


def test_dataset_is_reproducible_by_index() -> None:
    cfg = SyntheticDatasetConfig(
        classes=("E",),
        samples_per_class=1,
        envelope=EnvelopeConfig(length=64),
        seed=7,
    )
    dataset = SyntheticMorseCharacterDataset(cfg)

    a = dataset[0]
    b = dataset[0]

    assert np.array_equal(a.audio, b.audio)
    assert np.array_equal(a.envelope, b.envelope)
    assert a.params == b.params


def test_real_class_preset_contains_punctuation_and_prosigns() -> None:
    assert "?" in REAL_CLASSES
    assert "'" in REAL_CLASSES
    assert "<KN>" in REAL_CLASSES
    assert "<UR>" in REAL_CLASSES
    assert "<BT>" not in REAL_CLASSES
    assert "<AR>" not in REAL_CLASSES


def test_copy_prosign_preset_excludes_run_ons() -> None:
    classes = CLASS_PRESETS["copy-prosign"]

    assert "<KN>" in classes
    assert "<UR>" not in classes
    assert "<BK>" not in classes


def test_parse_class_spec_supports_multi_character_tokens() -> None:
    assert parse_class_spec("A,B,?,<KN>,<UR>") == ("A", "B", "?", "<KN>", "<UR>")


def test_dataset_rejects_unsupported_morse_tokens() -> None:
    with pytest.raises(ValueError):
        SyntheticMorseCharacterDataset(SyntheticDatasetConfig(classes=("%",)))


def test_dataset_can_synthesize_prosign_class() -> None:
    dataset = SyntheticMorseCharacterDataset(
        SyntheticDatasetConfig(
            classes=("<KN>",),
            samples_per_class=1,
            envelope=EnvelopeConfig(length=64),
        )
    )

    example = dataset[0]

    assert example.char == "<KN>"
    assert example.envelope.shape == (64,)


def test_dataset_records_stroke_amplitude_jitter() -> None:
    dataset = SyntheticMorseCharacterDataset(
        SyntheticDatasetConfig(
            classes=("B",),
            samples_per_class=1,
            envelope=EnvelopeConfig(length=64),
            stroke_amplitude_jitter_range=(0.25, 0.25),
            seed=11,
        )
    )

    example = dataset[0]
    again = dataset[0]

    assert example.params["stroke_amplitude_jitter"] == pytest.approx(0.25)
    assert example.params["keying_seed"] == again.params["keying_seed"]
    assert np.array_equal(example.audio, again.audio)
