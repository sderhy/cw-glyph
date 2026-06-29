"""Tests for envelope feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

from morse_char_recognizer.features import (
    EnvelopeConfig,
    extract_envelope,
    extract_unit_scaled_envelope,
)
from morse_synth.core import synthesize


def test_extract_envelope_returns_fixed_float32_vector() -> None:
    audio = synthesize("A", sample_rate=8000)
    envelope = extract_envelope(audio, 8000, EnvelopeConfig(length=128))

    assert envelope.dtype == np.float32
    assert envelope.shape == (128,)
    assert np.max(envelope) == pytest.approx(1.0)
    assert np.min(envelope) >= 0.0


def test_extract_envelope_handles_silence() -> None:
    envelope = extract_envelope(np.zeros(100, dtype=np.float32), 8000, EnvelopeConfig(length=64))

    assert envelope.shape == (64,)
    assert np.all(envelope == 0.0)


def test_extract_envelope_rejects_non_mono_audio() -> None:
    with pytest.raises(ValueError):
        extract_envelope(np.zeros((2, 100), dtype=np.float32), 8000)


def test_extract_unit_scaled_envelope_returns_fixed_vector() -> None:
    audio = synthesize("A", sample_rate=8000)
    envelope = extract_unit_scaled_envelope(audio, 8000, 480, EnvelopeConfig(length=96))

    assert envelope.shape == (96,)
    assert envelope.dtype == np.float32
    assert np.max(envelope) == pytest.approx(1.0)


def test_extract_envelope_can_binarize_after_normalization() -> None:
    audio = synthesize("A", sample_rate=8000)
    envelope = extract_envelope(
        audio,
        8000,
        EnvelopeConfig(length=128, binarize_threshold=0.35),
    )

    assert set(np.unique(envelope)).issubset({0.0, 1.0})
    assert np.any(envelope == 1.0)


def test_extract_envelope_rejects_invalid_binarize_threshold() -> None:
    with pytest.raises(ValueError):
        extract_envelope(
            np.ones(100, dtype=np.float32),
            8000,
            EnvelopeConfig(binarize_threshold=1.5),
        )
