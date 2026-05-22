"""Tests for CNN segment prediction helpers."""

from __future__ import annotations

import numpy as np
import torch

from morse_char_recognizer.features import EnvelopeConfig
from morse_char_recognizer.inference import join_segment_predictions, predict_audio_segments
from morse_char_recognizer.model import SmallCnn1d
from morse_char_recognizer.segment import Region


def test_predict_audio_segments_handles_silence() -> None:
    model = SmallCnn1d(num_classes=2)
    predictions = predict_audio_segments(
        model,
        np.zeros(256, dtype=np.float32),
        8000,
        classes=("E", "T"),
        envelope=EnvelopeConfig(length=64),
        device=torch.device("cpu"),
    )

    assert predictions == []


def test_join_segment_predictions_inserts_word_gap() -> None:
    text = join_segment_predictions(
        [
            ("C", 0.9, Region(0, 100)),
            ("Q", 0.9, Region(140, 240)),
            ("T", 0.9, Region(600, 700)),
        ],
        sample_rate=1000,
        word_gap_ms=250.0,
    )

    assert text == "CQ T"


def test_join_segment_predictions_can_use_unit_gap() -> None:
    text = join_segment_predictions(
        [
            ("C", 0.9, Region(0, 100)),
            ("Q", 0.9, Region(140, 240)),
            ("T", 0.9, Region(600, 700)),
        ],
        sample_rate=1000,
        unit_samples=80,
        word_gap_units=4.0,
    )

    assert text == "CQ T"
