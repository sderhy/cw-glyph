"""Tests for CNN checkpoint inference helpers."""

from __future__ import annotations

import numpy as np
import torch

from morse_char_recognizer.inference import filter_short_segment_predictions, predict_envelopes
from morse_char_recognizer.model import SmallCnn1d
from morse_char_recognizer.segment import Region


def test_predict_envelopes_returns_labels_and_scores() -> None:
    model = SmallCnn1d(num_classes=2)
    envelopes = np.zeros((3, 128), dtype=np.float32)

    labels, scores = predict_envelopes(model, envelopes, device=torch.device("cpu"))

    assert labels.shape == (3,)
    assert scores.shape == (3,)


def test_predict_envelopes_can_mask_allowed_classes() -> None:
    model = SmallCnn1d(num_classes=2)
    envelopes = np.zeros((3, 128), dtype=np.float32)

    labels, _scores = predict_envelopes(
        model,
        envelopes,
        classes=("E", "T"),
        allowed_classes=("T",),
        device=torch.device("cpu"),
    )

    assert labels.tolist() == [1, 1, 1]


def test_filter_short_segment_predictions_keeps_confident_one_element_classes() -> None:
    predictions = [
        ("E", 0.95, Region(0, 60)),
        ("I", 0.95, Region(100, 160)),
        ("A", 0.40, Region(200, 360)),
    ]

    filtered = filter_short_segment_predictions(
        predictions,
        unit_samples=50,
        min_segment_units=2.2,
        short_class_min_segment_units=0.8,
        short_class_min_score=0.8,
    )

    assert [char for char, _score, _segment in filtered] == ["E", "A"]
