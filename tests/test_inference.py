"""Tests for CNN checkpoint inference helpers."""

from __future__ import annotations

import numpy as np
import torch

from morse_char_recognizer.inference import predict_envelopes
from morse_char_recognizer.model import SmallCnn1d


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
