"""Tests for the small CNN model."""

from __future__ import annotations

import torch

from morse_char_recognizer.model import SmallCnn1d


def test_small_cnn_forward_shape() -> None:
    model = SmallCnn1d(num_classes=36)
    x = torch.zeros(4, 1, 128)

    logits = model(x)

    assert logits.shape == (4, 36)
