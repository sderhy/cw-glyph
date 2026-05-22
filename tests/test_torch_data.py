"""Tests for PyTorch dataset adapters."""

from __future__ import annotations

import torch

from morse_char_recognizer.dataset import SyntheticDatasetConfig
from morse_char_recognizer.features import EnvelopeConfig
from morse_char_recognizer.torch_data import TorchMorseCharacterDataset


def test_torch_dataset_returns_tensor_pair() -> None:
    dataset = TorchMorseCharacterDataset(
        SyntheticDatasetConfig(
            classes=("E",),
            samples_per_class=1,
            envelope=EnvelopeConfig(length=64),
        )
    )

    x, y = dataset[0]

    assert x.shape == (1, 64)
    assert x.dtype == torch.float32
    assert y.item() == 0
