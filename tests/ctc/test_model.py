"""Tests for the CRNN CTC model."""

from __future__ import annotations

import torch

from morse_char_recognizer.ctc.model import CrnnCtcConfig, CrnnCtcRecognizer
from morse_char_recognizer.ctc.vocab import CtcVocab


def test_forward_shape_is_time_major():
    vocab = CtcVocab()
    model = CrnnCtcRecognizer(CrnnCtcConfig(num_classes=vocab.num_classes))
    batch, time = 3, 400
    inputs = torch.randn(batch, 1, time)
    log_probs = model(inputs)
    assert log_probs.ndim == 3
    assert log_probs.shape[0] == time // model.total_stride
    assert log_probs.shape[1] == batch
    assert log_probs.shape[2] == vocab.num_classes


def test_output_log_probs_sum_to_one():
    vocab = CtcVocab()
    model = CrnnCtcRecognizer(CrnnCtcConfig(num_classes=vocab.num_classes))
    log_probs = model(torch.randn(2, 1, 200))
    probs = log_probs.exp().sum(dim=-1)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-5)


def test_output_lengths_reflect_stride():
    vocab = CtcVocab()
    model = CrnnCtcRecognizer(CrnnCtcConfig(num_classes=vocab.num_classes))
    lengths = torch.tensor([400, 100, 1], dtype=torch.long)
    out_lengths = model.output_lengths(lengths)
    assert out_lengths.tolist() == [100, 25, 1]
