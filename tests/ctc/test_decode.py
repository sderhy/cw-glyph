"""Tests for CTC decoders and metrics."""

from __future__ import annotations

import numpy as np
import torch

from morse_char_recognizer.ctc.decode import greedy_decode, greedy_decode_numpy
from morse_char_recognizer.ctc.metrics import aggregate_cer, edit_distance
from morse_char_recognizer.ctc.vocab import CtcVocab


def _one_hot_log_probs(indices: list[int], num_classes: int) -> np.ndarray:
    log_probs = np.full((len(indices), num_classes), -50.0, dtype=np.float32)
    for t, idx in enumerate(indices):
        log_probs[t, idx] = 0.0
    return log_probs


def test_greedy_collapses_repeats_and_blanks():
    vocab = CtcVocab()
    a = vocab.char_to_index("A")
    b = vocab.char_to_index("B")
    blank = vocab.blank_index
    # Pattern: A A blank A B B
    sequence = _one_hot_log_probs([a, a, blank, a, b, b], vocab.num_classes)
    text = greedy_decode_numpy(sequence, vocab)
    assert text == "AAB"


def test_greedy_batched_matches_numpy():
    vocab = CtcVocab()
    a = vocab.char_to_index("A")
    indices = [a, a, vocab.blank_index, a]
    sequence = _one_hot_log_probs(indices, vocab.num_classes)
    batched = torch.from_numpy(np.stack([sequence, sequence], axis=1))  # [T, B=2, V]
    decoded = greedy_decode(batched, torch.tensor([4, 4]), vocab)
    assert decoded == ["AA", "AA"]


def test_input_lengths_truncate():
    vocab = CtcVocab()
    a = vocab.char_to_index("A")
    b = vocab.char_to_index("B")
    sequence = _one_hot_log_probs([a, b, b], vocab.num_classes)
    batched = torch.from_numpy(sequence).unsqueeze(1)  # [T=3, B=1, V]
    short = greedy_decode(batched, torch.tensor([1]), vocab)
    full = greedy_decode(batched, torch.tensor([3]), vocab)
    assert short == ["A"]
    assert full == ["AB"]


def test_edit_distance_basic():
    assert edit_distance("CAT", "CAT") == 0
    assert edit_distance("CAT", "CUT") == 1
    assert edit_distance("CAT", "CATS") == 1
    assert edit_distance("KITTEN", "SITTING") == 3


def test_aggregate_cer():
    result = aggregate_cer([("ABCD", "ABCD"), ("HELLO", "HALLO")])
    assert result.distance == 1
    assert result.reference_length == 9
    assert result.cer == 1 / 9
