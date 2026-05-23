"""Tests for the CTC vocabulary."""

from __future__ import annotations

import pytest

from morse_char_recognizer.ctc.vocab import BLANK_INDEX, CTC_CHARS, CtcVocab


def test_blank_index_is_zero():
    vocab = CtcVocab()
    assert vocab.blank_index == BLANK_INDEX == 0


def test_num_classes_counts_blank():
    vocab = CtcVocab()
    assert vocab.num_classes == len(CTC_CHARS) + 1


def test_encode_decode_roundtrip():
    vocab = CtcVocab()
    text = "CQ DE F4ABC"
    indices = vocab.encode(text)
    assert vocab.decode(indices) == "CQ DE F4ABC"


def test_encode_drops_unknown():
    vocab = CtcVocab()
    indices = vocab.encode("hello, world!")
    assert vocab.decode(indices) == "HELLO WORLD"


def test_encode_collapses_internal_whitespace():
    vocab = CtcVocab()
    indices = vocab.encode("  A   B  ")
    assert vocab.decode(indices) == "A B"


def test_encode_unknown_character_raises():
    vocab = CtcVocab()
    with pytest.raises(ValueError):
        vocab.char_to_index("é")


def test_index_to_char_blank_is_empty():
    vocab = CtcVocab()
    assert vocab.index_to_char(0) == ""
