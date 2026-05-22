"""Tests for livetest helpers."""

from __future__ import annotations

import numpy as np
import pytest

from morse_char_recognizer.live import (
    compare_best_substring,
    compare_text,
    normalize_copy_text,
    slice_audio,
    window_starts,
)


def test_normalize_copy_text_maps_equivalent_tokens() -> None:
    assert normalize_copy_text("A <BT> B <AR>") == "A=B+"


def test_compare_text_uses_edit_distance_accuracy() -> None:
    comparison = compare_text("ABC", "ADC")

    assert comparison.distance == 1
    assert comparison.substitutions == 1
    assert comparison.insertions == 0
    assert comparison.deletions == 0
    assert comparison.cer == pytest.approx(1 / 3)
    assert comparison.accuracy == pytest.approx(2 / 3)


def test_compare_text_breaks_down_insertions_and_deletions() -> None:
    inserted = compare_text("ABC", "ABXC")
    deleted = compare_text("ABC", "AC")

    assert inserted.distance == 1
    assert inserted.insertions == 1
    assert deleted.distance == 1
    assert deleted.deletions == 1


def test_compare_best_substring_matches_window_inside_long_reference() -> None:
    comparison = compare_best_substring("HELLO CQ TEST 599", "CQ TEST")

    assert comparison.reference == "CQTEST"
    assert comparison.distance == 0
    assert comparison.accuracy == 1.0


def test_slice_audio_returns_offset() -> None:
    audio = np.arange(100, dtype=np.float32)
    sliced, offset = slice_audio(audio, 10, 2.0, 3.0)

    assert offset == 2.0
    assert sliced.tolist() == list(range(20, 50))


def test_window_starts_covers_long_audio() -> None:
    assert window_starts(125.0, 60.0, 60.0) == [0.0, 60.0, 120.0]
