"""Tests for classification metrics."""

from __future__ import annotations

import pytest

from morse_char_recognizer.metrics import classification_report


def test_classification_report_counts_confusions() -> None:
    report = classification_report([0, 0, 1], [0, 1, 1], ("A", "B"))

    assert report.total == 3
    assert report.correct == 2
    assert report.accuracy == pytest.approx(2 / 3)
    assert report.per_class_accuracy == {"A": 0.5, "B": 1.0}
    assert report.top_confusions() == [("A", "B", 1)]
