"""Tests for segmentation metrics."""

from __future__ import annotations

import pytest

from morse_char_recognizer.segment import Region
from morse_char_recognizer.segmentation_metrics import evaluate_segmentation


def test_evaluate_segmentation_counts_clean_matches() -> None:
    report = evaluate_segmentation(
        [Region(0, 100), Region(200, 300)],
        [Region(2, 102), Region(198, 298)],
        1000,
    )

    assert report.matched == 2
    assert report.missed == 0
    assert report.split == 0
    assert report.merge == 0
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.mean_boundary_error_ms == pytest.approx(2.0)


def test_evaluate_segmentation_counts_split_and_merge() -> None:
    split = evaluate_segmentation([Region(0, 100)], [Region(0, 50), Region(50, 100)], 1000)
    merge = evaluate_segmentation([Region(0, 100), Region(120, 220)], [Region(0, 220)], 1000)

    assert split.split == 1
    assert merge.merge == 1
