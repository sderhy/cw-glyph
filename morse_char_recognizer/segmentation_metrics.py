"""Segmentation metrics for Morse character candidate regions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from morse_char_recognizer.segment import Region


@dataclass(frozen=True)
class SegmentationReport:
    """Counts and boundary errors for predicted character regions."""

    true_count: int
    predicted_count: int
    matched: int
    missed: int
    split: int
    merge: int
    false_positive: int
    mean_boundary_error_ms: float

    @property
    def recall(self) -> float:
        return 0.0 if self.true_count == 0 else self.matched / self.true_count

    @property
    def precision(self) -> float:
        return 0.0 if self.predicted_count == 0 else self.matched / self.predicted_count


def evaluate_segmentation(
    truth: list[Region],
    predicted: list[Region],
    sample_rate: int,
    *,
    min_overlap_ratio: float = 0.2,
) -> SegmentationReport:
    """Evaluate predicted character regions against reference regions.

    A prediction overlaps a reference when their intersection covers at least
    ``min_overlap_ratio`` of the shorter region. This catches gross missed,
    split, and merge cases without requiring sample-perfect boundaries.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    overlaps = _overlap_matrix(truth, predicted, min_overlap_ratio)
    true_hits = [list(np.flatnonzero(row)) for row in overlaps]
    pred_hits = [list(np.flatnonzero(overlaps[:, j])) for j in range(len(predicted))]

    missed = sum(1 for hits in true_hits if not hits)
    split = sum(1 for hits in true_hits if len(hits) > 1)
    merge = sum(1 for hits in pred_hits if len(hits) > 1)
    false_positive = sum(1 for hits in pred_hits if not hits)

    matched_pairs = [
        (i, hits[0])
        for i, hits in enumerate(true_hits)
        if len(hits) == 1 and len(pred_hits[hits[0]]) == 1
    ]
    errors = [
        (abs(truth[i].start - predicted[j].start) + abs(truth[i].end - predicted[j].end)) / 2
        for i, j in matched_pairs
    ]
    mean_error_ms = 0.0
    if errors:
        mean_error_ms = float(np.mean(errors) / sample_rate * 1000.0)

    return SegmentationReport(
        true_count=len(truth),
        predicted_count=len(predicted),
        matched=len(matched_pairs),
        missed=missed,
        split=split,
        merge=merge,
        false_positive=false_positive,
        mean_boundary_error_ms=mean_error_ms,
    )


def _overlap_matrix(
    truth: list[Region],
    predicted: list[Region],
    min_overlap_ratio: float,
) -> np.ndarray:
    matrix = np.zeros((len(truth), len(predicted)), dtype=bool)
    for i, true_region in enumerate(truth):
        for j, pred_region in enumerate(predicted):
            overlap = max(
                0,
                min(true_region.end, pred_region.end)
                - max(true_region.start, pred_region.start),
            )
            shorter = max(1, min(true_region.duration, pred_region.duration))
            matrix[i, j] = overlap / shorter >= min_overlap_ratio
    return matrix
