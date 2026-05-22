"""Evaluation helpers for character-level Morse recognition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationReport:
    """Compact classification metrics."""

    classes: tuple[str, ...]
    confusion: np.ndarray

    @property
    def total(self) -> int:
        return int(np.sum(self.confusion))

    @property
    def correct(self) -> int:
        return int(np.trace(self.confusion))

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    @property
    def per_class_accuracy(self) -> dict[str, float]:
        out = {}
        for index, char in enumerate(self.classes):
            total = int(np.sum(self.confusion[index]))
            out[char] = 0.0 if total == 0 else float(self.confusion[index, index] / total)
        return out

    def top_confusions(self, limit: int = 10) -> list[tuple[str, str, int]]:
        """Return the most frequent true->predicted mistakes."""
        mistakes: list[tuple[str, str, int]] = []
        for true_index, true_char in enumerate(self.classes):
            for pred_index, pred_char in enumerate(self.classes):
                if true_index == pred_index:
                    continue
                count = int(self.confusion[true_index, pred_index])
                if count > 0:
                    mistakes.append((true_char, pred_char, count))
        mistakes.sort(key=lambda item: item[2], reverse=True)
        return mistakes[:limit]


def classification_report(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    classes: tuple[str, ...],
) -> ClassificationReport:
    """Build a confusion matrix and derived metrics."""
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if true.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {true.shape} and {pred.shape}"
        )

    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for t, p in zip(true, pred):
        if 0 <= t < len(classes) and 0 <= p < len(classes):
            confusion[int(t), int(p)] += 1
    return ClassificationReport(classes=classes, confusion=confusion)
