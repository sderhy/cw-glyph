"""Character error rate and aggregate evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CerResult:
    """Aggregate character-error-rate over a set of (reference, hypothesis) pairs."""

    distance: int
    reference_length: int

    @property
    def cer(self) -> float:
        if self.reference_length == 0:
            return 0.0
        return self.distance / self.reference_length

    @property
    def accuracy(self) -> float:
        return max(0.0, 1.0 - self.cer)


def edit_distance(reference: str, hypothesis: str) -> int:
    """Levenshtein distance between two strings."""
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp_char in enumerate(hypothesis, start=1):
            cost = 0 if ref_char == hyp_char else 1
            current[j] = min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + cost,
            )
        previous = current
    return previous[-1]


def aggregate_cer(pairs: list[tuple[str, str]]) -> CerResult:
    """Return aggregate CER over a list of ``(reference, hypothesis)`` pairs."""
    total_distance = 0
    total_reference = 0
    for reference, hypothesis in pairs:
        total_distance += edit_distance(reference, hypothesis)
        total_reference += len(reference)
    return CerResult(distance=total_distance, reference_length=total_reference)
