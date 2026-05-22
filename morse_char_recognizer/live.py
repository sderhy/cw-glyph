"""Helpers for real-world labelled Morse audio tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import welch


@dataclass(frozen=True)
class TextComparison:
    """Edit-distance comparison between normalized reference and hypothesis."""

    reference: str
    hypothesis: str
    distance: int
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0

    @property
    def reference_length(self) -> int:
        return len(self.reference)

    @property
    def hypothesis_length(self) -> int:
        return len(self.hypothesis)

    @property
    def accuracy(self) -> float:
        if not self.reference:
            return 1.0 if not self.hypothesis else 0.0
        return max(0.0, 1.0 - self.distance / len(self.reference))

    @property
    def cer(self) -> float:
        if not self.reference:
            return 0.0 if not self.hypothesis else 1.0
        return self.distance / len(self.reference)


def slice_audio(
    audio: np.ndarray,
    sample_rate: int,
    start_s: float,
    duration_s: float | None,
) -> tuple[np.ndarray, float]:
    """Return an audio slice and its actual start offset in seconds."""
    start = max(0, int(round(start_s * sample_rate)))
    if duration_s is None or duration_s <= 0:
        end = audio.size
    else:
        end = min(audio.size, start + int(round(duration_s * sample_rate)))
    return audio[start:end], start / sample_rate


def window_starts(duration_s: float, window_s: float, hop_s: float) -> list[float]:
    """Generate start times for fixed-duration evaluation windows."""
    if duration_s <= 0:
        return []
    if window_s <= 0:
        return [0.0]
    hop = hop_s if hop_s > 0 else window_s
    starts: list[float] = []
    current = 0.0
    while current < duration_s:
        starts.append(current)
        current += hop
        if current + 0.001 >= duration_s:
            break
    return starts


def estimate_carrier_hz(audio: np.ndarray, sample_rate: int, f_min: float, f_max: float) -> float:
    """Estimate dominant CW tone frequency in a bounded band."""
    if audio.size == 0:
        return 600.0
    nperseg = min(audio.size, 4096)
    frequencies, power = welch(audio, fs=sample_rate, nperseg=nperseg)
    mask = (frequencies >= f_min) & (frequencies <= f_max)
    if not mask.any():
        return 600.0
    return float(frequencies[mask][int(np.argmax(power[mask]))])


def default_label_path(wav_path: Path) -> Path | None:
    """Return sibling .txt label path when it exists."""
    candidate = wav_path.with_suffix(".txt")
    return candidate if candidate.exists() else None


def read_label(path: Path | None) -> str:
    """Read a label file as a single whitespace-normalized line."""
    if path is None or not path.exists():
        return ""
    return " ".join(path.read_text(encoding="utf-8", errors="replace").split())


def normalize_copy_text(text: str) -> str:
    """Normalize copied Morse text for approximate character-level comparison."""
    normalized = text.upper()
    normalized = normalized.replace("<BT>", "=")
    normalized = normalized.replace("<AR>", "+")
    normalized = normalized.replace(" % ", " 0/0 ")
    normalized = "".join(ch for ch in normalized if not ch.isspace())
    return normalized


def compare_text(reference: str, hypothesis: str) -> TextComparison:
    """Compare normalized texts with Levenshtein distance."""
    ref = normalize_copy_text(reference)
    hyp = normalize_copy_text(hypothesis)
    distance, substitutions, insertions, deletions = _levenshtein_breakdown(ref, hyp)
    return TextComparison(ref, hyp, distance, substitutions, insertions, deletions)


def compare_best_substring(reference: str, hypothesis: str) -> TextComparison:
    """Compare hypothesis with the best similarly-sized substring of reference."""
    ref = normalize_copy_text(reference)
    hyp = normalize_copy_text(hypothesis)
    if not ref or not hyp:
        distance, substitutions, insertions, deletions = _levenshtein_breakdown(ref, hyp)
        return TextComparison(ref, hyp, distance, substitutions, insertions, deletions)

    best_distance: int | None = None
    best_ref = ref
    best_breakdown = (0, 0, 0)
    min_len = max(0, min(len(ref), len(hyp) - max(3, len(hyp) // 4)))
    max_len = min(len(ref), len(hyp) + max(3, len(hyp) // 4))
    step = max(1, len(hyp) // 10)
    for length in range(min_len, max_len + 1):
        if length == 0:
            continue
        starts = set(range(0, len(ref) - length + 1, step))
        if len(ref) >= length:
            starts.add(len(ref) - length)
        for start in starts:
            candidate = ref[start : start + length]
            distance, substitutions, insertions, deletions = _levenshtein_breakdown(candidate, hyp)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_ref = candidate
                best_breakdown = (substitutions, insertions, deletions)
    return TextComparison(best_ref, hyp, best_distance or 0, *best_breakdown)


def _levenshtein(a: str, b: str) -> int:
    return _levenshtein_breakdown(a, b)[0]


def _levenshtein_breakdown(a: str, b: str) -> tuple[int, int, int, int]:
    if a == b:
        return 0, 0, 0, 0
    if not a:
        return len(b), 0, len(b), 0
    if not b:
        return len(a), 0, 0, len(a)

    # Each cell stores (distance, substitutions, insertions, deletions).
    previous = [(j, 0, j, 0) for j in range(len(b) + 1)]
    for i, ca in enumerate(a, start=1):
        current = [(i, 0, 0, i)]
        for j, cb in enumerate(b, start=1):
            delete = _add_op(previous[j], deletions=1)
            insert = _add_op(current[j - 1], insertions=1)
            if ca == cb:
                substitute = previous[j - 1]
            else:
                substitute = _add_op(previous[j - 1], substitutions=1)
            current.append(min(delete, insert, substitute, key=_alignment_sort_key))
        previous = current
    return previous[-1]


def _add_op(
    value: tuple[int, int, int, int],
    *,
    substitutions: int = 0,
    insertions: int = 0,
    deletions: int = 0,
) -> tuple[int, int, int, int]:
    distance, subs, ins, dels = value
    return (
        distance + substitutions + insertions + deletions,
        subs + substitutions,
        ins + insertions,
        dels + deletions,
    )


def _alignment_sort_key(value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    distance, substitutions, insertions, deletions = value
    return distance, substitutions, insertions, deletions
