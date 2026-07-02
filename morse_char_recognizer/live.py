"""Helpers for real-world labelled Morse audio tests."""

from __future__ import annotations

from collections.abc import Sequence
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
    reference_units: int | None = None
    hypothesis_units: int | None = None

    @property
    def reference_length(self) -> int:
        return self.reference_units if self.reference_units is not None else len(self.reference)

    @property
    def hypothesis_length(self) -> int:
        return self.hypothesis_units if self.hypothesis_units is not None else len(self.hypothesis)

    @property
    def accuracy(self) -> float:
        if self.reference_length == 0:
            return 1.0 if self.hypothesis_length == 0 else 0.0
        return max(0.0, 1.0 - self.distance / self.reference_length)

    @property
    def cer(self) -> float:
        if self.reference_length == 0:
            return 0.0 if self.hypothesis_length == 0 else 1.0
        return self.distance / self.reference_length


@dataclass(frozen=True)
class AlignmentOp:
    """One normalized edit operation between reference and hypothesis."""

    op: str
    reference: str
    hypothesis: str
    reference_index: int
    hypothesis_index: int


@dataclass(frozen=True)
class CarrierFrame:
    """Dominant tone estimate for one short audio window."""

    start_s: float
    end_s: float
    center_hz: float
    power: float


@dataclass(frozen=True)
class CarrierSegment:
    """Piecewise-stable carrier estimate over a larger audio interval."""

    start_s: float
    end_s: float
    center_hz: float
    frames: int


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


def estimate_carrier_track(
    audio: np.ndarray,
    sample_rate: int,
    f_min: float,
    f_max: float,
    *,
    window_s: float = 4.0,
    hop_s: float = 2.0,
) -> list[CarrierFrame]:
    """Estimate dominant CW tone frequency in short overlapping windows."""
    if audio.size == 0:
        return []
    if window_s <= 0:
        raise ValueError(f"window_s must be positive, got {window_s}")
    if hop_s <= 0:
        raise ValueError(f"hop_s must be positive, got {hop_s}")

    window = max(1, int(round(window_s * sample_rate)))
    hop = max(1, int(round(hop_s * sample_rate)))
    starts = list(range(0, max(1, audio.size - window + 1), hop))
    if not starts or starts[-1] + window < audio.size:
        starts.append(max(0, audio.size - window))

    frames: list[CarrierFrame] = []
    for start in starts:
        end = min(audio.size, start + window)
        chunk = audio[start:end]
        if chunk.size == 0:
            continue
        nperseg = min(chunk.size, 4096)
        frequencies, power = welch(chunk, fs=sample_rate, nperseg=nperseg)
        mask = (frequencies >= f_min) & (frequencies <= f_max)
        if not mask.any():
            center = estimate_carrier_hz(chunk, sample_rate, f_min, f_max)
            peak_power = 0.0
        else:
            band_frequencies = frequencies[mask]
            band_power = power[mask]
            peak = int(np.argmax(band_power))
            center = float(band_frequencies[peak])
            peak_power = float(band_power[peak])
        frames.append(
            CarrierFrame(
                start / sample_rate,
                end / sample_rate,
                center,
                peak_power,
            )
        )
    return frames


def carrier_track_is_stable(
    frames: Sequence[CarrierFrame],
    *,
    tolerance_hz: float = 50.0,
) -> bool:
    """Return True when a carrier track stays within a narrow frequency span."""
    if not frames:
        return True
    centers = np.array([frame.center_hz for frame in frames], dtype=np.float32)
    return float(np.percentile(centers, 90) - np.percentile(centers, 10)) <= tolerance_hz


def carrier_segments_from_track(
    frames: Sequence[CarrierFrame],
    audio_duration_s: float,
    *,
    jump_hz: float = 60.0,
    min_segment_s: float = 6.0,
) -> list[CarrierSegment]:
    """Collapse a carrier track into piecewise-stable frequency segments."""
    if not frames:
        return []
    if jump_hz <= 0:
        raise ValueError(f"jump_hz must be positive, got {jump_hz}")
    if min_segment_s < 0:
        raise ValueError(f"min_segment_s must be non-negative, got {min_segment_s}")

    centers = np.array([frame.center_hz for frame in frames], dtype=np.float32)
    smoothed = _median_smooth(centers, width=3)
    groups: list[tuple[int, int]] = []
    start = 0
    current_values = [float(smoothed[0])]
    for index in range(1, len(frames)):
        current_center = float(np.median(current_values))
        if abs(float(smoothed[index]) - current_center) > jump_hz:
            groups.append((start, index))
            start = index
            current_values = [float(smoothed[index])]
        else:
            current_values.append(float(smoothed[index]))
    groups.append((start, len(frames)))

    segments = _segments_from_groups(frames, groups, centers, audio_duration_s)
    return _merge_short_carrier_segments(segments, min_segment_s=min_segment_s, jump_hz=jump_hz)


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
    return "".join(tokenize_copy_text(text))


def tokenize_copy_text(text: str) -> list[str]:
    """Normalize copied Morse text into glyph tokens for comparison."""
    normalized = text.upper().replace(" % ", " 0/0 ")
    tokens: list[str] = []
    index = 0
    while index < len(normalized):
        char = normalized[index]
        if char.isspace():
            index += 1
            continue
        if char == "<":
            end = normalized.find(">", index + 1)
            if end >= 0:
                token = normalized[index : end + 1]
                tokens.append(_canonical_token(token))
                index = end + 1
                continue
        tokens.append(_canonical_token(char))
        index += 1
    return tokens


def compare_text(reference: str, hypothesis: str) -> TextComparison:
    """Compare normalized texts with Levenshtein distance."""
    ref_tokens = tokenize_copy_text(reference)
    hyp_tokens = tokenize_copy_text(hypothesis)
    ref = "".join(ref_tokens)
    hyp = "".join(hyp_tokens)
    distance, substitutions, insertions, deletions = _levenshtein_breakdown(
        ref_tokens,
        hyp_tokens,
    )
    return TextComparison(
        ref,
        hyp,
        distance,
        substitutions,
        insertions,
        deletions,
        len(ref_tokens),
        len(hyp_tokens),
    )


def align_text(reference: str, hypothesis: str) -> list[AlignmentOp]:
    """Return a normalized character-level edit script."""
    return _alignment_ops(tokenize_copy_text(reference), tokenize_copy_text(hypothesis))


def alignment_summary(
    reference: str,
    hypothesis: str,
    *,
    sample_limit: int = 80,
) -> dict:
    """Summarize alignment errors for JSON reports."""
    ops = align_text(reference, hypothesis)
    substitution_counts: dict[str, int] = {}
    insertion_counts: dict[str, int] = {}
    deletion_counts: dict[str, int] = {}
    samples = []
    for op in ops:
        if op.op == "equal":
            continue
        if op.op == "substitute":
            key = f"{op.reference}->{op.hypothesis}"
            substitution_counts[key] = substitution_counts.get(key, 0) + 1
        elif op.op == "insert":
            insertion_counts[op.hypothesis] = insertion_counts.get(op.hypothesis, 0) + 1
        elif op.op == "delete":
            deletion_counts[op.reference] = deletion_counts.get(op.reference, 0) + 1
        if len(samples) < sample_limit:
            samples.append(
                {
                    "op": op.op,
                    "reference": op.reference,
                    "hypothesis": op.hypothesis,
                    "reference_index": op.reference_index,
                    "hypothesis_index": op.hypothesis_index,
                }
            )
    return {
        "top_substitutions": _sorted_counts(substitution_counts),
        "top_insertions": _sorted_counts(insertion_counts),
        "top_deletions": _sorted_counts(deletion_counts),
        "sample_errors": samples,
    }


def compare_best_substring(reference: str, hypothesis: str) -> TextComparison:
    """Compare hypothesis with the best similarly-sized substring of reference."""
    ref_tokens = tokenize_copy_text(reference)
    hyp_tokens = tokenize_copy_text(hypothesis)
    ref = "".join(ref_tokens)
    hyp = "".join(hyp_tokens)
    if not ref or not hyp:
        distance, substitutions, insertions, deletions = _levenshtein_breakdown(
            ref_tokens,
            hyp_tokens,
        )
        return TextComparison(
            ref,
            hyp,
            distance,
            substitutions,
            insertions,
            deletions,
            len(ref_tokens),
            len(hyp_tokens),
        )

    best_distance: int | None = None
    best_ref = ref
    best_breakdown = (0, 0, 0)
    best_units = len(ref_tokens)
    min_len = max(0, min(len(ref_tokens), len(hyp_tokens) - max(3, len(hyp_tokens) // 4)))
    max_len = min(len(ref_tokens), len(hyp_tokens) + max(3, len(hyp_tokens) // 4))
    step = max(1, len(hyp_tokens) // 10)
    for length in range(min_len, max_len + 1):
        if length == 0:
            continue
        starts = set(range(0, len(ref_tokens) - length + 1, step))
        if len(ref_tokens) >= length:
            starts.add(len(ref_tokens) - length)
        for start in starts:
            candidate_tokens = ref_tokens[start : start + length]
            distance, substitutions, insertions, deletions = _levenshtein_breakdown(
                candidate_tokens,
                hyp_tokens,
            )
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_ref = "".join(candidate_tokens)
                best_breakdown = (substitutions, insertions, deletions)
                best_units = len(candidate_tokens)
    return TextComparison(
        best_ref,
        hyp,
        best_distance or 0,
        *best_breakdown,
        best_units,
        len(hyp_tokens),
    )


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    return _levenshtein_breakdown(a, b)[0]


def _levenshtein_breakdown(a: Sequence[str], b: Sequence[str]) -> tuple[int, int, int, int]:
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


def _alignment_ops(a: Sequence[str], b: Sequence[str]) -> list[AlignmentOp]:
    rows = len(a) + 1
    cols = len(b) + 1
    distances = np.zeros((rows, cols), dtype=np.int32)
    for i in range(1, rows):
        distances[i, 0] = i
    for j in range(1, cols):
        distances[0, j] = j
    for i, ca in enumerate(a, start=1):
        for j, cb in enumerate(b, start=1):
            substitution_cost = 0 if ca == cb else 1
            distances[i, j] = min(
                distances[i - 1, j] + 1,
                distances[i, j - 1] + 1,
                distances[i - 1, j - 1] + substitution_cost,
            )

    ops: list[AlignmentOp] = []
    i = len(a)
    j = len(b)
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            substitution_cost = 0 if a[i - 1] == b[j - 1] else 1
            if distances[i, j] == distances[i - 1, j - 1] + substitution_cost:
                ops.append(
                    AlignmentOp(
                        "equal" if substitution_cost == 0 else "substitute",
                        a[i - 1],
                        b[j - 1],
                        i - 1,
                        j - 1,
                    )
                )
                i -= 1
                j -= 1
                continue
        if i > 0 and distances[i, j] == distances[i - 1, j] + 1:
            ops.append(AlignmentOp("delete", a[i - 1], "", i - 1, j))
            i -= 1
            continue
        if j > 0 and distances[i, j] == distances[i, j - 1] + 1:
            ops.append(AlignmentOp("insert", "", b[j - 1], i, j - 1))
            j -= 1
            continue
        raise RuntimeError("failed to backtrack alignment")
    ops.reverse()
    return ops


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


def _sorted_counts(counts: dict[str, int]) -> list[dict[str, int | str]]:
    return [
        {"item": item, "count": count}
        for item, count in sorted(counts.items(), key=lambda entry: (-entry[1], entry[0]))
    ]


def _median_smooth(values: np.ndarray, *, width: int) -> np.ndarray:
    if values.size == 0 or width <= 1:
        return values.astype(np.float32, copy=True)
    radius = width // 2
    out = np.empty_like(values, dtype=np.float32)
    for index in range(values.size):
        start = max(0, index - radius)
        end = min(values.size, index + radius + 1)
        out[index] = float(np.median(values[start:end]))
    return out


def _segments_from_groups(
    frames: Sequence[CarrierFrame],
    groups: Sequence[tuple[int, int]],
    centers: np.ndarray,
    audio_duration_s: float,
) -> list[CarrierSegment]:
    segments: list[CarrierSegment] = []
    for group_index, (start_index, end_index) in enumerate(groups):
        if start_index >= end_index:
            continue
        if group_index == 0:
            start_s = 0.0
        else:
            prev_mid = _frame_midpoint(frames[start_index - 1])
            this_mid = _frame_midpoint(frames[start_index])
            start_s = (prev_mid + this_mid) / 2.0
        if group_index == len(groups) - 1:
            end_s = audio_duration_s
        else:
            last_mid = _frame_midpoint(frames[end_index - 1])
            next_mid = _frame_midpoint(frames[end_index])
            end_s = (last_mid + next_mid) / 2.0
        center_hz = float(np.median(centers[start_index:end_index]))
        segments.append(
            CarrierSegment(
                max(0.0, start_s),
                min(audio_duration_s, end_s),
                center_hz,
                end_index - start_index,
            )
        )
    return segments


def _merge_short_carrier_segments(
    segments: list[CarrierSegment],
    *,
    min_segment_s: float,
    jump_hz: float,
) -> list[CarrierSegment]:
    if len(segments) <= 1 or min_segment_s <= 0:
        return segments
    merged = list(segments)
    index = 0
    while index < len(merged):
        segment = merged[index]
        duration = segment.end_s - segment.start_s
        if duration >= min_segment_s:
            index += 1
            continue
        if len(merged) == 1:
            break
        if index == 0:
            neighbor = 1
        elif index == len(merged) - 1:
            neighbor = index - 1
        else:
            left_delta = abs(segment.center_hz - merged[index - 1].center_hz)
            right_delta = abs(segment.center_hz - merged[index + 1].center_hz)
            neighbor = index - 1 if left_delta <= right_delta else index + 1
        combined = _combine_carrier_segments(segment, merged[neighbor])
        replace_at = min(index, neighbor)
        drop_at = max(index, neighbor)
        merged[replace_at] = combined
        del merged[drop_at]
        index = max(0, replace_at - 1)

    coalesced: list[CarrierSegment] = []
    for segment in merged:
        if (
            coalesced
            and abs(segment.center_hz - coalesced[-1].center_hz) <= jump_hz
        ):
            coalesced[-1] = _combine_carrier_segments(coalesced[-1], segment)
        else:
            coalesced.append(segment)
    return coalesced


def _combine_carrier_segments(a: CarrierSegment, b: CarrierSegment) -> CarrierSegment:
    frames = a.frames + b.frames
    if frames <= 0:
        center_hz = (a.center_hz + b.center_hz) / 2.0
    else:
        center_hz = (a.center_hz * a.frames + b.center_hz * b.frames) / frames
    return CarrierSegment(
        min(a.start_s, b.start_s),
        max(a.end_s, b.end_s),
        center_hz,
        frames,
    )


def _frame_midpoint(frame: CarrierFrame) -> float:
    return (frame.start_s + frame.end_s) / 2.0


def _canonical_token(token: str) -> str:
    if token == "<BT>":
        return "="
    if token == "<AR>":
        return "+"
    return token
