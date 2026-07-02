"""Element-level Morse decoder with global beam/DP segmentation.

Instead of cutting characters with a hard threshold on inter-element gaps
(which is fragile when an operator's spacing is irregular), this decoder:

1. detects key-down elements and classifies each as dit/dah by duration;
2. searches *globally* for the way to group consecutive elements into
   characters that are all valid Morse glyphs, using the gaps only as a soft
   prior (large gaps prefer a character boundary, but never force one).

Because a Morse character is at most a handful of elements, the optimal
segmentation is found exactly by dynamic programming (an infinite-width beam).
A single dit (`.` -> E) is a valid character, so short glyphs are never dropped
the way a `min-segment-units` noise filter drops them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .morse_table import INVERSE_PROSIGNS, INVERSE_TABLE
from .segment import (
    Region,
    SegmentConfig,
    detect_active_regions,
    estimate_unit_samples,
)


@dataclass
class BeamConfig:
    """Scoring weights for the segmentation search (all gaps are in dit units).

    Gaps are modelled with two soft modes: an intra-element gap (~1 unit) kept
    inside a character, and a character gap (~3 units) sitting on a boundary.
    The boundary threshold (~2 units) is the soft midpoint between them. A gap
    kept inside a character is penalised when it grows past the threshold; a gap
    used as a boundary is penalised when it falls below it. Neither is a hard
    cut: the Morse-validity constraint can still force a split on a small gap (so
    `...--.` must become S+G) or a merge across a wide one.
    """

    min_element_units: float = 0.45  # drop elements shorter than this * unit (noise)
    dot_dash_split: float = 1.8  # element is a dah when duration >= this * unit
    max_elements_per_char: int = 8  # longest valid glyph (<HH> = 8 dits)
    char_cost: float = 0.0  # optional MDL prior: penalty per emitted character
    boundary_threshold: float = 2.0  # gap units separating intra-element from char gap
    w_internal: float = 1.0  # penalty per unit a kept-internal gap exceeds threshold
    w_boundary: float = 1.0  # penalty per unit a boundary gap is below threshold
    word_gap_units: float = 5.0  # boundary gap >= this emits a word space
    prefer_letters: bool = True  # resolve dit/dah aliases to letters, not prosigns
    allowed_tokens: tuple[str, ...] | None = None  # optional decode vocabulary


def _lookup(
    pattern: str,
    *,
    prefer_letters: bool,
    allowed_tokens: tuple[str, ...] | None = None,
) -> str | None:
    """Map a dit/dah pattern to a character token, or None if it is not valid."""
    candidates = _lookup_candidates(pattern, prefer_letters=prefer_letters)
    if allowed_tokens is None:
        return candidates[0] if candidates else None
    allowed = set(allowed_tokens)
    for candidate in candidates:
        if candidate in allowed:
            return candidate
    return None


def _lookup_candidates(pattern: str, *, prefer_letters: bool) -> list[str]:
    candidates: list[str | None]
    if prefer_letters:
        candidates = [INVERSE_TABLE.get(pattern), INVERSE_PROSIGNS.get(pattern)]
    else:
        candidates = [INVERSE_PROSIGNS.get(pattern), INVERSE_TABLE.get(pattern)]
    out: list[str] = []
    for candidate in candidates:
        if candidate is not None and candidate not in out:
            out.append(candidate)
    return out


def classify_elements(
    regions: list[Region], unit_samples: int, *, dot_dash_split: float = 2.0
) -> list[str]:
    """Classify each active region as a dit (`.`) or dah (`-`)."""
    threshold = dot_dash_split * unit_samples
    return ["-" if r.duration >= threshold else "." for r in regions]


def filter_regions(
    regions: list[Region], unit_samples: int, min_element_units: float
) -> list[Region]:
    """Drop sub-unit specks that would otherwise become spurious dits."""
    if unit_samples > 0 and min_element_units > 0:
        floor = min_element_units * unit_samples
        return [r for r in regions if r.duration >= floor]
    return list(regions)


def enumerate_valid_spans(
    symbols: list[str],
    *,
    max_elements_per_char: int,
    prefer_letters: bool = True,
    allowed_tokens: tuple[str, ...] | None = None,
) -> list[tuple[int, int, str]]:
    """List every ``(j, i, char)`` where ``symbols[j:i]`` is a valid Morse glyph.

    These are exactly the character candidates the DP in
    :func:`beam_decode_regions` considers, so an external scorer (e.g. a CNN)
    can be evaluated on the same spans and indexed by ``(j, i)``.
    """
    n = len(symbols)
    spans: list[tuple[int, int, str]] = []
    for i in range(1, n + 1):
        for length in range(1, min(max_elements_per_char, i) + 1):
            j = i - length
            char = _lookup(
                "".join(symbols[j:i]),
                prefer_letters=prefer_letters,
                allowed_tokens=allowed_tokens,
            )
            if char is not None:
                spans.append((j, i, char))
    return spans


def beam_decode_regions(
    regions: list[Region],
    unit_samples: int,
    config: BeamConfig | None = None,
    *,
    span_score: Callable[[int, int, str], float] | None = None,
) -> str:
    """Decode a character string from detected elements by exact DP segmentation.

    ``span_score(j, i, char)`` is an optional extra term (already weighted) added
    to the score of grouping the *filtered* elements ``[j:i)`` into ``char`` — the
    hook used to fold in CNN glyph log-probabilities. Its ``(j, i)`` indices are
    into the region list *after* the ``min_element_units`` noise filter, matching
    :func:`enumerate_valid_spans` on the same filtered regions.
    """
    config = config or BeamConfig()
    regions = filter_regions(regions, unit_samples, config.min_element_units)
    n = len(regions)
    if n == 0 or unit_samples <= 0:
        return ""

    symbols = classify_elements(regions, unit_samples, dot_dash_split=config.dot_dash_split)
    # gaps[k] = gap (in dit units) between element k and element k+1
    gaps = [
        (regions[k + 1].start - regions[k].end) / unit_samples for k in range(n - 1)
    ]

    neg = float("-inf")
    score = [neg] * (n + 1)
    score[0] = 0.0
    back: list[tuple[int, str, bool] | None] = [None] * (n + 1)

    for i in range(1, n + 1):
        max_len = min(config.max_elements_per_char, i)
        for length in range(1, max_len + 1):
            j = i - length
            if score[j] == neg:
                continue
            char = _lookup(
                "".join(symbols[j:i]),
                prefer_letters=config.prefer_letters,
                allowed_tokens=config.allowed_tokens,
            )
            if char is None:
                continue
            s = score[j] - config.char_cost
            if span_score is not None:
                s += span_score(j, i, char)
            # penalise over-long gaps kept *inside* this character
            for k in range(j, i - 1):
                s -= config.w_internal * max(0.0, gaps[k] - config.boundary_threshold)
            # penalise the boundary before this character if its gap is too small
            space_before = False
            if j > 0:
                boundary_gap = gaps[j - 1]
                s -= config.w_boundary * max(0.0, config.boundary_threshold - boundary_gap)
                space_before = boundary_gap >= config.word_gap_units
            if s > score[i]:
                score[i] = s
                back[i] = (j, char, space_before)

    if score[n] == neg:
        return ""

    segments: list[tuple[str, bool]] = []
    i = n
    while i > 0:
        entry = back[i]
        assert entry is not None
        j, char, space_before = entry
        segments.append((char, space_before))
        i = j
    segments.reverse()

    parts: list[str] = []
    for char, space_before in segments:
        if space_before:
            parts.append(" ")
        parts.append(char)
    return "".join(parts).strip()


def beam_decode_audio(
    audio,
    sample_rate: int,
    *,
    segment_config: SegmentConfig | None = None,
    beam_config: BeamConfig | None = None,
    unit_samples: int | None = None,
) -> tuple[str, list[Region], int]:
    """Detect elements from audio and decode them. Returns (text, regions, unit)."""
    segment_config = segment_config or SegmentConfig()
    regions = detect_active_regions(audio, sample_rate, segment_config)
    if unit_samples is None:
        min_unit_ms = segment_config.min_unit_ms or 35.0
        unit_samples = estimate_unit_samples(
            regions,
            sample_rate,
            min_unit_ms=min_unit_ms,
            max_unit_ms=segment_config.max_unit_ms,
            histogram_bin_ms=segment_config.unit_histogram_bin_ms,
        )
    text = beam_decode_regions(regions, unit_samples, beam_config)
    return text, regions, unit_samples
