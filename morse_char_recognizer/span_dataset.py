"""Span-level training data for a *segmentation-aware* Morse glyph CNN.

The isolated-glyph CNN (:mod:`morse_char_recognizer.dataset`) only ever sees
audio that is exactly one well-cut character, so at decode time it is confidently
wrong on the *merged* spans the beam search inevitably proposes: two adjacent
letters keyed with a character gap (``T`` + ``N``) look, once summed into a
single span, like a plausible ``G`` (``--.``). That over-confidence is what makes
the CNN hurt rather than help the beam (it introduces a systematic merge bias).

This module renders *continuous multi-character* audio and, using the exact
element→character alignment recovered from the synthesiser's own event list,
labels every candidate glyph span the beam would consider:

* a span whose elements are exactly one true character -> that character (positive)
* a span that crosses a character boundary (a merge of >= 2 characters) whose
  dit/dah pattern still forms a valid glyph -> a dedicated ``<JUNK>`` class

Sub-spans strictly *inside* a single character (e.g. the first two dits of an
``S``, acoustically identical to a real ``I``) are deliberately *not* emitted:
nothing in the span's audio can distinguish them from the real glyph — that is
the timing model's job, not the CNN's. Teaching the CNN to reject them would make
it reject genuine glyphs.

A CNN trained with the ``<JUNK>`` class puts merged spans' probability mass on
``<JUNK>``, so ``logP(real char)`` collapses for a bad merge and the beam stops
preferring it — with no change to the decoder itself.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from morse_char_recognizer.beam_decode import _lookup
from morse_char_recognizer.classes import REAL_CLASSES
from morse_char_recognizer.features import EnvelopeConfig, extract_unit_scaled_envelope
from morse_char_recognizer.morse_table import unit_seconds
from morse_synth.channel import ChannelConfig
from morse_synth.core import render
from morse_synth.keying import KeyingConfig
from morse_synth.operator import OperatorConfig, _token_code, _word_tokens, build_events

Range = tuple[float, float]

JUNK_TOKEN = "<JUNK>"


@dataclass(frozen=True)
class SpanDatasetConfig:
    """Parameter ranges for span-level (merge-aware) training data."""

    classes: tuple[str, ...] = REAL_CLASSES
    junk_token: str = JUNK_TOKEN
    sample_rate: int = 8000
    envelope: EnvelopeConfig = field(
        default_factory=lambda: EnvelopeConfig(length=128, scale_mode="unit", canonical_units=32.0)
    )
    num_messages: int = 2000
    tokens_per_message_range: tuple[int, int] = (2, 7)
    junk_per_positive: float = 1.5  # cap on junk spans emitted per positive span
    max_elements_per_char: int = 8
    dot_dash_split: float = 2.0
    pad_ms: float = 20.0
    seed: int | None = 0
    # operator / keying / channel augmentation (mirrors the glyph dataset)
    wpm_range: Range = (12.0, 30.0)
    freq_range: Range = (500.0, 750.0)
    amplitude_range: Range = (0.35, 0.9)
    rise_ms_range: Range = (2.0, 8.0)
    element_jitter_range: Range = (0.0, 0.12)
    gap_jitter_range: Range = (0.0, 0.12)
    dash_dot_ratio_range: Range = (2.7, 3.4)
    gap_inflation_range: Range = (0.9, 1.25)
    snr_db_range: Range | None = None
    qrn_rate_per_sec_range: Range = (0.0, 0.0)
    qrn_amplitude_db_range: Range = (-12.0, -6.0)


@dataclass(frozen=True)
class LabeledSpan:
    """One extracted span envelope and the class token it should be trained to."""

    token: str  # a real class token, or the junk token
    envelope: np.ndarray


def _uniform(rng: np.random.Generator, value_range: Range) -> float:
    low, high = value_range
    return float(low) if low == high else float(rng.uniform(low, high))


def _token_element_counts(text: str) -> list[tuple[str, int]]:
    """Recover, in order, each rendered token and how many elements it keys.

    Mirrors :func:`morse_synth.operator.build_events` tokenisation (dropping
    unknown tokens), so the returned counts line up one-for-one with the ON
    events of the same text.
    """
    out: list[tuple[str, int]] = []
    for word in text.upper().split():
        for token in _word_tokens(word):
            code = _token_code(token)
            if code:
                out.append((token, len(code)))
    return out


def _element_bounds(events, sample_rate: int) -> list[tuple[int, int]]:
    """Sample [start, end) of every keydown, matching ``render_events`` rounding."""
    cursor = 0
    bounds: list[tuple[int, int]] = []
    for is_on, dur in events:
        span = int(round(dur * sample_rate))
        if is_on:
            bounds.append((cursor, cursor + span))
        cursor += span
    return bounds


def _random_message(rng: np.random.Generator, config: SpanDatasetConfig) -> str:
    """Build a space-separated message by sampling class tokens roughly uniformly."""
    lo, hi = config.tokens_per_message_range
    n = int(rng.integers(lo, hi + 1))
    tokens = [config.classes[int(rng.integers(0, len(config.classes)))] for _ in range(n)]
    # group into words of 1-4 tokens so word gaps appear too
    words: list[str] = []
    i = 0
    while i < n:
        take = int(rng.integers(1, 5))
        words.append("".join(tokens[i : i + take]))
        i += take
    return " ".join(words)


def spans_from_message(
    text: str,
    rng: np.random.Generator,
    config: SpanDatasetConfig | None = None,
) -> list[LabeledSpan]:
    """Render one message and emit its positive and junk (merge) spans."""
    config = config or SpanDatasetConfig()
    allowed = set(config.classes)

    wpm = _uniform(rng, config.wpm_range)
    freq = _uniform(rng, config.freq_range)
    amplitude = _uniform(rng, config.amplitude_range)
    rise_ms = _uniform(rng, config.rise_ms_range)
    operator = OperatorConfig(
        wpm=wpm,
        element_jitter=_uniform(rng, config.element_jitter_range),
        gap_jitter=_uniform(rng, config.gap_jitter_range),
        dash_dot_ratio=_uniform(rng, config.dash_dot_ratio_range),
        gap_inflation=_uniform(rng, config.gap_inflation_range),
        seed=int(rng.integers(0, np.iinfo(np.int32).max)),
    )
    events = build_events(text, operator)
    bounds = _element_bounds(events, config.sample_rate)
    counts = _token_element_counts(text)
    if not bounds or sum(n for _t, n in counts) != len(bounds):
        return []

    channel = ChannelConfig(
        qrn_rate_per_sec=_uniform(rng, config.qrn_rate_per_sec_range),
        qrn_amplitude_db=_uniform(rng, config.qrn_amplitude_db_range),
        rx_filter_centre=freq,
        seed=int(rng.integers(0, np.iinfo(np.int32).max)),
    )
    if config.snr_db_range is not None:
        channel.snr_db = _uniform(rng, config.snr_db_range)

    audio = render(
        text,
        operator=operator,
        keying=KeyingConfig(rise_ms=rise_ms, seed=int(rng.integers(0, np.iinfo(np.int32).max))),
        channel=channel,
        freq=freq,
        sample_rate=config.sample_rate,
        amplitude=amplitude,
    )

    unit_samples = max(1, int(round(unit_seconds(wpm) * config.sample_rate)))
    threshold = config.dot_dash_split * unit_samples
    symbols = ["-" if (end - start) >= threshold else "." for start, end in bounds]

    # token element ranges: token k occupies elements [starts[k], starts[k]+len)
    token_start: list[int] = []
    acc = 0
    for _tok, n in counts:
        token_start.append(acc)
        acc += n
    # map each element index -> owning token index
    owner = [0] * len(bounds)
    for k, (_tok, n) in enumerate(counts):
        for e in range(token_start[k], token_start[k] + n):
            owner[e] = k
    exact_range = {(token_start[k], token_start[k] + n): k for k, (_tok, n) in enumerate(counts)}

    positives: list[tuple[int, int, str]] = []
    junks: list[tuple[int, int]] = []
    n_elems = len(bounds)
    for i in range(1, n_elems + 1):
        for length in range(1, min(config.max_elements_per_char, i) + 1):
            j = i - length
            char = _lookup("".join(symbols[j:i]), prefer_letters=True)
            if char is None:
                continue
            token_k = exact_range.get((j, i))
            if token_k is not None:
                tok = counts[token_k][0]
                if tok in allowed:
                    positives.append((j, i, tok))
                continue
            # not an exact single-token span: junk only if it crosses a boundary
            if owner[j] != owner[i - 1]:
                junks.append((j, i))

    # balance: cap junk relative to positives in this message
    max_junk = int(round(config.junk_per_positive * max(1, len(positives))))
    if len(junks) > max_junk:
        pick = rng.choice(len(junks), size=max_junk, replace=False)
        junks = [junks[p] for p in sorted(pick)]

    pad = int(round(config.pad_ms / 1000.0 * config.sample_rate))
    out: list[LabeledSpan] = []
    for j, i, tok in positives:
        out.append(LabeledSpan(tok, _span_envelope(audio, bounds, j, i, pad, unit_samples, config)))
    for j, i in junks:
        out.append(
            LabeledSpan(config.junk_token, _span_envelope(audio, bounds, j, i, pad, unit_samples, config))
        )
    return out


def _span_envelope(audio, bounds, j, i, pad, unit_samples, config: SpanDatasetConfig) -> np.ndarray:
    start = max(0, bounds[j][0] - pad)
    end = min(len(audio), bounds[i - 1][1] + pad)
    return extract_unit_scaled_envelope(audio[start:end], config.sample_rate, unit_samples, config.envelope)


def iter_labeled_spans(config: SpanDatasetConfig | None = None) -> Iterator[LabeledSpan]:
    """Yield labeled spans from ``num_messages`` randomly generated messages."""
    config = config or SpanDatasetConfig()
    rng = np.random.default_rng(config.seed)
    for _ in range(config.num_messages):
        text = _random_message(rng, config)
        yield from spans_from_message(text, rng, config)


def build_span_dataset(
    config: SpanDatasetConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Materialise ``(X, y, classes)`` where ``classes`` ends with the junk token."""
    config = config or SpanDatasetConfig()
    classes = (*config.classes, config.junk_token)
    class_index = {c: k for k, c in enumerate(classes)}
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for span in iter_labeled_spans(config):
        xs.append(span.envelope)
        ys.append(class_index[span.token])
    if not xs:
        return (
            np.zeros((0, config.envelope.length), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            classes,
        )
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), classes
