"""Tests for span-level (merge-aware) training data generation."""

import numpy as np

from morse_char_recognizer.span_dataset import (
    JUNK_TOKEN,
    SpanDatasetConfig,
    _element_bounds,
    _token_element_counts,
    build_span_dataset,
    spans_from_message,
)
from morse_synth.operator import OperatorConfig, build_events

CLEAN = SpanDatasetConfig(
    wpm_range=(20.0, 20.0),
    freq_range=(600.0, 600.0),
    amplitude_range=(0.6, 0.6),
    rise_ms_range=(4.0, 4.0),
    element_jitter_range=(0.0, 0.0),
    gap_jitter_range=(0.0, 0.0),
    dash_dot_ratio_range=(3.0, 3.0),
    gap_inflation_range=(1.0, 1.0),
    qrn_rate_per_sec_range=(0.0, 0.0),
)


def test_token_counts_align_with_events():
    text = "TN X"
    events = build_events(text, OperatorConfig(wpm=20.0))
    bounds = _element_bounds(events, 8000)
    counts = _token_element_counts(text)
    assert counts == [("T", 1), ("N", 2), ("X", 4)]
    assert sum(n for _t, n in counts) == len(bounds)


def test_merge_span_is_labeled_junk_not_a_glyph():
    # "T" + "N" keyed with a character gap must not be trainable as "G" (--.).
    spans = spans_from_message("TN", np.random.default_rng(0), CLEAN)
    positives = sorted(s.token for s in spans if s.token != JUNK_TOKEN)
    junks = [s for s in spans if s.token == JUNK_TOKEN]
    assert positives == ["N", "T"]  # both true characters recovered
    assert junks  # the T+N (and T+N-dah = M) merges are present as junk


def test_sub_spans_inside_one_character_are_not_emitted():
    # "S" is three dits; its 2-dit and 1-dit sub-spans are acoustically real
    # I / E glyphs, so they must be neither positive nor junk (skipped).
    spans = spans_from_message("S", np.random.default_rng(1), CLEAN)
    tokens = [s.token for s in spans]
    assert tokens == ["S"]  # only the whole character, nothing else


def test_build_span_dataset_appends_junk_class():
    config = SpanDatasetConfig(
        num_messages=40,
        tokens_per_message_range=(2, 5),
        wpm_range=(18.0, 24.0),
        seed=3,
    )
    x, y, classes = build_span_dataset(config)
    assert classes[-1] == JUNK_TOKEN
    assert x.shape[0] == y.shape[0] > 0
    assert x.shape[1] == config.envelope.length
    junk_index = len(classes) - 1
    # both real and junk labels must be present for the model to learn rejection
    assert (y == junk_index).any()
    assert (y != junk_index).any()
