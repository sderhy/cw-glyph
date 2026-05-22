"""Tests for envelope-based Morse segmentation."""

from __future__ import annotations

from morse_char_recognizer.segment import detect_active_regions, segment_characters
from morse_synth.core import synthesize


def test_detect_active_regions_finds_morse_elements() -> None:
    audio = synthesize("S", wpm=20, sample_rate=8000, rise_ms=1.0)

    regions = detect_active_regions(audio, 8000)

    assert len(regions) == 3
    assert all(region.duration > 0 for region in regions)


def test_segment_characters_splits_clean_word() -> None:
    audio = synthesize("SOS", wpm=20, sample_rate=8000, rise_ms=1.0)

    segments = segment_characters(audio, 8000)

    assert len(segments) == 3
    assert segments[0].start < segments[0].end <= segments[1].start
    assert segments[1].start < segments[1].end <= segments[2].start
