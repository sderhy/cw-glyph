"""Tests for envelope-based Morse segmentation."""

from __future__ import annotations

import numpy as np

from morse_char_recognizer.segment import SegmentConfig, detect_active_regions, segment_characters
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


def test_adaptive_threshold_recovers_faded_keydowns() -> None:
    sample_rate = 8000
    audio = _tone_pulse_train(sample_rate, amplitudes=[1.0, 0.08, 1.0])

    fixed = detect_active_regions(audio, sample_rate, SegmentConfig(threshold=0.18))
    adaptive = detect_active_regions(
        audio,
        sample_rate,
        SegmentConfig(
            threshold_mode="adaptive",
            threshold=0.18,
            adaptive_window_ms=220.0,
            adaptive_min_threshold=0.01,
        ),
    )

    assert len(fixed) == 2
    assert len(adaptive) == 3


def _tone_pulse_train(sample_rate: int, amplitudes: list[float]) -> np.ndarray:
    chunks = []
    pulse_samples = int(round(0.08 * sample_rate))
    gap_samples = int(round(0.24 * sample_rate))
    t = np.arange(pulse_samples, dtype=np.float32) / sample_rate
    tone = np.sin(2 * np.pi * 600.0 * t).astype(np.float32)
    for index, amplitude in enumerate(amplitudes):
        if index > 0:
            chunks.append(np.zeros(gap_samples, dtype=np.float32))
        chunks.append((amplitude * tone).astype(np.float32))
    return np.concatenate(chunks)
