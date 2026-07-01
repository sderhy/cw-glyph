"""Tests for envelope-based Morse segmentation."""

from __future__ import annotations

import numpy as np

from morse_char_recognizer.segment import (
    Region,
    SegmentConfig,
    _hysteresis_mask,
    detect_active_regions,
    estimate_unit_samples,
    segment_characters,
)
from morse_synth.core import synthesize


def test_hysteresis_keeps_weak_body_and_drops_peakless_noise() -> None:
    # run A peaks above `high` with a weak tail above `low` -> kept whole;
    # run B never reaches `high` -> dropped entirely.
    env = np.array([0.0, 0.15, 0.30, 0.15, 0.0, 0.15, 0.15, 0.0, 0.30, 0.0], dtype=np.float32)
    mask = _hysteresis_mask(env, high=0.22, low=0.12)
    kept = set(np.flatnonzero(mask).tolist())
    assert kept == {1, 2, 3, 8}


def test_hysteresis_falls_back_to_fixed_when_low_not_below_high() -> None:
    env = np.array([0.0, 0.25, 0.10, 0.30], dtype=np.float32)
    mask = _hysteresis_mask(env, high=0.22, low=0.22)
    assert mask.tolist() == (env >= 0.22).tolist()


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


def test_detect_active_regions_uses_configured_bandpass_center() -> None:
    audio = synthesize("E", wpm=20, freq=900.0, sample_rate=8000, rise_ms=1.0)

    regions = detect_active_regions(
        audio,
        8000,
        SegmentConfig(bandpass_center_hz=900.0, bandpass_width_hz=200.0),
    )

    assert len(regions) == 1


def test_estimate_unit_samples_ignores_implausibly_short_fragments() -> None:
    sample_rate = 8000
    durations_ms = [11, 14, 19, 22, 56, 58, 61, 64, 171, 183]
    cursor = 0
    regions = []
    for duration_ms in durations_ms:
        duration = int(round(duration_ms / 1000.0 * sample_rate))
        regions.append(Region(cursor, cursor + duration))
        cursor += duration + int(round(0.05 * sample_rate))

    unit_ms = estimate_unit_samples(regions, sample_rate) / sample_rate * 1000.0

    assert 55.0 <= unit_ms <= 65.0


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


def test_segment_characters_can_split_overlong_candidate() -> None:
    audio = synthesize("OL", wpm=20, sample_rate=8000, rise_ms=1.0)

    merged = segment_characters(audio, 8000, SegmentConfig(char_gap_units=4.0))
    split = segment_characters(
        audio,
        8000,
        SegmentConfig(char_gap_units=4.0, max_character_units=20.0),
    )

    assert len(merged) == 1
    assert len(split) == 2


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
