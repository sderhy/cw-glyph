"""Envelope-based segmentation of continuous Morse audio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from morse_char_recognizer.features import (
    EnvelopeConfig,
    EnvelopeMethod,
    amplitude_envelope,
    extract_envelope,
    extract_unit_scaled_envelope,
)


@dataclass(frozen=True)
class Region:
    """Sample interval using half-open coordinates: [start, end)."""

    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class SegmentConfig:
    """Configuration for envelope-threshold Morse segmentation."""

    method: EnvelopeMethod = "hilbert"
    smoothing_ms: float = 4.0
    threshold: float = 0.18
    min_keydown_ms: float = 10.0
    merge_gap_ms: float = 8.0
    char_gap_units: float = 2.0
    fixed_char_gap_ms: float | None = None
    pad_ms: float = 20.0


def detect_active_regions(
    audio: np.ndarray,
    sample_rate: int,
    config: SegmentConfig | None = None,
) -> list[Region]:
    """Detect keydown regions in continuous Morse audio."""
    config = config or SegmentConfig()
    envelope = amplitude_envelope(
        audio,
        sample_rate,
        method=config.method,
        smoothing_ms=config.smoothing_ms,
    )
    regions = _runs(envelope >= config.threshold)
    regions = _merge_close_regions(regions, _ms_to_samples(config.merge_gap_ms, sample_rate))
    min_keydown = _ms_to_samples(config.min_keydown_ms, sample_rate)
    return [region for region in regions if region.duration >= min_keydown]


def segment_characters(
    audio: np.ndarray,
    sample_rate: int,
    config: SegmentConfig | None = None,
) -> list[Region]:
    """Split continuous Morse audio into character candidate regions."""
    config = config or SegmentConfig()
    active = detect_active_regions(audio, sample_rate, config)
    if not active:
        return []

    gap_threshold = _char_gap_threshold(active, sample_rate, config)
    pad = _ms_to_samples(config.pad_ms, sample_rate)

    segments: list[Region] = []
    start = active[0].start
    prev = active[0]
    for region in active[1:]:
        gap = region.start - prev.end
        if gap >= gap_threshold:
            segments.append(_padded_region(start, prev.end, pad, len(audio)))
            start = region.start
        prev = region
    segments.append(_padded_region(start, prev.end, pad, len(audio)))
    return segments


def extract_segment_envelopes(
    audio: np.ndarray,
    sample_rate: int,
    segments: list[Region],
    envelope_config: EnvelopeConfig | None = None,
    unit_samples: int | None = None,
) -> list[np.ndarray]:
    """Extract fixed-length envelopes for character candidate regions."""
    envelope_config = envelope_config or EnvelopeConfig()
    if envelope_config.scale_mode == "unit" and unit_samples is not None:
        return [
            extract_unit_scaled_envelope(
                audio[segment.start : segment.end],
                sample_rate,
                unit_samples,
                envelope_config,
            )
            for segment in segments
        ]
    return [
        extract_envelope(audio[segment.start : segment.end], sample_rate, envelope_config)
        for segment in segments
    ]


def estimate_unit_samples(active_regions: list[Region]) -> int:
    """Estimate one Morse dot unit from detected keydown durations."""
    if not active_regions:
        return 0
    durations = np.array([region.duration for region in active_regions], dtype=np.float32)
    return max(1, int(round(float(np.percentile(durations, 20.0)))))


def _char_gap_threshold(
    active_regions: list[Region],
    sample_rate: int,
    config: SegmentConfig,
) -> int:
    if config.fixed_char_gap_ms is not None:
        return _ms_to_samples(config.fixed_char_gap_ms, sample_rate)
    unit = estimate_unit_samples(active_regions)
    return max(1, int(round(unit * config.char_gap_units)))


def _runs(mask: np.ndarray) -> list[Region]:
    if mask.size == 0 or not mask.any():
        return []
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [Region(int(start), int(end)) for start, end in edges.reshape(-1, 2)]


def _merge_close_regions(regions: list[Region], max_gap: int) -> list[Region]:
    if not regions:
        return []
    merged = [regions[0]]
    for region in regions[1:]:
        prev = merged[-1]
        if region.start - prev.end <= max_gap:
            merged[-1] = Region(prev.start, region.end)
        else:
            merged.append(region)
    return merged


def _padded_region(start: int, end: int, pad: int, audio_len: int) -> Region:
    return Region(max(0, start - pad), min(audio_len, end + pad))


def _ms_to_samples(ms: float, sample_rate: int) -> int:
    if ms < 0:
        raise ValueError(f"duration must be non-negative, got {ms}")
    return int(round(ms / 1000.0 * sample_rate))
