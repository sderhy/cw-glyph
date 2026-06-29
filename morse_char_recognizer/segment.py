"""Envelope-based segmentation of continuous Morse audio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import percentile_filter

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
    bandpass_center_hz: float | None = 600.0
    bandpass_width_hz: float = 500.0
    noise_floor_percentile: float = 20.0
    threshold_mode: Literal["fixed", "adaptive"] = "fixed"
    threshold: float = 0.18
    adaptive_window_ms: float = 650.0
    adaptive_floor_percentile: float = 20.0
    adaptive_peak_percentile: float = 90.0
    adaptive_min_threshold: float = 0.03
    min_keydown_ms: float = 10.0
    merge_gap_ms: float = 8.0
    min_unit_ms: float = 0.0
    max_unit_ms: float = 140.0
    unit_histogram_bin_ms: float = 10.0
    char_gap_units: float = 2.0
    fixed_char_gap_ms: float | None = None
    max_character_units: float | None = None
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
        bandpass_center_hz=config.bandpass_center_hz,
        bandpass_width_hz=config.bandpass_width_hz,
        noise_floor_percentile=config.noise_floor_percentile,
    )
    regions = _runs(_active_mask(envelope, sample_rate, config))
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

    unit = estimate_unit_samples(
        active,
        sample_rate,
        min_unit_ms=config.min_unit_ms,
        max_unit_ms=config.max_unit_ms,
        histogram_bin_ms=config.unit_histogram_bin_ms,
    )
    segments: list[Region] = []
    group: list[Region] = [active[0]]
    prev = active[0]
    for region in active[1:]:
        gap = region.start - prev.end
        if gap >= gap_threshold:
            segments.extend(_character_segments(group, unit, pad, len(audio), config))
            group = [region]
        else:
            group.append(region)
        prev = region
    segments.extend(_character_segments(group, unit, pad, len(audio), config))
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


def estimate_unit_samples(
    active_regions: list[Region],
    sample_rate: int | None = None,
    *,
    min_unit_ms: float = 35.0,
    max_unit_ms: float = 140.0,
    histogram_bin_ms: float = 10.0,
) -> int:
    """Estimate one Morse dot unit from detected keydown durations."""
    if not active_regions:
        return 0
    durations = np.array([region.duration for region in active_regions], dtype=np.float32)

    if sample_rate is not None and min_unit_ms > 0 and max_unit_ms > min_unit_ms:
        min_unit = _ms_to_samples(min_unit_ms, sample_rate)
        max_unit = _ms_to_samples(max_unit_ms, sample_rate)
        plausible = durations[(durations >= min_unit) & (durations <= max_unit)]
        if plausible.size >= 3:
            bin_size = max(1, _ms_to_samples(histogram_bin_ms, sample_rate))
            bins = np.arange(min_unit, max_unit + bin_size, bin_size, dtype=np.float32)
            if bins.size >= 2:
                counts, edges = np.histogram(plausible, bins=bins)
                if counts.size and int(counts.max()) > 0:
                    peak = int(np.argmax(counts))
                    in_peak = plausible[
                        (plausible >= edges[peak]) & (plausible < edges[peak + 1])
                    ]
                    if in_peak.size:
                        return max(1, int(round(float(np.median(in_peak)))))

    short = durations[durations <= np.percentile(durations, 45.0)]
    if short.size >= 3:
        estimate = float(np.median(short))
    else:
        estimate = float(np.percentile(durations, 20.0))
    return max(1, int(round(estimate)))


def _active_mask(
    envelope: np.ndarray,
    sample_rate: int,
    config: SegmentConfig,
) -> np.ndarray:
    if config.threshold_mode == "fixed":
        return envelope >= config.threshold
    if config.threshold_mode != "adaptive":
        raise ValueError(f"unknown threshold mode: {config.threshold_mode!r}")
    _validate_adaptive_config(config)

    window = max(3, _ms_to_samples(config.adaptive_window_ms, sample_rate))
    if window % 2 == 0:
        window += 1
    floor = percentile_filter(
        envelope,
        percentile=config.adaptive_floor_percentile,
        size=window,
        mode="nearest",
    )
    peak = percentile_filter(
        envelope,
        percentile=config.adaptive_peak_percentile,
        size=window,
        mode="nearest",
    )
    threshold = floor + config.threshold * np.maximum(peak - floor, 0.0)
    threshold = np.maximum(threshold, config.adaptive_min_threshold)
    return envelope >= threshold


def _validate_adaptive_config(config: SegmentConfig) -> None:
    if config.adaptive_window_ms <= 0:
        raise ValueError(
            f"adaptive_window_ms must be positive, got {config.adaptive_window_ms}"
        )
    if not 0 <= config.adaptive_floor_percentile <= 100:
        raise ValueError(
            "adaptive_floor_percentile must be in [0, 100], "
            f"got {config.adaptive_floor_percentile}"
        )
    if not 0 <= config.adaptive_peak_percentile <= 100:
        raise ValueError(
            "adaptive_peak_percentile must be in [0, 100], "
            f"got {config.adaptive_peak_percentile}"
        )
    if config.adaptive_floor_percentile >= config.adaptive_peak_percentile:
        raise ValueError("adaptive_floor_percentile must be below adaptive_peak_percentile")
    if config.adaptive_min_threshold < 0:
        raise ValueError(
            f"adaptive_min_threshold must be non-negative, got {config.adaptive_min_threshold}"
        )


def _char_gap_threshold(
    active_regions: list[Region],
    sample_rate: int,
    config: SegmentConfig,
) -> int:
    if config.fixed_char_gap_ms is not None:
        return _ms_to_samples(config.fixed_char_gap_ms, sample_rate)
    unit = estimate_unit_samples(
        active_regions,
        sample_rate,
        min_unit_ms=config.min_unit_ms,
        max_unit_ms=config.max_unit_ms,
        histogram_bin_ms=config.unit_histogram_bin_ms,
    )
    return max(1, int(round(unit * config.char_gap_units)))


def _character_segments(
    active_group: list[Region],
    unit_samples: int,
    pad: int,
    audio_len: int,
    config: SegmentConfig,
) -> list[Region]:
    if not active_group:
        return []
    if (
        config.max_character_units is None
        or config.max_character_units <= 0
        or unit_samples <= 0
        or len(active_group) <= 1
    ):
        return [_padded_region(active_group[0].start, active_group[-1].end, pad, audio_len)]

    active_span = active_group[-1].end - active_group[0].start
    if active_span <= unit_samples * config.max_character_units:
        return [_padded_region(active_group[0].start, active_group[-1].end, pad, audio_len)]

    gaps = [
        (index, active_group[index].start - active_group[index - 1].end)
        for index in range(1, len(active_group))
    ]
    split_index, _gap = max(gaps, key=lambda item: item[1])
    left = active_group[:split_index]
    right = active_group[split_index:]
    return [
        *_character_segments(left, unit_samples, pad, audio_len, config),
        *_character_segments(right, unit_samples, pad, audio_len, config),
    ]


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
