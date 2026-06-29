"""Feature extraction for isolated Morse-character audio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import butter, hilbert, resample, sosfiltfilt

EnvelopeMethod = Literal["abs", "hilbert"]
NormalizeMode = Literal["peak", "rms", "none"]


@dataclass(frozen=True)
class EnvelopeConfig:
    """Configuration for fixed-length 1D envelope extraction."""

    length: int = 256
    method: EnvelopeMethod = "hilbert"
    smoothing_ms: float = 8.0
    trim_threshold: float = 0.03
    trim_pad_ms: float = 12.0
    normalize: NormalizeMode = "peak"
    bandpass_center_hz: float | None = 600.0
    bandpass_width_hz: float = 500.0
    noise_floor_percentile: float = 20.0
    scale_mode: Literal["stretch", "unit"] = "stretch"
    canonical_units: float = 24.0
    binarize_threshold: float | None = None


def extract_envelope(
    audio: np.ndarray,
    sample_rate: int,
    config: EnvelopeConfig | None = None,
) -> np.ndarray:
    """Convert mono audio to a normalized fixed-length envelope.

    The output is a float32 vector of ``config.length`` samples. Leading and
    trailing low-energy silence are removed before resampling; internal Morse
    gaps are preserved.
    """
    config = config or EnvelopeConfig()
    _validate_config(config, sample_rate)

    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"audio must be a 1D mono array, got shape {x.shape}")
    if x.size == 0 or not np.any(x):
        return np.zeros(config.length, dtype=np.float32)

    envelope = amplitude_envelope(
        x,
        sample_rate,
        method=config.method,
        smoothing_ms=config.smoothing_ms,
        normalize="none",
        bandpass_center_hz=config.bandpass_center_hz,
        bandpass_width_hz=config.bandpass_width_hz,
        noise_floor_percentile=config.noise_floor_percentile,
    )
    envelope = _trim_silence(envelope, sample_rate, config.trim_threshold, config.trim_pad_ms)
    envelope = resample(envelope, config.length).astype(np.float32)
    envelope = np.maximum(envelope, 0.0)
    return _finish_envelope(envelope, config)


def extract_unit_scaled_envelope(
    audio: np.ndarray,
    sample_rate: int,
    unit_samples: int,
    config: EnvelopeConfig | None = None,
) -> np.ndarray:
    """Extract an envelope after normalizing time by estimated Morse dot units."""
    config = config or EnvelopeConfig()
    _validate_config(config, sample_rate)
    if unit_samples <= 0:
        return extract_envelope(audio, sample_rate, config)

    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"audio must be a 1D mono array, got shape {x.shape}")
    if x.size == 0 or not np.any(x):
        return np.zeros(config.length, dtype=np.float32)

    envelope = amplitude_envelope(
        x,
        sample_rate,
        method=config.method,
        smoothing_ms=config.smoothing_ms,
        normalize="none",
        bandpass_center_hz=config.bandpass_center_hz,
        bandpass_width_hz=config.bandpass_width_hz,
        noise_floor_percentile=config.noise_floor_percentile,
    )
    envelope = _trim_silence(envelope, sample_rate, config.trim_threshold, config.trim_pad_ms)
    target_len = max(2, int(round(config.canonical_units * unit_samples)))
    envelope = _center_or_crop(envelope, target_len)
    envelope = resample(envelope, config.length).astype(np.float32)
    envelope = np.maximum(envelope, 0.0)
    return _finish_envelope(envelope, config)


def amplitude_envelope(
    audio: np.ndarray,
    sample_rate: int,
    *,
    method: EnvelopeMethod = "hilbert",
    smoothing_ms: float = 4.0,
    normalize: NormalizeMode = "peak",
    bandpass_center_hz: float | None = 600.0,
    bandpass_width_hz: float = 500.0,
    noise_floor_percentile: float = 20.0,
) -> np.ndarray:
    """Return a full-resolution amplitude envelope for audio analysis."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if smoothing_ms < 0:
        raise ValueError(f"smoothing_ms must be non-negative, got {smoothing_ms}")

    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"audio must be a 1D mono array, got shape {x.shape}")
    if x.size == 0 or not np.any(x):
        return np.zeros(x.shape, dtype=np.float32)

    x = _bandpass(x, sample_rate, bandpass_center_hz, bandpass_width_hz)
    envelope = _raw_envelope(x, method)
    envelope = _smooth(envelope, sample_rate, smoothing_ms)
    envelope = _subtract_noise_floor(envelope, noise_floor_percentile)
    envelope = np.maximum(envelope, 0.0)
    return _normalize(envelope, normalize)


def _validate_config(config: EnvelopeConfig, sample_rate: int) -> None:
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if config.length <= 0:
        raise ValueError(f"length must be positive, got {config.length}")
    if config.smoothing_ms < 0:
        raise ValueError(f"smoothing_ms must be non-negative, got {config.smoothing_ms}")
    if not 0 <= config.trim_threshold <= 1:
        raise ValueError(f"trim_threshold must be in [0, 1], got {config.trim_threshold}")
    if config.trim_pad_ms < 0:
        raise ValueError(f"trim_pad_ms must be non-negative, got {config.trim_pad_ms}")
    if config.bandpass_width_hz <= 0:
        raise ValueError(f"bandpass_width_hz must be positive, got {config.bandpass_width_hz}")
    if not 0 <= config.noise_floor_percentile <= 100:
        raise ValueError(
            "noise_floor_percentile must be in [0, 100], "
            f"got {config.noise_floor_percentile}"
        )
    if config.canonical_units <= 0:
        raise ValueError(f"canonical_units must be positive, got {config.canonical_units}")
    binarize_threshold = getattr(config, "binarize_threshold", None)
    if binarize_threshold is not None and not 0 <= binarize_threshold <= 1:
        raise ValueError(
            "binarize_threshold must be in [0, 1] or None, "
            f"got {binarize_threshold}"
        )


def _raw_envelope(audio: np.ndarray, method: EnvelopeMethod) -> np.ndarray:
    if method == "abs":
        return np.abs(audio).astype(np.float32)
    if method == "hilbert":
        return np.abs(hilbert(audio)).astype(np.float32)
    raise ValueError(f"unknown envelope method: {method!r}")


def _bandpass(
    audio: np.ndarray,
    sample_rate: int,
    center_hz: float | None,
    width_hz: float,
) -> np.ndarray:
    if center_hz is None:
        return audio
    if width_hz <= 0:
        raise ValueError(f"bandpass width must be positive, got {width_hz}")

    nyquist = sample_rate / 2.0
    low = max(1.0, center_hz - width_hz / 2.0)
    high = min(nyquist - 1.0, center_hz + width_hz / 2.0)
    if low >= high:
        return audio

    sos = butter(4, [low, high], btype="band", fs=sample_rate, output="sos")
    # scipy's filtfilt needs padding. Very short clips are better left
    # unfiltered than rejected during segmentation experiments.
    if audio.size <= 3 * (2 * sos.shape[0] + 1):
        return audio
    return sosfiltfilt(sos, audio).astype(np.float32)


def _smooth(envelope: np.ndarray, sample_rate: int, smoothing_ms: float) -> np.ndarray:
    window = int(round(smoothing_ms / 1000.0 * sample_rate))
    if window <= 1:
        return envelope.astype(np.float32, copy=False)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(envelope, kernel, mode="same").astype(np.float32)


def _subtract_noise_floor(envelope: np.ndarray, percentile: float) -> np.ndarray:
    if percentile <= 0:
        return envelope.astype(np.float32, copy=False)
    floor = float(np.percentile(envelope, percentile))
    return np.maximum(envelope - floor, 0.0).astype(np.float32)


def _trim_silence(
    envelope: np.ndarray,
    sample_rate: int,
    threshold_ratio: float,
    pad_ms: float,
) -> np.ndarray:
    peak = float(np.max(envelope))
    if peak <= 0:
        return envelope

    active = np.flatnonzero(envelope >= peak * threshold_ratio)
    if active.size == 0:
        return envelope

    pad = int(round(pad_ms / 1000.0 * sample_rate))
    start = max(0, int(active[0]) - pad)
    end = min(envelope.size, int(active[-1]) + pad + 1)
    return envelope[start:end]


def _center_or_crop(envelope: np.ndarray, target_len: int) -> np.ndarray:
    if envelope.size == target_len:
        return envelope
    if envelope.size > target_len:
        start = max(0, (envelope.size - target_len) // 2)
        return envelope[start : start + target_len]

    out = np.zeros(target_len, dtype=np.float32)
    start = (target_len - envelope.size) // 2
    out[start : start + envelope.size] = envelope
    return out


def _normalize(envelope: np.ndarray, mode: NormalizeMode) -> np.ndarray:
    if mode == "none":
        return envelope.astype(np.float32, copy=False)
    if mode == "peak":
        scale = float(np.max(envelope))
    elif mode == "rms":
        scale = float(np.sqrt(np.mean(envelope**2)))
    else:
        raise ValueError(f"unknown normalize mode: {mode!r}")

    if scale <= 1e-12:
        return np.zeros_like(envelope, dtype=np.float32)
    return (envelope / scale).astype(np.float32)


def _finish_envelope(envelope: np.ndarray, config: EnvelopeConfig) -> np.ndarray:
    envelope = _normalize(envelope, config.normalize)
    binarize_threshold = getattr(config, "binarize_threshold", None)
    if binarize_threshold is None:
        return envelope
    return (envelope >= binarize_threshold).astype(np.float32)
