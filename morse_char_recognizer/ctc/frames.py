"""Audio → fixed-rate envelope frames for the CTC recogniser.

The CTC network does not need per-character segmentation. It needs a 1D
time-series whose temporal resolution is fine enough to expose dits and
gaps, but coarse enough to keep RNN sequence lengths small. ``FrameConfig``
controls that trade-off.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

from morse_char_recognizer.features import amplitude_envelope


@dataclass(frozen=True)
class FrameConfig:
    """Parameters for the envelope-frame encoder."""

    frame_rate_hz: float = 200.0
    smoothing_ms: float = 4.0
    bandpass_center_hz: float | None = 600.0
    bandpass_width_hz: float = 500.0
    noise_floor_percentile: float = 20.0
    normalize: bool = True


def audio_to_frames(
    audio: np.ndarray,
    sample_rate: int,
    config: FrameConfig | None = None,
) -> np.ndarray:
    """Convert mono audio to a 1D envelope frame sequence.

    Returns a float32 array of shape ``(num_frames,)``.
    """
    config = config or FrameConfig()
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if config.frame_rate_hz <= 0:
        raise ValueError(f"frame_rate_hz must be positive, got {config.frame_rate_hz}")

    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"audio must be 1D, got shape {x.shape}")
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)

    envelope = amplitude_envelope(
        x,
        sample_rate,
        smoothing_ms=config.smoothing_ms,
        normalize="none",
        bandpass_center_hz=config.bandpass_center_hz,
        bandpass_width_hz=config.bandpass_width_hz,
        noise_floor_percentile=config.noise_floor_percentile,
    )
    frames = _resample_to_frame_rate(envelope, sample_rate, config.frame_rate_hz)
    frames = np.maximum(frames, 0.0)
    if config.normalize:
        peak = float(np.max(frames))
        if peak > 1e-12:
            frames = frames / peak
    return frames.astype(np.float32, copy=False)


def _resample_to_frame_rate(
    envelope: np.ndarray,
    sample_rate: int,
    frame_rate_hz: float,
) -> np.ndarray:
    if envelope.size == 0:
        return envelope.astype(np.float32, copy=False)
    target = max(1, int(round(envelope.size * frame_rate_hz / sample_rate)))
    up, down = _ratio(target, envelope.size)
    return resample_poly(envelope, up, down).astype(np.float32)


def _ratio(target: int, source: int) -> tuple[int, int]:
    from math import gcd

    if source <= 0:
        raise ValueError("source size must be positive")
    g = gcd(target, source)
    return max(1, target // g), max(1, source // g)
