"""Diagnostic plots for Morse/CW audio decoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import spectrogram


@dataclass(frozen=True)
class FrequencyOverlay:
    """Horizontal frequency guide over a time interval."""

    start_s: float
    end_s: float
    center_hz: float
    label: str | None = None


@dataclass(frozen=True)
class FrequencyPoint:
    """One point in a tracked center-frequency curve."""

    time_s: float
    center_hz: float


def write_spectrogram_png(
    audio: np.ndarray,
    sample_rate: int,
    out: str | Path,
    *,
    title: str | None = None,
    min_hz: float = 0.0,
    max_hz: float = 1400.0,
    offset_s: float = 0.0,
    overlays: list[FrequencyOverlay] | None = None,
    points: list[FrequencyPoint] | None = None,
    decoded: str | None = None,
    reference: str | None = None,
) -> Path:
    """Write a grayscale time/frequency spectrogram PNG."""
    import matplotlib.pyplot as plt

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(12, 4.2))
    _plot_grayscale_spectrogram(axis, audio, sample_rate, offset_s, min_hz, max_hz)
    for overlay in overlays or []:
        axis.hlines(
            overlay.center_hz,
            overlay.start_s,
            overlay.end_s,
            colors="#1f77b4",
            linewidth=1.4,
        )
        if overlay.label:
            axis.text(
                overlay.start_s,
                overlay.center_hz,
                overlay.label,
                va="bottom",
                ha="left",
                color="#1f77b4",
                fontsize=8,
            )
    if points:
        axis.plot(
            [point.time_s for point in points],
            [point.center_hz for point in points],
            marker=".",
            linestyle="none",
            color="#d62728",
            markersize=3,
        )
    axis.set_ylim(min_hz, max_hz)
    axis.set_xlabel("time (s)")
    axis.set_ylabel("frequency (Hz)")
    if title:
        axis.set_title(title)
    if decoded or reference:
        text = _preview_text(decoded, reference)
        fig.text(0.01, 0.01, text, ha="left", va="bottom", family="monospace", fontsize=8)
        fig.subplots_adjust(bottom=0.23)
    else:
        fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _plot_grayscale_spectrogram(
    axis: Any,
    audio: np.ndarray,
    sample_rate: int,
    offset_s: float,
    min_hz: float,
    max_hz: float,
) -> None:
    if audio.size < 128:
        axis.set_xlim(offset_s, offset_s)
        return
    nperseg = min(512, max(128, audio.size // 250))
    noverlap = int(nperseg * 0.75)
    frequencies, times, power = spectrogram(
        audio,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )
    mask = (frequencies >= min_hz) & (frequencies <= max_hz)
    if not mask.any():
        return
    image = 20.0 * np.log10(np.maximum(power[mask], 1e-12))
    low = float(np.percentile(image, 5.0))
    high = float(np.percentile(image, 99.5))
    if high <= low:
        high = low + 1.0
    axis.imshow(
        image,
        aspect="auto",
        origin="lower",
        extent=[
            offset_s,
            offset_s + audio.size / sample_rate,
            float(frequencies[mask][0]),
            float(frequencies[mask][-1]),
        ],
        cmap="gray",
        vmin=low,
        vmax=high,
    )


def _preview_text(decoded: str | None, reference: str | None) -> str:
    parts = []
    if decoded:
        parts.append(f"decoded:   {_truncate(decoded)}")
    if reference:
        parts.append(f"reference: {_truncate(reference)}")
    return "\n".join(parts)


def _truncate(text: str, *, limit: int = 120) -> str:
    return text[:limit] + ("..." if len(text) > limit else "")
