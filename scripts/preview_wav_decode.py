"""Visual preview of WAV decoding with envelope, segmentation, and CNN labels."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morse_char_recognizer.audio_io import read_wav_mono
from morse_char_recognizer.classes import CLASS_PRESETS, parse_class_spec
from morse_char_recognizer.features import amplitude_envelope
from morse_char_recognizer.inference import (
    join_segment_predictions,
    load_checkpoint_model,
    predict_audio_segments,
)
from morse_char_recognizer.live import (
    default_label_path,
    estimate_carrier_hz,
    read_label,
    slice_audio,
)
from morse_char_recognizer.segment import (
    SegmentConfig,
    detect_active_regions,
    estimate_unit_samples,
)


def main() -> None:
    args = _parse_args()
    sample_rate, full_audio = read_wav_mono(args.wav)
    audio, offset_s = slice_audio(full_audio, sample_rate, args.start, args.duration)
    model, classes, envelope, _checkpoint_args = load_checkpoint_model(
        args.checkpoint,
        device=args.device,
    )
    allowed_classes = parse_class_spec(
        args.allowed_classes,
        preset=args.allowed_class_preset,
    )
    if args.scale_mode != "checkpoint":
        envelope = replace(
            envelope,
            scale_mode=args.scale_mode,
            canonical_units=args.canonical_units,
        )

    center_hz = args.center_hz
    if args.auto_center:
        center_hz = estimate_carrier_hz(audio, sample_rate, args.freq_min, args.freq_max)
    if center_hz is not None:
        envelope = replace(envelope, bandpass_center_hz=center_hz)

    segment_config = SegmentConfig(
        threshold_mode=args.threshold_mode,
        threshold=args.threshold,
        adaptive_window_ms=args.adaptive_window_ms,
        adaptive_floor_percentile=args.adaptive_floor_percentile,
        adaptive_peak_percentile=args.adaptive_peak_percentile,
        adaptive_min_threshold=args.adaptive_min_threshold,
        min_keydown_ms=args.min_keydown_ms,
        merge_gap_ms=args.merge_gap_ms,
        char_gap_units=args.char_gap_units,
        max_character_units=args.max_character_units,
        pad_ms=args.pad_ms,
    )
    predictions = predict_audio_segments(
        model,
        audio,
        sample_rate,
        classes=classes,
        envelope=envelope,
        segment_config=segment_config,
        allowed_classes=allowed_classes,
        device=args.device,
    )
    if args.min_score > 0:
        predictions = [
            (char, score, segment)
            for char, score, segment in predictions
            if score >= args.min_score
        ]
    active_regions = detect_active_regions(audio, sample_rate, segment_config)
    unit_samples = estimate_unit_samples(active_regions)
    if args.min_segment_units > 0 and unit_samples > 0:
        predictions = [
            (char, score, segment)
            for char, score, segment in predictions
            if segment.duration >= args.min_segment_units * unit_samples
        ]
    env = amplitude_envelope(
        audio,
        sample_rate,
        method=envelope.method,
        smoothing_ms=envelope.smoothing_ms,
        bandpass_center_hz=envelope.bandpass_center_hz,
        bandpass_width_hz=envelope.bandpass_width_hz,
        noise_floor_percentile=envelope.noise_floor_percentile,
    )

    decoded = join_segment_predictions(
        predictions,
        sample_rate=sample_rate,
        word_gap_ms=args.word_gap_ms,
        unit_samples=unit_samples if args.word_gap_mode == "unit" else None,
        word_gap_units=args.word_gap_units,
    )
    label = read_label(Path(args.label) if args.label else default_label_path(Path(args.wav)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _plot_preview(
        out,
        audio,
        env,
        sample_rate,
        predictions,
        active_regions,
        decoded,
        label,
        offset_s,
        center_hz,
        args,
    )
    print(f"decoded: {decoded}")
    print(f"segments={len(predictions)} keydowns={len(active_regions)}")
    if center_hz is not None:
        print(f"center_hz={center_hz:.1f}")
    if unit_samples:
        print(f"unit_ms={unit_samples / sample_rate * 1000.0:.1f}")
    print(f"wrote {out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("wav")
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", default="outputs/wav_decode_preview.png")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--threshold-mode", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--threshold", type=float, default=0.18)
    parser.add_argument("--adaptive-window-ms", type=float, default=650.0)
    parser.add_argument("--adaptive-floor-percentile", type=float, default=20.0)
    parser.add_argument("--adaptive-peak-percentile", type=float, default=90.0)
    parser.add_argument("--adaptive-min-threshold", type=float, default=0.03)
    parser.add_argument("--min-keydown-ms", type=float, default=10.0)
    parser.add_argument("--merge-gap-ms", type=float, default=8.0)
    parser.add_argument("--char-gap-units", type=float, default=2.0)
    parser.add_argument("--max-character-units", type=float, default=None)
    parser.add_argument("--pad-ms", type=float, default=20.0)
    parser.add_argument("--word-gap-ms", type=float, default=250.0)
    parser.add_argument("--word-gap-mode", choices=("fixed", "unit"), default="fixed")
    parser.add_argument("--word-gap-units", type=float, default=4.5)
    parser.add_argument(
        "--scale-mode",
        choices=("checkpoint", "stretch", "unit"),
        default="checkpoint",
    )
    parser.add_argument("--canonical-units", type=float, default=24.0)
    parser.add_argument("--center-hz", type=float, default=None)
    parser.add_argument("--auto-center", action="store_true")
    parser.add_argument("--freq-min", type=float, default=250.0)
    parser.add_argument("--freq-max", type=float, default=1200.0)
    parser.add_argument("--spectrogram-max-hz", type=float, default=1400.0)
    parser.add_argument("--allowed-class-preset", choices=tuple(CLASS_PRESETS), default="real")
    parser.add_argument("--allowed-classes", default=None)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-segment-units", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _plot_preview(
    out: Path,
    audio: np.ndarray,
    envelope: np.ndarray,
    sample_rate: int,
    predictions,
    active_regions,
    decoded: str,
    label: str,
    offset_s: float,
    center_hz: float | None,
    args: argparse.Namespace,
) -> None:
    t = offset_s + np.arange(audio.size) / sample_rate
    duration_s = audio.size / sample_rate if sample_rate > 0 else 0.0
    width = max(16.0, min(40.0, duration_s * 0.55))
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(width, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.2, 2.0, 1.0]},
    )
    fig.subplots_adjust(left=0.055, right=0.995, top=0.93, bottom=0.055, hspace=0.22)

    _plot_waveform(axes[0], t, audio)
    _plot_envelope(axes[1], t, envelope, sample_rate, active_regions, predictions, offset_s)
    _plot_spectrogram(axes[2], audio, sample_rate, offset_s, args.spectrogram_max_hz)
    _plot_text_panel(axes[3], decoded, label, center_hz)

    title = f"{Path(args.wav).name}  start={args.start:.2f}s duration={args.duration:.2f}s"
    fig.suptitle(title, fontsize=13)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_waveform(ax, t: np.ndarray, audio: np.ndarray) -> None:
    ax.plot(t, audio, color="#4c566a", linewidth=0.5)
    ax.set_ylabel("audio")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(axis="x", color="#dddddd", linewidth=0.4)


def _plot_envelope(
    ax,
    t: np.ndarray,
    envelope: np.ndarray,
    sample_rate: int,
    active_regions,
    predictions,
    offset_s: float,
) -> None:
    ax.plot(t, envelope, color="#1f77b4", linewidth=1.0)
    ax.fill_between(t, envelope, color="#1f77b4", alpha=0.16)
    for region in active_regions:
        ax.axvspan(
            offset_s + region.start / sample_rate,
            offset_s + region.end / sample_rate,
            color="#2ca02c",
            alpha=0.12,
        )
    for char, score, segment in predictions:
        start = offset_s + segment.start / sample_rate
        end = offset_s + segment.end / sample_rate
        ax.axvline(start, color="#d62728", linewidth=0.7)
        ax.axvline(end, color="#d62728", linewidth=0.7)
        ax.text(
            (start + end) / 2,
            1.04,
            f"{char}\n{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#d62728",
        )
    ax.set_ylim(-0.05, 1.2)
    ax.set_ylabel("envelope")
    ax.grid(axis="x", color="#dddddd", linewidth=0.4)


def _plot_spectrogram(
    ax,
    audio: np.ndarray,
    sample_rate: int,
    offset_s: float,
    max_hz: float,
) -> None:
    if audio.size < 128:
        ax.set_ylabel("spectrogram")
        return
    nperseg = min(512, max(128, audio.size // 100))
    noverlap = int(nperseg * 0.75)
    frequencies, times, power = spectrogram(
        audio,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )
    mask = frequencies <= max_hz
    image = np.log1p(power[mask])
    ax.imshow(
        image,
        aspect="auto",
        origin="lower",
        extent=[
            offset_s + times[0],
            offset_s + times[-1],
            frequencies[mask][0],
            frequencies[mask][-1],
        ],
        cmap="magma",
    )
    ax.set_ylabel("Hz")


def _plot_text_panel(ax, decoded: str, label: str, center_hz: float | None) -> None:
    ax.axis("off")
    center = "auto n/a" if center_hz is None else f"{center_hz:.1f} Hz"
    label_preview = label[:500] + ("..." if len(label) > 500 else "")
    text = f"decoded: {decoded}\ncenter: {center}"
    if label_preview:
        text += f"\nlabel:   {label_preview}"
    ax.text(0.01, 0.92, text, va="top", ha="left", family="monospace", fontsize=10)


if __name__ == "__main__":
    main()
