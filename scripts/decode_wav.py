"""Decode a WAV file with a trained Morse glyph CNN checkpoint."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morse_char_recognizer.audio_io import read_wav_mono
from morse_char_recognizer.classes import CLASS_PRESETS, parse_class_spec
from morse_char_recognizer.inference import (
    filter_short_segment_predictions,
    join_segment_predictions,
    load_checkpoint_model,
    predict_audio_segments,
)
from morse_char_recognizer.live import estimate_carrier_hz
from morse_char_recognizer.segment import (
    SegmentConfig,
    detect_active_regions,
    estimate_unit_samples,
)


def main() -> None:
    args = _parse_args()
    sample_rate, audio = read_wav_mono(args.wav)
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
    if args.bandpass_width_hz is not None:
        envelope = replace(envelope, bandpass_width_hz=args.bandpass_width_hz)
    segment_config = SegmentConfig(
        bandpass_center_hz=envelope.bandpass_center_hz,
        bandpass_width_hz=envelope.bandpass_width_hz,
        noise_floor_percentile=envelope.noise_floor_percentile,
        threshold_mode=args.threshold_mode,
        threshold=args.threshold,
        adaptive_window_ms=args.adaptive_window_ms,
        adaptive_floor_percentile=args.adaptive_floor_percentile,
        adaptive_peak_percentile=args.adaptive_peak_percentile,
        adaptive_min_threshold=args.adaptive_min_threshold,
        min_keydown_ms=args.min_keydown_ms,
        merge_gap_ms=args.merge_gap_ms,
        min_unit_ms=args.min_unit_ms,
        max_unit_ms=args.max_unit_ms,
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
    unit_samples = estimate_unit_samples(
        detect_active_regions(audio, sample_rate, segment_config),
        sample_rate,
        min_unit_ms=segment_config.min_unit_ms,
        max_unit_ms=segment_config.max_unit_ms,
        histogram_bin_ms=segment_config.unit_histogram_bin_ms,
    )
    if args.min_segment_units > 0 and unit_samples > 0:
        predictions = filter_short_segment_predictions(
            predictions,
            unit_samples=unit_samples,
            min_segment_units=args.min_segment_units,
            short_class_min_segment_units=args.short_class_min_segment_units,
            short_classes=_short_classes(args.short_classes),
            short_class_min_score=args.short_class_min_score,
        )
    decoded = join_segment_predictions(
        predictions,
        sample_rate=sample_rate,
        word_gap_ms=args.word_gap_ms,
        unit_samples=unit_samples if args.word_gap_mode == "unit" else None,
        word_gap_units=args.word_gap_units,
    )

    print(decoded)
    if args.verbose:
        for index, (char, score, segment) in enumerate(predictions, start=1):
            start_s = segment.start / sample_rate
            end_s = segment.end / sample_rate
            print(f"{index:03d} {char} score={score:.3f} segment={start_s:.3f}-{end_s:.3f}s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("wav")
    parser.add_argument("--threshold-mode", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--threshold", type=float, default=0.18)
    parser.add_argument("--adaptive-window-ms", type=float, default=650.0)
    parser.add_argument("--adaptive-floor-percentile", type=float, default=20.0)
    parser.add_argument("--adaptive-peak-percentile", type=float, default=90.0)
    parser.add_argument("--adaptive-min-threshold", type=float, default=0.03)
    parser.add_argument("--min-keydown-ms", type=float, default=10.0)
    parser.add_argument("--merge-gap-ms", type=float, default=8.0)
    parser.add_argument("--min-unit-ms", type=float, default=0.0)
    parser.add_argument("--max-unit-ms", type=float, default=140.0)
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
    parser.add_argument("--bandpass-width-hz", type=float, default=None)
    parser.add_argument("--allowed-class-preset", choices=tuple(CLASS_PRESETS), default="real")
    parser.add_argument("--allowed-classes", default=None)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-segment-units", type=float, default=0.0)
    parser.add_argument("--short-class-min-segment-units", type=float, default=None)
    parser.add_argument("--short-classes", default="E,T")
    parser.add_argument("--short-class-min-score", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _short_classes(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    main()
