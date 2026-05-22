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
    join_segment_predictions,
    load_checkpoint_model,
    predict_audio_segments,
)
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
    unit_samples = estimate_unit_samples(detect_active_regions(audio, sample_rate, segment_config))
    if args.min_segment_units > 0 and unit_samples > 0:
        predictions = [
            (char, score, segment)
            for char, score, segment in predictions
            if segment.duration >= args.min_segment_units * unit_samples
        ]
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
    parser.add_argument("--char-gap-units", type=float, default=2.0)
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
    parser.add_argument("--allowed-class-preset", choices=tuple(CLASS_PRESETS), default="real")
    parser.add_argument("--allowed-classes", default=None)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-segment-units", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
