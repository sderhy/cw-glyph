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
    predictions = predict_audio_segments(
        model,
        audio,
        sample_rate,
        classes=classes,
        envelope=envelope,
        allowed_classes=allowed_classes,
        device=args.device,
    )
    decoded = join_segment_predictions(
        predictions,
        sample_rate=sample_rate,
        word_gap_ms=args.word_gap_ms,
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
    parser.add_argument("--word-gap-ms", type=float, default=250.0)
    parser.add_argument(
        "--scale-mode",
        choices=("checkpoint", "stretch", "unit"),
        default="checkpoint",
    )
    parser.add_argument("--canonical-units", type=float, default=24.0)
    parser.add_argument("--allowed-class-preset", choices=tuple(CLASS_PRESETS), default="real")
    parser.add_argument("--allowed-classes", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
