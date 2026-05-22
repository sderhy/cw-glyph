"""Decode synthetic continuous Morse audio with a trained CNN checkpoint."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morse_char_recognizer.classes import CLASS_PRESETS, parse_class_spec
from morse_char_recognizer.inference import (
    join_segment_predictions,
    load_checkpoint_model,
    predict_audio_segments,
)
from morse_synth.channel import ChannelConfig
from morse_synth.core import render, synthesize


def main() -> None:
    args = _parse_args()
    model, classes, envelope, _checkpoint_args = load_checkpoint_model(
        args.checkpoint,
        device=args.device,
    )
    allowed_classes = parse_class_spec(
        args.allowed_classes,
        preset=args.allowed_class_preset,
    )

    if args.snr_db is None:
        audio = synthesize(
            args.text,
            wpm=args.wpm,
            freq=args.freq,
            sample_rate=args.sample_rate,
            rise_ms=args.rise_ms,
        )
    else:
        from morse_synth.keying import KeyingConfig
        from morse_synth.operator import OperatorConfig

        audio = render(
            args.text,
            operator=OperatorConfig(wpm=args.wpm),
            keying=KeyingConfig(rise_ms=args.rise_ms),
            channel=ChannelConfig(snr_db=args.snr_db, seed=args.seed),
            freq=args.freq,
            sample_rate=args.sample_rate,
        )

    if args.edge_pad_ms > 0:
        pad = np.zeros(int(round(args.edge_pad_ms / 1000.0 * args.sample_rate)), dtype=np.float32)
        audio = np.concatenate([pad, audio.astype(np.float32, copy=False), pad])

    predictions = predict_audio_segments(
        model,
        audio,
        args.sample_rate,
        classes=classes,
        envelope=envelope,
        allowed_classes=allowed_classes,
        device=args.device,
    )

    decoded = join_segment_predictions(
        predictions,
        sample_rate=args.sample_rate,
        word_gap_ms=args.word_gap_ms,
    )
    expected = args.text.upper()
    print(f"input:    {args.text}")
    print(f"expected: {expected}")
    print(f"decoded:  {decoded}")
    snr_label = "clean" if args.snr_db is None or math.isinf(args.snr_db) else f"{args.snr_db:g}"
    print(f"snr_db:   {snr_label}")
    for index, (char, score, segment) in enumerate(predictions, start=1):
        start_s = segment.start / args.sample_rate
        end_s = segment.end / args.sample_rate
        print(f"{index:02d} {char} score={score:.3f} segment={start_s:.3f}-{end_s:.3f}s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--text", default="CQ TEST")
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--wpm", type=float, default=20.0)
    parser.add_argument("--freq", type=float, default=600.0)
    parser.add_argument("--rise-ms", type=float, default=3.0)
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--edge-pad-ms", type=float, default=200.0)
    parser.add_argument("--word-gap-ms", type=float, default=250.0)
    parser.add_argument("--allowed-class-preset", choices=tuple(CLASS_PRESETS), default="real")
    parser.add_argument("--allowed-classes", default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()
