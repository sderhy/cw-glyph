"""Run the template baseline on synthetic continuous Morse audio."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morse_char_recognizer.baseline import TemplateConfig
from morse_char_recognizer.classes import CLASS_PRESETS, parse_class_spec
from morse_char_recognizer.decode import decode_audio, decoded_text
from morse_char_recognizer.features import EnvelopeConfig
from morse_synth.core import synthesize


def main() -> None:
    args = _parse_args()
    envelope = EnvelopeConfig(length=args.length)
    audio = synthesize(
        args.text,
        wpm=args.wpm,
        freq=args.freq,
        sample_rate=args.sample_rate,
        rise_ms=args.rise_ms,
    )
    decoded = decode_audio(
        audio,
        args.sample_rate,
        envelope_config=envelope,
        template_config=TemplateConfig(
            classes=parse_class_spec(args.classes, preset=args.class_preset),
            sample_rate=args.sample_rate,
            envelope=envelope,
            wpm=args.wpm,
            freq=args.freq,
            rise_ms=args.rise_ms,
        ),
    )

    print(f"input:   {args.text}")
    print(f"decoded: {decoded_text(decoded)}")
    for index, item in enumerate(decoded, start=1):
        start_s = item.segment.start / args.sample_rate
        end_s = item.segment.end / args.sample_rate
        print(f"{index:02d} {item.char} score={item.score:.3f} segment={start_s:.3f}-{end_s:.3f}s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="SOS")
    parser.add_argument("--class-preset", choices=tuple(CLASS_PRESETS), default="basic")
    parser.add_argument("--classes", default=None)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--wpm", type=float, default=20.0)
    parser.add_argument("--freq", type=float, default=600.0)
    parser.add_argument("--rise-ms", type=float, default=3.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
