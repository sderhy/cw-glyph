#!/usr/bin/env python3
"""Decode a WAV with the element-level beam/DP Morse decoder and report CER.

Unlike the CNN glyph pipeline, this decoder needs no checkpoint: it groups
detected dit/dah elements into valid Morse characters by global DP, using gaps
as a soft prior. Use it to probe how much of the error is segmentation vs the
classifier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from morse_char_recognizer.audio_io import read_wav_mono
from morse_char_recognizer.beam_decode import BeamConfig, beam_decode_audio
from morse_char_recognizer.classes import CLASS_PRESETS, parse_class_spec
from morse_char_recognizer.live import (
    compare_text,
    default_label_path,
    estimate_carrier_hz,
    read_label,
)
from morse_char_recognizer.preview import FrequencyOverlay, write_spectrogram_png
from morse_char_recognizer.segment import SegmentConfig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("wav")
    p.add_argument("--label", default=None, help="truth .txt (default: sibling file)")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=0.0, help="0 = whole file")
    # segmentation / element detection
    p.add_argument("--threshold", type=float, default=0.22)
    p.add_argument(
        "--threshold-mode",
        choices=["fixed", "adaptive", "hysteresis"],
        default="hysteresis",
    )
    p.add_argument("--hysteresis-low", type=float, default=0.12)
    p.add_argument("--bandpass-center-hz", type=float, default=None)
    p.add_argument("--bandpass-width-hz", type=float, default=100.0)
    p.add_argument("--auto-center", action="store_true")
    p.add_argument("--freq-min", type=float, default=400.0)
    p.add_argument("--freq-max", type=float, default=900.0)
    p.add_argument("--min-keydown-ms", type=float, default=10.0)
    p.add_argument("--merge-gap-ms", type=float, default=8.0)
    p.add_argument("--min-unit-ms", type=float, default=35.0)
    p.add_argument("--max-unit-ms", type=float, default=140.0)
    # beam / DP scoring
    p.add_argument("--dot-dash-split", type=float, default=1.8)
    p.add_argument("--char-cost", type=float, default=0.0)
    p.add_argument("--boundary-threshold", type=float, default=2.0)
    p.add_argument("--w-internal", type=float, default=1.0)
    p.add_argument("--w-boundary", type=float, default=1.0)
    p.add_argument("--word-gap-units", type=float, default=5.0)
    p.add_argument("--allowed-class-preset", choices=tuple(CLASS_PRESETS), default=None)
    p.add_argument("--allowed-classes", default=None)
    p.add_argument("--json-out", default=None)
    p.add_argument("--preview-out", default=None)
    p.add_argument("--preview-min-hz", type=float, default=0.0)
    p.add_argument("--preview-max-hz", type=float, default=1400.0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    wav_path = Path(args.wav)
    sample_rate, audio = read_wav_mono(wav_path)

    start = int(args.start * sample_rate)
    end = len(audio) if args.duration <= 0 else start + int(args.duration * sample_rate)
    audio = audio[start:end]

    center = args.bandpass_center_hz
    if args.auto_center:
        center = estimate_carrier_hz(audio, sample_rate, args.freq_min, args.freq_max)
    if center is None:
        center = 600.0

    segment_config = SegmentConfig(
        threshold_mode=args.threshold_mode,
        threshold=args.threshold,
        hysteresis_low=args.hysteresis_low,
        bandpass_center_hz=center,
        bandpass_width_hz=args.bandpass_width_hz,
        min_keydown_ms=args.min_keydown_ms,
        merge_gap_ms=args.merge_gap_ms,
        min_unit_ms=args.min_unit_ms,
        max_unit_ms=args.max_unit_ms,
    )
    beam_config = BeamConfig(
        dot_dash_split=args.dot_dash_split,
        char_cost=args.char_cost,
        boundary_threshold=args.boundary_threshold,
        w_internal=args.w_internal,
        w_boundary=args.w_boundary,
        word_gap_units=args.word_gap_units,
        allowed_tokens=(
            parse_class_spec(args.allowed_classes, preset=args.allowed_class_preset)
            if args.allowed_class_preset is not None or args.allowed_classes is not None
            else None
        ),
    )

    decoded, regions, unit_samples = beam_decode_audio(
        audio,
        sample_rate,
        segment_config=segment_config,
        beam_config=beam_config,
    )

    label_path = Path(args.label) if args.label else default_label_path(wav_path)
    reference = read_label(label_path)

    unit_ms = 1000.0 * unit_samples / sample_rate
    print(
        f"center_hz={center:.1f} elements={len(regions)} "
        f"unit={unit_samples}smp ({unit_ms:.1f}ms)"
    )
    print(f"decoded: {decoded}")

    summary = {
        "wav": str(wav_path),
        "center_hz": center,
        "elements": len(regions),
        "unit_samples": unit_samples,
        "decoded": decoded,
        "reference": reference,
        "allowed_class_preset": args.allowed_class_preset,
        "allowed_classes": args.allowed_classes,
    }
    if reference:
        cmp = compare_text(reference, decoded)
        print(f"reference: {reference}")
        print(
            f"CER={cmp.cer:.3f} accuracy={cmp.accuracy:.3f} "
            f"subs={cmp.substitutions} ins={cmp.insertions} del={cmp.deletions} "
            f"dist={cmp.distance}/{cmp.reference_length}"
        )
        summary.update(
            cer=cmp.cer,
            accuracy=cmp.accuracy,
            substitutions=cmp.substitutions,
            insertions=cmp.insertions,
            deletions=cmp.deletions,
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"wrote {args.json_out}")

    if args.preview_out:
        out = write_spectrogram_png(
            audio,
            sample_rate,
            args.preview_out,
            title=f"{wav_path.name} beam decode",
            min_hz=args.preview_min_hz,
            max_hz=args.preview_max_hz,
            offset_s=args.start,
            overlays=[
                FrequencyOverlay(
                    args.start,
                    args.start + audio.size / sample_rate,
                    center,
                    f"{center:.1f} Hz",
                )
            ],
            decoded=decoded,
            reference=reference,
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
