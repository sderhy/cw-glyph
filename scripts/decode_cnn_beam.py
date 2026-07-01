#!/usr/bin/env python3
"""Decode a WAV with the CNN-scored element-level beam decoder and report CER.

Like ``scripts/decode_beam.py`` this groups detected dit/dah elements into valid
Morse glyphs by global DP, but it also loads a trained glyph CNN checkpoint and
adds the CNN log-probability of each candidate glyph to the search. The CNN only
reweights among *legal* segmentations, so it breaks timing ambiguities
(``EDIKE`` -> ``LIKE``) without inventing illegal glyphs. Use ``--w-cnn 0`` to
recover the pure-timing decoder.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morse_char_recognizer.audio_io import read_wav_mono
from morse_char_recognizer.beam_decode import BeamConfig
from morse_char_recognizer.cnn_beam import CnnBeamConfig, cnn_beam_decode_audio
from morse_char_recognizer.inference import load_checkpoint_model
from morse_char_recognizer.live import (
    compare_text,
    default_label_path,
    estimate_carrier_hz,
    read_label,
)
from morse_char_recognizer.segment import SegmentConfig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("wav")
    p.add_argument("--label", default=None, help="truth .txt (default: sibling file)")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=0.0, help="0 = whole file")
    # segmentation / element detection
    p.add_argument("--threshold", type=float, default=0.22)
    p.add_argument("--threshold-mode", choices=["fixed", "adaptive", "hysteresis"], default="hysteresis")
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
    p.add_argument("--dot-dash-split", type=float, default=2.0)
    p.add_argument("--char-cost", type=float, default=0.0)
    p.add_argument("--boundary-threshold", type=float, default=2.0)
    p.add_argument("--w-internal", type=float, default=1.0)
    p.add_argument("--w-boundary", type=float, default=1.0)
    p.add_argument("--word-gap-units", type=float, default=5.0)
    # CNN scoring
    p.add_argument("--w-cnn", type=float, default=1.0, help="weight on CNN log-prob (0 = off)")
    p.add_argument("--char-bonus", type=float, default=0.0, help="per-char reward vs merge bias")
    p.add_argument("--neutral-logprob", type=float, default=-4.0)
    p.add_argument("--pad-ms", type=float, default=20.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--json-out", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    wav_path = Path(args.wav)
    sample_rate, audio = read_wav_mono(wav_path)

    start = int(args.start * sample_rate)
    end = len(audio) if args.duration <= 0 else start + int(args.duration * sample_rate)
    audio = audio[start:end]

    model, classes, envelope, _checkpoint_args = load_checkpoint_model(
        args.checkpoint, device=args.device
    )

    center = args.bandpass_center_hz
    if args.auto_center:
        center = estimate_carrier_hz(audio, sample_rate, args.freq_min, args.freq_max)
    if center is None:
        center = 600.0
    envelope = replace(envelope, bandpass_center_hz=center)

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
    )
    cnn_config = CnnBeamConfig(
        w_cnn=args.w_cnn,
        char_bonus=args.char_bonus,
        neutral_logprob=args.neutral_logprob,
        pad_ms=args.pad_ms,
    )

    decoded, regions, unit_samples = cnn_beam_decode_audio(
        audio,
        sample_rate,
        model,
        classes,
        envelope,
        segment_config=segment_config,
        beam_config=beam_config,
        cnn_config=cnn_config,
        device=args.device,
    )

    label_path = Path(args.label) if args.label else default_label_path(wav_path)
    reference = read_label(label_path)

    unit_ms = 1000.0 * unit_samples / sample_rate if unit_samples else 0.0
    print(
        f"center_hz={center:.1f} elements={len(regions)} "
        f"unit={unit_samples}smp ({unit_ms:.1f}ms) w_cnn={args.w_cnn:g}"
    )
    print(f"decoded: {decoded}")

    summary = {
        "wav": str(wav_path),
        "checkpoint": str(args.checkpoint),
        "center_hz": center,
        "elements": len(regions),
        "unit_samples": unit_samples,
        "w_cnn": args.w_cnn,
        "decoded": decoded,
        "reference": reference,
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


if __name__ == "__main__":
    main()
