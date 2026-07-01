#!/usr/bin/env python3
"""Harvest real, span-level glyph examples from labeled livetest audio.

Forced-aligns detected elements to each file's truth text and extracts a
unit-scaled envelope for every *cleanly aligned* character (a real positive
example of that glyph) plus adjacent-pair merges that form a valid glyph
(real <JUNK> negatives). The output .npz feeds the CNN fine-tuning that closes
the synthetic->real domain gap.

Usage:
  python scripts/harvest_real_spans.py CKPT --out real.npz \
      livetests/g3ses/*.wav livetests/QSOs/*.wav livetests/FAV22/*.wav
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morse_char_recognizer.audio_io import read_wav_mono
from morse_char_recognizer.beam_decode import _lookup, classify_elements, filter_regions
from morse_char_recognizer.classes import morse_code_for_class
from morse_char_recognizer.inference import load_checkpoint_model
from morse_char_recognizer.live import default_label_path, estimate_carrier_hz, read_label, tokenize_copy_text
from morse_char_recognizer.real_align import clean_char_spans
from morse_char_recognizer.segment import (
    Region,
    SegmentConfig,
    detect_active_regions,
    estimate_unit_samples,
    extract_segment_envelopes,
)
from morse_char_recognizer.span_dataset import JUNK_TOKEN

MAX_ELEMENTS_PER_CHAR = 8
MIN_ELEMENT_UNITS = 0.45
DOT_DASH_SPLIT = 2.0
PAD_MS = 20.0


def _truth_tokens(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for tok in tokenize_copy_text(text):
        code = morse_code_for_class(tok)
        if code:
            out.append((tok, code))
    return out


def harvest_file(wav: Path, envelope, junk_per_positive: float):
    sample_rate, audio = read_wav_mono(wav)
    truth = read_label(default_label_path(wav))
    truth_tokens = _truth_tokens(truth)
    if not truth_tokens:
        return [], [], 0

    center = estimate_carrier_hz(audio, sample_rate, 400.0, 900.0) or 600.0
    env = replace(envelope, bandpass_center_hz=center)
    seg = SegmentConfig(
        bandpass_center_hz=center, bandpass_width_hz=100.0, min_unit_ms=35.0, max_unit_ms=140.0
    )
    regions = detect_active_regions(audio, sample_rate, seg)
    unit = estimate_unit_samples(regions, sample_rate, min_unit_ms=35.0, max_unit_ms=140.0)
    filtered = filter_regions(regions, unit, MIN_ELEMENT_UNITS)
    if not filtered or unit <= 0:
        return [], [], len(truth_tokens)
    symbols = classify_elements(filtered, unit, dot_dash_split=DOT_DASH_SPLIT)

    positives = clean_char_spans(symbols, truth_tokens)  # (token, lo, hi) elem indices

    # real junk: merge adjacent cleanly-aligned chars that are contiguous in the
    # element stream and whose combined pattern is still a valid glyph.
    junks: list[tuple[int, int]] = []
    for a, b in zip(positives, positives[1:]):
        _t1, lo1, hi1 = a
        _t2, lo2, hi2 = b
        if hi1 == lo2 and (hi2 - lo1) <= MAX_ELEMENTS_PER_CHAR:
            if _lookup("".join(symbols[lo1:hi2]), prefer_letters=True) is not None:
                junks.append((lo1, hi2))
    max_junk = int(round(junk_per_positive * max(1, len(positives))))
    if len(junks) > max_junk:
        junks = junks[:max_junk]

    pad = int(round(PAD_MS / 1000.0 * sample_rate))
    span_regions = []
    labels: list[str] = []
    for tok, lo, hi in positives:
        span_regions.append(Region(max(0, filtered[lo].start - pad), min(len(audio), filtered[hi - 1].end + pad)))
        labels.append(tok)
    for lo, hi in junks:
        span_regions.append(Region(max(0, filtered[lo].start - pad), min(len(audio), filtered[hi - 1].end + pad)))
        labels.append(JUNK_TOKEN)
    envs = extract_segment_envelopes(audio, sample_rate, span_regions, env, unit_samples=unit)
    return envs, labels, len(truth_tokens)


def main() -> None:
    args = _parse_args()
    _model, _classes, envelope, _a = load_checkpoint_model(args.checkpoint, device="cpu")
    files: list[Path] = []
    for pattern in args.wavs:
        files.extend(sorted(Path(p) for p in glob.glob(pattern)))

    all_env: list[np.ndarray] = []
    all_lbl: list[str] = []
    truth_total = 0
    per_class = Counter()
    for wav in files:
        envs, labels, n_truth = harvest_file(wav, envelope, args.junk_per_positive)
        all_env.extend(envs)
        all_lbl.extend(labels)
        truth_total += n_truth
        per_class.update(labels)
        pos = sum(1 for l in labels if l != JUNK_TOKEN)
        print(f"{wav.name:24} truth={n_truth:4d} clean_pos={pos:4d} junk={len(labels) - pos:4d}")

    pos_total = sum(1 for l in all_lbl if l != JUNK_TOKEN)
    print("-" * 60)
    print(f"files={len(files)} truth_chars={truth_total} clean_positives={pos_total} "
          f"({100.0 * pos_total / max(1, truth_total):.1f}% coverage) "
          f"junk={len(all_lbl) - pos_total}")
    covered = {c for c in per_class if c != JUNK_TOKEN}
    print(f"classes_covered={len(covered)}")

    if args.out and all_env:
        x = np.asarray(all_env, dtype=np.float32)
        y = np.asarray(all_lbl, dtype=object)
        np.savez_compressed(args.out, x=x, y=y)
        print(f"wrote {args.out}  X={x.shape}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="checkpoint to source the EnvelopeConfig from")
    p.add_argument("wavs", nargs="+", help="wav files or globs")
    p.add_argument("--out", default=None)
    p.add_argument("--junk-per-positive", type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    main()
