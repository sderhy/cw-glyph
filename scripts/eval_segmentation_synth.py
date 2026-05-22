"""Evaluate character segmentation on synthetic continuous Morse audio."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morse_char_recognizer.segment import Region, SegmentConfig, segment_characters
from morse_char_recognizer.segmentation_metrics import evaluate_segmentation
from morse_synth.core import synthesize


def main() -> None:
    args = _parse_args()
    audio, truth = _synthetic_sequence(
        args.text,
        sample_rate=args.sample_rate,
        wpm=args.wpm,
        char_gap_ms=args.char_gap_ms,
        word_gap_ms=args.word_gap_ms,
    )
    config = SegmentConfig(
        threshold=args.threshold,
        char_gap_units=args.char_gap_units,
        pad_ms=args.pad_ms,
    )
    predicted = segment_characters(audio, args.sample_rate, config)
    report = evaluate_segmentation(truth, predicted, args.sample_rate)
    print(
        f"truth={report.true_count} predicted={report.predicted_count} "
        f"matched={report.matched} missed={report.missed} split={report.split} "
        f"merge={report.merge} false_positive={report.false_positive} "
        f"boundary_ms={report.mean_boundary_error_ms:.1f} "
        f"precision={report.precision:.3f} recall={report.recall:.3f}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="CQ TEST 599")
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--wpm", type=float, default=20.0)
    parser.add_argument("--char-gap-ms", type=float, default=180.0)
    parser.add_argument("--word-gap-ms", type=float, default=420.0)
    parser.add_argument("--threshold", type=float, default=0.18)
    parser.add_argument("--char-gap-units", type=float, default=2.3)
    parser.add_argument("--pad-ms", type=float, default=20.0)
    return parser.parse_args()


def _synthetic_sequence(
    text: str,
    *,
    sample_rate: int,
    wpm: float,
    char_gap_ms: float,
    word_gap_ms: float,
) -> tuple[np.ndarray, list[Region]]:
    chunks: list[np.ndarray] = []
    truth: list[Region] = []
    cursor = 0
    words = text.upper().split()
    for word_index, word in enumerate(words):
        if word_index > 0:
            gap = np.zeros(int(round(word_gap_ms / 1000.0 * sample_rate)), dtype=np.float32)
            chunks.append(gap)
            cursor += gap.size
        for char_index, char in enumerate(word):
            if char_index > 0:
                gap = np.zeros(int(round(char_gap_ms / 1000.0 * sample_rate)), dtype=np.float32)
                chunks.append(gap)
                cursor += gap.size
            audio = synthesize(char, wpm=wpm, sample_rate=sample_rate, rise_ms=3.0)
            chunks.append(audio)
            truth.append(Region(cursor, cursor + audio.size))
            cursor += audio.size
    return np.concatenate(chunks), truth


if __name__ == "__main__":
    main()
