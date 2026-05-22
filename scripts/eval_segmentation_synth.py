"""Evaluate character segmentation on synthetic continuous Morse audio."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morse_char_recognizer.segment import Region, SegmentConfig, segment_characters
from morse_char_recognizer.segmentation_metrics import evaluate_segmentation
from morse_synth.channel import ChannelConfig, apply_channel
from morse_synth.core import synthesize


def main() -> None:
    args = _parse_args()
    results = [_run_case(args, wpm, seed) for wpm in args.wpm for seed in args.seeds]
    report = _summarize(results)
    print(
        f"cases={report['cases']} truth={report['true_count']} "
        f"predicted={report['predicted_count']} matched={report['matched']} "
        f"missed={report['missed']} split={report['split']} merge={report['merge']} "
        f"false_positive={report['false_positive']} "
        f"boundary_ms={report['mean_boundary_error_ms']:.1f} "
        f"precision={report['precision']:.3f} recall={report['recall']:.3f}"
    )
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "metadata": _metadata(),
                    "args": _json_safe(vars(args)),
                    "summary": report,
                    "results": results,
                },
                indent=2,
                allow_nan=False,
            )
        )
        print(f"wrote {out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="CQ TEST 599")
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--wpm", type=float, nargs="+", default=[20.0])
    parser.add_argument("--char-gap-ms", type=float, default=180.0)
    parser.add_argument("--word-gap-ms", type=float, default=420.0)
    parser.add_argument("--threshold-mode", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--threshold", type=float, default=0.18)
    parser.add_argument("--adaptive-window-ms", type=float, default=650.0)
    parser.add_argument("--adaptive-floor-percentile", type=float, default=20.0)
    parser.add_argument("--adaptive-peak-percentile", type=float, default=90.0)
    parser.add_argument("--adaptive-min-threshold", type=float, default=0.03)
    parser.add_argument("--char-gap-units", type=float, default=2.3)
    parser.add_argument("--max-character-units", type=float, default=None)
    parser.add_argument("--pad-ms", type=float, default=20.0)
    parser.add_argument("--snr-db", type=float, default=float("inf"))
    parser.add_argument("--qsb-rate-hz", type=float, default=0.0)
    parser.add_argument("--qsb-depth-db", type=float, default=0.0)
    parser.add_argument("--qrn-rate-per-sec", type=float, default=0.0)
    parser.add_argument("--carrier-drift-hz-per-s", type=float, default=0.0)
    parser.add_argument("--rx-filter-bw", type=float, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def _run_case(args: argparse.Namespace, wpm: float, seed: int) -> dict:
    audio, truth = _synthetic_sequence(
        args.text,
        sample_rate=args.sample_rate,
        wpm=wpm,
        char_gap_ms=args.char_gap_ms,
        word_gap_ms=args.word_gap_ms,
    )
    audio = apply_channel(
        audio,
        args.sample_rate,
        ChannelConfig(
            snr_db=args.snr_db,
            qrn_rate_per_sec=args.qrn_rate_per_sec,
            qsb_rate_hz=args.qsb_rate_hz,
            qsb_depth_db=args.qsb_depth_db,
            carrier_drift_hz_per_s=args.carrier_drift_hz_per_s,
            rx_filter_bw=args.rx_filter_bw,
            seed=seed,
        ),
    )
    config = SegmentConfig(
        threshold_mode=args.threshold_mode,
        threshold=args.threshold,
        adaptive_window_ms=args.adaptive_window_ms,
        adaptive_floor_percentile=args.adaptive_floor_percentile,
        adaptive_peak_percentile=args.adaptive_peak_percentile,
        adaptive_min_threshold=args.adaptive_min_threshold,
        char_gap_units=args.char_gap_units,
        max_character_units=args.max_character_units,
        pad_ms=args.pad_ms,
    )
    predicted = segment_characters(audio, args.sample_rate, config)
    report = evaluate_segmentation(truth, predicted, args.sample_rate)
    return {
        "wpm": wpm,
        "seed": seed,
        "true_count": report.true_count,
        "predicted_count": report.predicted_count,
        "matched": report.matched,
        "missed": report.missed,
        "split": report.split,
        "merge": report.merge,
        "false_positive": report.false_positive,
        "mean_boundary_error_ms": report.mean_boundary_error_ms,
        "precision": report.precision,
        "recall": report.recall,
    }


def _summarize(results: list[dict]) -> dict:
    true_count = sum(item["true_count"] for item in results)
    predicted_count = sum(item["predicted_count"] for item in results)
    matched = sum(item["matched"] for item in results)
    boundary_errors = [
        item["mean_boundary_error_ms"] for item in results if item["matched"] > 0
    ]
    return {
        "cases": len(results),
        "true_count": true_count,
        "predicted_count": predicted_count,
        "matched": matched,
        "missed": sum(item["missed"] for item in results),
        "split": sum(item["split"] for item in results),
        "merge": sum(item["merge"] for item in results),
        "false_positive": sum(item["false_positive"] for item in results),
        "mean_boundary_error_ms": float(np.mean(boundary_errors)) if boundary_errors else 0.0,
        "precision": 0.0 if predicted_count == 0 else matched / predicted_count,
        "recall": 0.0 if true_count == 0 else matched / true_count,
    }


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _metadata() -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "command": sys.argv,
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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
