"""Grid-search decoding parameters on labelled real WAV windows."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    args = _parse_args()
    best = None
    for threshold in args.thresholds:
        for char_gap_units in args.char_gap_units:
            result = _run_eval(args, threshold, char_gap_units)
            print(
                f"threshold={threshold:g} char_gap_units={char_gap_units:g} "
                f"weighted_accuracy={result['weighted_accuracy']:.3f}"
            )
            if best is None or result["weighted_accuracy"] > best["weighted_accuracy"]:
                best = {
                    "threshold": threshold,
                    "char_gap_units": char_gap_units,
                    **result,
                }
    if best is not None:
        print(
            "best: "
            f"threshold={best['threshold']:g} "
            f"char_gap_units={best['char_gap_units']:g} "
            f"weighted_accuracy={best['weighted_accuracy']:.3f}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("wav")
    parser.add_argument("--allowed-class-preset", default="copy")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.12, 0.15, 0.18, 0.22])
    parser.add_argument("--char-gap-units", type=float, nargs="+", default=[1.7, 2.0, 2.3])
    parser.add_argument("--window", type=float, default=30.0)
    parser.add_argument("--hop", type=float, default=30.0)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--auto-center", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _run_eval(args: argparse.Namespace, threshold: float, char_gap_units: float) -> dict:
    cmd = [
        sys.executable,
        "scripts/eval_livetest.py",
        args.checkpoint,
        args.wav,
        "--allowed-class-preset",
        args.allowed_class_preset,
        "--threshold",
        str(threshold),
        "--char-gap-units",
        str(char_gap_units),
        "--window",
        str(args.window),
        "--hop",
        str(args.hop),
        "--max-windows",
        str(args.max_windows),
        "--device",
        args.device,
    ]
    if args.auto_center:
        cmd.append("--auto-center")
    completed = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )
    return _parse_summary(completed.stdout)


def _parse_summary(output: str) -> dict:
    for line in reversed(output.splitlines()):
        if not line.startswith("summary:"):
            continue
        parts = dict(part.split("=", 1) for part in line.removeprefix("summary: ").split())
        return {
            "windows": int(parts["windows"]),
            "avg_accuracy": float(parts["avg_accuracy"]),
            "weighted_accuracy": float(parts["weighted_accuracy"]),
        }
    raise ValueError(f"eval_livetest summary not found in output:\n{output}")


if __name__ == "__main__":
    main()
