"""Grid-search decoding parameters on labelled real WAV windows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    args = _parse_args()
    best = None
    results = []
    for threshold in args.thresholds:
        for char_gap_units in args.char_gap_units:
            for min_score in args.min_scores:
                result = _run_eval(args, threshold, char_gap_units, min_score)
                results.append(
                    {
                        "threshold": threshold,
                        "char_gap_units": char_gap_units,
                        "min_score": min_score,
                        **result,
                    }
                )
                metric = _metric(result)
                metric_text = "n/a" if metric is None else f"{metric:.3f}"
                cer_text = "n/a" if result["cer"] is None else f"{result['cer']:.3f}"
                print(
                    f"threshold={threshold:g} char_gap_units={char_gap_units:g} "
                    f"min_score={min_score:g} cer={cer_text} score={metric_text} "
                    f"segments={result['segments']} rejected={result['rejected_segments']}"
                )
                if metric is not None and (best is None or metric < best["score"]):
                    best = {
                        "threshold": threshold,
                        "char_gap_units": char_gap_units,
                        "min_score": min_score,
                        "score": metric,
                        **result,
                    }
    if best is not None:
        print(
            "best: "
            f"threshold={best['threshold']:g} "
            f"char_gap_units={best['char_gap_units']:g} "
            f"min_score={best['min_score']:g} "
            f"cer={best['cer']:.3f} segments={best['segments']}"
        )
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"best": best, "results": results}, indent=2))
        print(f"wrote {out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("wav")
    parser.add_argument("--allowed-class-preset", default="copy")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.12, 0.15, 0.18, 0.22])
    parser.add_argument("--char-gap-units", type=float, nargs="+", default=[1.7, 2.0, 2.3])
    parser.add_argument("--min-scores", type=float, nargs="+", default=[0.0])
    parser.add_argument("--threshold-mode", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument(
        "--scale-mode",
        choices=("checkpoint", "stretch", "unit"),
        default="checkpoint",
    )
    parser.add_argument("--canonical-units", type=float, default=24.0)
    parser.add_argument("--word-gap-mode", choices=("fixed", "unit"), default="fixed")
    parser.add_argument("--word-gap-units", type=float, default=4.5)
    parser.add_argument("--window", type=float, default=30.0)
    parser.add_argument("--hop", type=float, default=30.0)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--match-mode", choices=("full", "best-substring"), default="full")
    parser.add_argument("--auto-center", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def _run_eval(
    args: argparse.Namespace,
    threshold: float,
    char_gap_units: float,
    min_score: float,
) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        cmd = [
            sys.executable,
            "scripts/eval_livetest.py",
            args.checkpoint,
            args.wav,
            "--allowed-class-preset",
            args.allowed_class_preset,
            "--threshold-mode",
            args.threshold_mode,
            "--threshold",
            str(threshold),
            "--char-gap-units",
            str(char_gap_units),
            "--scale-mode",
            args.scale_mode,
            "--canonical-units",
            str(args.canonical_units),
            "--word-gap-mode",
            args.word_gap_mode,
            "--word-gap-units",
            str(args.word_gap_units),
            "--window",
            str(args.window),
            "--hop",
            str(args.hop),
            "--match-mode",
            args.match_mode,
            "--min-score",
            str(min_score),
            "--json-out",
            tmp.name,
            "--device",
            args.device,
        ]
        if args.max_windows is not None:
            cmd.extend(["--max-windows", str(args.max_windows)])
        if args.auto_center:
            cmd.append("--auto-center")
        subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(Path(tmp.name).read_text())
    summary = payload["summary"]
    return {
        "windows": summary["windows"],
        "complete_files": summary["complete_files"],
        "cer": summary["cer"],
        "accuracy": summary["accuracy"],
        "substitutions": summary["substitutions"],
        "insertions": summary["insertions"],
        "deletions": summary["deletions"],
        "segments": sum(item["segments"] for item in payload["results"]),
        "rejected_segments": sum(item["rejected_segments"] for item in payload["results"]),
    }


def _metric(result: dict) -> float | None:
    return result["cer"]


if __name__ == "__main__":
    main()
