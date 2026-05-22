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
    best_ref = {"best": None}
    results = []
    for threshold in args.thresholds:
        for char_gap_units in args.char_gap_units:
            for min_score in args.min_scores:
                for min_segment_units in args.min_segment_units:
                    for max_character_units in args.max_character_units:
                        result = _run_eval(
                            args,
                            threshold,
                            char_gap_units,
                            min_score,
                            min_segment_units,
                            max_character_units,
                        )
                        _handle_result(
                            best_ref,
                            results,
                            threshold,
                            char_gap_units,
                            min_score,
                            min_segment_units,
                            max_character_units,
                            result,
                        )
    best = best_ref["best"]
    if best is not None:
        print(
            "best: "
            f"threshold={best['threshold']:g} "
            f"char_gap_units={best['char_gap_units']:g} "
            f"min_score={best['min_score']:g} "
            f"min_segment_units={best['min_segment_units']:g} "
            f"max_character_units={best['max_character_units']} "
            f"cer={best['cer']:.3f} segments={best['segments']}"
        )
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"best": best, "results": results}, indent=2))
        print(f"wrote {out}")


def _handle_result(
    best_ref: dict,
    results: list[dict],
    threshold: float,
    char_gap_units: float,
    min_score: float,
    min_segment_units: float,
    max_character_units: float | None,
    result: dict,
) -> None:
    results.append(
        {
            "threshold": threshold,
            "char_gap_units": char_gap_units,
            "min_score": min_score,
            "min_segment_units": min_segment_units,
            "max_character_units": max_character_units,
            **result,
        }
    )
    metric = _metric(result)
    metric_text = "n/a" if metric is None else f"{metric:.3f}"
    cer_text = "n/a" if result["cer"] is None else f"{result['cer']:.3f}"
    print(
        f"threshold={threshold:g} char_gap_units={char_gap_units:g} "
        f"min_score={min_score:g} min_segment_units={min_segment_units:g} "
        f"max_character_units={max_character_units} "
        f"cer={cer_text} score={metric_text} "
        f"segments={result['segments']} rejected={result['rejected_segments']}"
    )
    best = best_ref["best"]
    if metric is not None and (best is None or metric < best["score"]):
        best_ref["best"] = {
            "threshold": threshold,
            "char_gap_units": char_gap_units,
            "min_score": min_score,
            "min_segment_units": min_segment_units,
            "max_character_units": max_character_units,
            "score": metric,
            **result,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("wav")
    parser.add_argument("--allowed-class-preset", default="copy")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.12, 0.15, 0.18, 0.22])
    parser.add_argument("--char-gap-units", type=float, nargs="+", default=[1.7, 2.0, 2.3])
    parser.add_argument("--min-scores", type=float, nargs="+", default=[0.0])
    parser.add_argument("--min-segment-units", type=float, nargs="+", default=[0.0])
    parser.add_argument("--max-character-units", type=float, nargs="+", default=[0.0])
    parser.add_argument("--threshold-mode", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument(
        "--scale-mode",
        choices=("checkpoint", "stretch", "unit"),
        default="checkpoint",
    )
    parser.add_argument("--canonical-units", type=float, default=24.0)
    parser.add_argument("--word-gap-mode", choices=("fixed", "unit"), default="fixed")
    parser.add_argument("--word-gap-units", type=float, default=4.5)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--window", type=float, default=None)
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
    min_segment_units: float,
    max_character_units: float | None,
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
            "--duration",
            str(args.duration),
            "--match-mode",
            args.match_mode,
            "--min-score",
            str(min_score),
            "--min-segment-units",
            str(min_segment_units),
            "--json-out",
            tmp.name,
            "--device",
            args.device,
        ]
        if max_character_units is not None and max_character_units > 0:
            cmd.extend(["--max-character-units", str(max_character_units)])
        if args.window is not None:
            cmd.extend(["--window", str(args.window), "--hop", str(args.hop)])
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
