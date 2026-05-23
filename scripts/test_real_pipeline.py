"""Run the current best real-audio CW-Glyph pipeline with one short command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    args = _parse_args()
    repo = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "scripts/eval_livetest.py",
        args.checkpoint,
        args.wav,
        "--auto-center",
        "--allowed-class-preset",
        args.allowed_class_preset,
        "--scale-mode",
        "unit",
        "--canonical-units",
        "32",
        "--threshold",
        str(args.threshold),
        "--char-gap-units",
        str(args.char_gap_units),
        "--min-segment-units",
        str(args.min_segment_units),
        "--max-character-units",
        str(args.max_character_units),
        "--word-gap-mode",
        "unit",
        "--duration",
        "0",
        "--top-k",
        str(args.top_k),
        "--json-out",
        args.json_out,
    ]
    if args.label:
        cmd.extend(["--label", args.label])

    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/glyph52_real_unit32_cnn_snr5_25_v2.pt",
    )
    parser.add_argument("--wav", default="livetests/g6pz/G1.wav")
    parser.add_argument("--label", default=None)
    parser.add_argument("--allowed-class-preset", default="real")
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--char-gap-units", type=float, default=1.6)
    parser.add_argument("--min-segment-units", type=float, default=1.4)
    parser.add_argument("--max-character-units", type=float, default=19.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-out", default="outputs/real_pipeline_test.json")
    return parser.parse_args()


if __name__ == "__main__":
    main()
