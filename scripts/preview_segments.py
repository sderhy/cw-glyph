"""Preview envelope-based segmentation on continuous Morse audio."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morse_char_recognizer.features import amplitude_envelope
from morse_char_recognizer.segment import SegmentConfig, detect_active_regions, segment_characters
from morse_synth.core import synthesize


def main() -> None:
    args = _parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    audio = synthesize(
        args.text,
        wpm=args.wpm,
        freq=args.freq,
        sample_rate=args.sample_rate,
        rise_ms=args.rise_ms,
        tail_ms=args.tail_ms,
    )
    config = SegmentConfig(threshold=args.threshold)
    envelope = amplitude_envelope(audio, args.sample_rate)
    active_regions = detect_active_regions(audio, args.sample_rate, config)
    char_segments = segment_characters(audio, args.sample_rate, config)

    _plot(out, envelope, args.sample_rate, active_regions, char_segments, args.text)
    print(f"wrote {out}")
    print(f"detected keydowns={len(active_regions)} character_segments={len(char_segments)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="CQ TEST")
    parser.add_argument("--out", default="outputs/segment_preview.png")
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--wpm", type=float, default=20.0)
    parser.add_argument("--freq", type=float, default=600.0)
    parser.add_argument("--rise-ms", type=float, default=3.0)
    parser.add_argument("--tail-ms", type=float, default=80.0)
    parser.add_argument("--threshold", type=float, default=0.18)
    return parser.parse_args()


def _plot(out: Path, envelope, sample_rate: int, active_regions, char_segments, text: str) -> None:
    t = np.arange(envelope.size) / sample_rate
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)

    ax.plot(t, envelope, color="#1f77b4", linewidth=1.1)
    ax.fill_between(t, envelope, color="#1f77b4", alpha=0.16)

    for region in active_regions:
        ax.axvspan(
            region.start / sample_rate,
            region.end / sample_rate,
            color="#2ca02c",
            alpha=0.16,
        )
    for index, segment in enumerate(char_segments, start=1):
        start_s = segment.start / sample_rate
        end_s = segment.end / sample_rate
        ax.axvline(start_s, color="#d62728", linewidth=0.8)
        ax.axvline(end_s, color="#d62728", linewidth=0.8)
        ax.text(
            (start_s + end_s) / 2,
            1.04,
            str(index),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#d62728",
        )

    ax.set_title(f"Continuous Morse segmentation preview: {text!r}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("envelope")
    ax.set_ylim(-0.05, 1.12)
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
