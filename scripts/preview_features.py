"""Preview synthetic Morse-character features as glyph-like plots.

This script generates a small visual inspection sheet, not a training dataset.
It is meant to answer: "does the model input preserve the pattern that is easy
to see when zooming Morse in an audio editor?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morse_char_recognizer.classes import CLASS_PRESETS, parse_class_spec
from morse_char_recognizer.dataset import SyntheticDatasetConfig, SyntheticMorseCharacterDataset
from morse_char_recognizer.features import EnvelopeConfig
from morse_char_recognizer.morse_table import MORSE_TABLE


def main() -> None:
    args = _parse_args()
    chars = parse_class_spec(args.classes, preset=args.class_preset)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    config = SyntheticDatasetConfig(
        classes=chars,
        samples_per_class=args.variants,
        sample_rate=args.sample_rate,
        envelope=EnvelopeConfig(length=args.length),
        seed=args.seed,
        snr_db_range=(args.snr_db, args.snr_db) if args.snr_db is not None else None,
    )
    dataset = SyntheticMorseCharacterDataset(config)

    examples = [dataset[i] for i in range(min(len(dataset), args.max_examples))]
    _plot_preview(examples, out, args.spectrogram)
    print(f"wrote {out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/feature_preview.png")
    parser.add_argument("--class-preset", choices=tuple(CLASS_PRESETS), default="basic")
    parser.add_argument("--classes", "--chars", default=None)
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=36)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--spectrogram", action="store_true")
    return parser.parse_args()


def _plot_preview(examples, out: Path, include_spectrogram: bool) -> None:
    rows = len(examples)
    height_per_row = 1.25 if include_spectrogram else 0.85
    fig, axes = plt.subplots(
        rows,
        2 if include_spectrogram else 1,
        figsize=(12, max(4, rows * height_per_row)),
        squeeze=False,
        constrained_layout=True,
    )

    for row, example in enumerate(examples):
        _plot_envelope(axes[row, 0], example)
        if include_spectrogram:
            _plot_spectrogram(axes[row, 1], example)

    fig.suptitle("Synthetic isolated Morse character previews", fontsize=14)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_envelope(ax, example) -> None:
    code = MORSE_TABLE.get(example.char, "")
    ax.plot(example.envelope, color="#1f77b4", linewidth=1.5)
    ax.fill_between(np.arange(example.envelope.size), example.envelope, color="#1f77b4", alpha=0.18)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(0, example.envelope.size - 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel(f"{example.char}  {code}", rotation=0, ha="right", va="center", labelpad=28)
    ax.grid(axis="x", color="#dddddd", linewidth=0.4)


def _plot_spectrogram(ax, example) -> None:
    frequencies, times, power = spectrogram(
        example.audio,
        fs=example.sample_rate,
        nperseg=128,
        noverlap=96,
        scaling="spectrum",
        mode="magnitude",
    )
    mask = frequencies <= 1200
    image = np.log1p(power[mask])
    ax.imshow(
        image,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], frequencies[mask][0], frequencies[mask][-1]],
        cmap="magma",
    )
    ax.set_yticks([0, 600, 1200])
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=7)


if __name__ == "__main__":
    main()
