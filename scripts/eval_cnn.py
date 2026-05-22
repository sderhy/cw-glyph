"""Evaluate a trained CNN checkpoint on synthetic Morse glyph variants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morse_char_recognizer.dataset import SyntheticDatasetConfig, SyntheticMorseCharacterDataset
from morse_char_recognizer.inference import load_checkpoint_model, predict_envelopes
from morse_char_recognizer.metrics import classification_report


def main() -> None:
    args = _parse_args()
    model, classes, envelope, _checkpoint_args = load_checkpoint_model(
        args.checkpoint,
        device=args.device,
    )

    if args.sweep_snr:
        for snr_db in args.sweep_snr:
            report = _evaluate(args, model, classes, envelope, (snr_db, snr_db))
            _print_report(report, title=f"SNR {snr_db:g} dB")
        return

    report = _evaluate(args, model, classes, envelope, _snr_range(args))
    _print_report(report, title="cnn")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--wpm-min", type=float, default=12.0)
    parser.add_argument("--wpm-max", type=float, default=30.0)
    parser.add_argument("--jitter-max", type=float, default=0.12)
    parser.add_argument("--snr-min", type=float, default=None)
    parser.add_argument("--snr-max", type=float, default=None)
    parser.add_argument("--sweep-snr", type=float, nargs="*", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--show-confusions", type=int, default=12)
    return parser.parse_args()


def _evaluate(args, model, classes, envelope, snr_range):
    dataset = SyntheticMorseCharacterDataset(
        SyntheticDatasetConfig(
            classes=classes,
            samples_per_class=args.samples_per_class,
            sample_rate=args.sample_rate,
            envelope=envelope,
            seed=args.seed,
            wpm_range=(args.wpm_min, args.wpm_max),
            element_jitter_range=(0.0, args.jitter_max),
            gap_jitter_range=(0.0, args.jitter_max),
            snr_db_range=snr_range,
        )
    )
    envelopes = []
    y_true = []
    for example in dataset:
        envelopes.append(example.envelope)
        y_true.append(example.label)

    y_pred, _scores = predict_envelopes(model, envelopes, device=args.device)
    return classification_report(y_true, y_pred, classes)


def _snr_range(args: argparse.Namespace) -> tuple[float, float] | None:
    if args.snr_min is None and args.snr_max is None:
        return None
    if args.snr_min is None or args.snr_max is None:
        raise ValueError("--snr-min and --snr-max must be provided together")
    return (args.snr_min, args.snr_max)


def _print_report(report, *, title: str) -> None:
    print(f"{title}: accuracy={report.accuracy:.3f} correct={report.correct}/{report.total}")
    confusions = report.top_confusions(limit=12)
    if confusions:
        print("top confusions:")
        for true_char, pred_char, count in confusions:
            print(f"  {true_char} -> {pred_char}: {count}")


if __name__ == "__main__":
    main()
