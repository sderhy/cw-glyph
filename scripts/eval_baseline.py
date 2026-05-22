"""Evaluate the template baseline on synthetic isolated-character variants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morse_char_recognizer.baseline import TemplateClassifier, TemplateConfig
from morse_char_recognizer.classes import CLASS_PRESETS, parse_class_spec
from morse_char_recognizer.dataset import SyntheticDatasetConfig, SyntheticMorseCharacterDataset
from morse_char_recognizer.features import EnvelopeConfig
from morse_char_recognizer.metrics import ClassificationReport, classification_report


def main() -> None:
    args = _parse_args()
    classes = parse_class_spec(args.classes, preset=args.class_preset)
    envelope = EnvelopeConfig(length=args.length)

    classifier = TemplateClassifier(
        TemplateConfig(
            classes=classes,
            sample_rate=args.sample_rate,
            envelope=envelope,
            wpm=args.template_wpm,
            freq=args.template_freq,
            rise_ms=args.template_rise_ms,
        )
    )

    if args.sweep_snr:
        for snr_db in args.sweep_snr:
            report = _evaluate(args, classes, envelope, classifier, (snr_db, snr_db))
            _print_report(report, title=f"SNR {snr_db:g} dB")
        return

    report = _evaluate(args, classes, envelope, classifier, _snr_range(args))
    _print_report(report, title="baseline")
    if args.confusion_out:
        out = Path(args.confusion_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        _plot_confusion(report, out)
        print(f"wrote {out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-preset", choices=tuple(CLASS_PRESETS), default="basic")
    parser.add_argument("--classes", default=None)
    parser.add_argument("--samples-per-class", type=int, default=25)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wpm-min", type=float, default=12.0)
    parser.add_argument("--wpm-max", type=float, default=30.0)
    parser.add_argument("--jitter-max", type=float, default=0.12)
    parser.add_argument("--snr-min", type=float, default=None)
    parser.add_argument("--snr-max", type=float, default=None)
    parser.add_argument("--sweep-snr", type=float, nargs="*", default=None)
    parser.add_argument("--template-wpm", type=float, default=20.0)
    parser.add_argument("--template-freq", type=float, default=600.0)
    parser.add_argument("--template-rise-ms", type=float, default=3.0)
    parser.add_argument("--confusion-out", default="outputs/baseline_confusion.png")
    return parser.parse_args()


def _evaluate(
    args: argparse.Namespace,
    classes: tuple[str, ...],
    envelope: EnvelopeConfig,
    classifier: TemplateClassifier,
    snr_range: tuple[float, float] | None,
) -> ClassificationReport:
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

    y_true: list[int] = []
    y_pred: list[int] = []
    for example in dataset:
        prediction = classifier.predict_one(example.envelope)
        y_true.append(example.label)
        y_pred.append(prediction.label)
    return classification_report(y_true, y_pred, classes)


def _snr_range(args: argparse.Namespace) -> tuple[float, float] | None:
    if args.snr_min is None and args.snr_max is None:
        return None
    if args.snr_min is None or args.snr_max is None:
        raise ValueError("--snr-min and --snr-max must be provided together")
    return (args.snr_min, args.snr_max)


def _print_report(report: ClassificationReport, *, title: str) -> None:
    print(f"{title}: accuracy={report.accuracy:.3f} correct={report.correct}/{report.total}")
    confusions = report.top_confusions(limit=12)
    if confusions:
        print("top confusions:")
        for true_char, pred_char, count in confusions:
            print(f"  {true_char} -> {pred_char}: {count}")

    weak = sorted(report.per_class_accuracy.items(), key=lambda item: item[1])[:8]
    print("weakest classes:")
    for char, accuracy in weak:
        print(f"  {char}: {accuracy:.3f}")


def _plot_confusion(report: ClassificationReport, out: Path) -> None:
    confusion = report.confusion.astype(np.float32)
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_sums, out=np.zeros_like(confusion), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    image = ax.imshow(normalized, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(report.classes)), labels=report.classes, fontsize=7)
    ax.set_yticks(np.arange(len(report.classes)), labels=report.classes, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Template baseline confusion matrix, accuracy={report.accuracy:.3f}")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
