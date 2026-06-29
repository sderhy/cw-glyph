"""Train a 1D CNN on synthetic isolated Morse-character glyphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morse_char_recognizer.classes import CLASS_PRESETS, parse_class_spec
from morse_char_recognizer.dataset import SyntheticDatasetConfig
from morse_char_recognizer.features import EnvelopeConfig
from morse_char_recognizer.metrics import classification_report
from morse_char_recognizer.model import MorseGlyphCnn1d, SmallCnn1d
from morse_char_recognizer.torch_data import (
    CachedTorchMorseCharacterDataset,
    TorchMorseCharacterDataset,
)


def main() -> None:
    args = _parse_args()
    device = _device(args.device)
    classes = parse_class_spec(args.classes, preset=args.class_preset)
    envelope = EnvelopeConfig(
        length=args.length,
        scale_mode=args.scale_mode,
        canonical_units=args.canonical_units,
        binarize_threshold=args.envelope_binarize_threshold,
    )
    torch.manual_seed(args.seed)

    dataset_cls = CachedTorchMorseCharacterDataset if args.cache else TorchMorseCharacterDataset
    train_ds = dataset_cls(
        _dataset_config(args, classes, envelope, seed=args.seed, samples=args.samples_per_class)
    )
    val_ds = dataset_cls(
        _dataset_config(
            args,
            classes,
            envelope,
            seed=args.seed + 1,
            samples=args.val_samples_per_class,
        )
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = _build_model(args, len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_report = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, loss_fn, device)
        report = _evaluate(model, val_loader, classes, device)
        scheduler.step()
        if best_report is None or report.accuracy > best_report.accuracy:
            best_report = report
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_accuracy={report.accuracy:.3f} correct={report.correct}/{report.total}"
        )
        if args.show_confusions:
            _print_confusions(report, limit=args.show_confusions)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": best_state or model.state_dict(),
                "classes": classes,
                "envelope": envelope,
                "args": vars(args),
                "best_accuracy": None if best_report is None else best_report.accuracy,
            },
            out,
        )
        print(f"wrote {out}")
    if best_report is not None:
        print(
            f"best_val_accuracy={best_report.accuracy:.3f} "
            f"correct={best_report.correct}/{best_report.total}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-preset", choices=tuple(CLASS_PRESETS), default="basic")
    parser.add_argument("--classes", default=None)
    parser.add_argument("--samples-per-class", type=int, default=250)
    parser.add_argument("--val-samples-per-class", type=int, default=60)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--scale-mode", choices=("stretch", "unit"), default="stretch")
    parser.add_argument("--canonical-units", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wpm-min", type=float, default=12.0)
    parser.add_argument("--wpm-max", type=float, default=30.0)
    parser.add_argument("--jitter-max", type=float, default=0.12)
    parser.add_argument("--dash-dot-ratio-min", type=float, default=2.7)
    parser.add_argument("--dash-dot-ratio-max", type=float, default=3.4)
    parser.add_argument("--gap-inflation-min", type=float, default=0.9)
    parser.add_argument("--gap-inflation-max", type=float, default=1.25)
    parser.add_argument("--rise-ms-min", type=float, default=2.0)
    parser.add_argument("--rise-ms-max", type=float, default=8.0)
    parser.add_argument("--stroke-amplitude-jitter-max", type=float, default=0.0)
    parser.add_argument("--envelope-binarize-threshold", type=float, default=None)
    parser.add_argument("--snr-min", type=float, default=None)
    parser.add_argument("--snr-max", type=float, default=None)
    parser.add_argument("--qrn-rate-min", type=float, default=0.0)
    parser.add_argument("--qrn-rate-max", type=float, default=0.0)
    parser.add_argument("--qrn-amplitude-min-db", type=float, default=-12.0)
    parser.add_argument("--qrn-amplitude-max-db", type=float, default=-6.0)
    parser.add_argument("--qsb-rate-min-hz", type=float, default=0.0)
    parser.add_argument("--qsb-rate-max-hz", type=float, default=0.0)
    parser.add_argument("--qsb-depth-min-db", type=float, default=0.0)
    parser.add_argument("--qsb-depth-max-db", type=float, default=0.0)
    parser.add_argument("--carrier-drift-min-hz-per-s", type=float, default=0.0)
    parser.add_argument("--carrier-drift-max-hz-per-s", type=float, default=0.0)
    parser.add_argument("--rx-filter-bw-min", type=float, default=None)
    parser.add_argument("--rx-filter-bw-max", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", choices=("small", "glyph"), default="glyph")
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-confusions", type=int, default=0)
    parser.add_argument("--out", default="outputs/glyph_cnn.pt")
    return parser.parse_args()


def _build_model(args: argparse.Namespace, num_classes: int) -> nn.Module:
    if args.model == "small":
        return SmallCnn1d(num_classes=num_classes)
    return MorseGlyphCnn1d(num_classes=num_classes, dropout=args.dropout)


def _dataset_config(
    args: argparse.Namespace,
    classes: tuple[str, ...],
    envelope: EnvelopeConfig,
    *,
    seed: int,
    samples: int,
) -> SyntheticDatasetConfig:
    return SyntheticDatasetConfig(
        classes=classes,
        samples_per_class=samples,
        sample_rate=args.sample_rate,
        envelope=envelope,
        seed=seed,
        wpm_range=(args.wpm_min, args.wpm_max),
        rise_ms_range=(args.rise_ms_min, args.rise_ms_max),
        element_jitter_range=(0.0, args.jitter_max),
        gap_jitter_range=(0.0, args.jitter_max),
        dash_dot_ratio_range=(args.dash_dot_ratio_min, args.dash_dot_ratio_max),
        gap_inflation_range=(args.gap_inflation_min, args.gap_inflation_max),
        stroke_amplitude_jitter_range=(0.0, args.stroke_amplitude_jitter_max),
        snr_db_range=_snr_range(args),
        qrn_rate_per_sec_range=(args.qrn_rate_min, args.qrn_rate_max),
        qrn_amplitude_db_range=(args.qrn_amplitude_min_db, args.qrn_amplitude_max_db),
        qsb_rate_hz_range=(args.qsb_rate_min_hz, args.qsb_rate_max_hz),
        qsb_depth_db_range=(args.qsb_depth_min_db, args.qsb_depth_max_db),
        carrier_drift_hz_per_s_range=(
            args.carrier_drift_min_hz_per_s,
            args.carrier_drift_max_hz_per_s,
        ),
        rx_filter_bw_range=_rx_filter_bw_range(args),
    )


def _snr_range(args: argparse.Namespace) -> tuple[float, float] | None:
    if args.snr_min is None and args.snr_max is None:
        return None
    if args.snr_min is None or args.snr_max is None:
        raise ValueError("--snr-min and --snr-max must be provided together")
    return (args.snr_min, args.snr_max)


def _rx_filter_bw_range(args: argparse.Namespace) -> tuple[float, float] | None:
    if args.rx_filter_bw_min is None and args.rx_filter_bw_max is None:
        return None
    if args.rx_filter_bw_min is None or args.rx_filter_bw_max is None:
        raise ValueError("--rx-filter-bw-min and --rx-filter-bw-max must be provided together")
    return (args.rx_filter_bw_min, args.rx_filter_bw_max)


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * x.size(0)
        total += x.size(0)
    return total_loss / max(1, total)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    classes: tuple[str, ...],
    device: torch.device,
):
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for x, y in loader:
        logits = model(x.to(device))
        pred = torch.argmax(logits, dim=1).cpu()
        y_true.extend(y.tolist())
        y_pred.extend(pred.tolist())
    return classification_report(y_true, y_pred, classes)


def _print_confusions(report, *, limit: int) -> None:
    confusions = report.top_confusions(limit=limit)
    if not confusions:
        print("  no confusions")
        return
    text = ", ".join(f"{true}->{pred}:{count}" for true, pred, count in confusions)
    print(f"  confusions: {text}")


if __name__ == "__main__":
    main()
