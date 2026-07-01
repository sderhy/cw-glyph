"""Train a segmentation-aware Morse glyph CNN with a <JUNK> merge-rejection class.

Unlike ``scripts/train_cnn.py`` (isolated, perfectly-cut glyphs), this trains on
the spans the beam search actually proposes: positive spans are exact single
characters, and boundary-crossing merges (e.g. ``T``+``N`` -> the ``G``-shaped
span) are labeled ``<JUNK>``. The resulting model puts merge spans' probability
mass on ``<JUNK>``, so ``logP(real char)`` collapses for a bad merge and the CNN
finally *helps* the beam instead of biasing it toward merging.

The checkpoint is drop-in for ``morse_char_recognizer.cnn_beam`` /
``scripts/decode_cnn_beam.py`` — the extra ``<JUNK>`` class is never proposed by
the decoder, it only absorbs probability.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morse_char_recognizer.classes import CLASS_PRESETS, parse_class_spec
from morse_char_recognizer.features import EnvelopeConfig
from morse_char_recognizer.model import MorseGlyphCnn1d
from morse_char_recognizer.span_dataset import JUNK_TOKEN, SpanDatasetConfig, build_span_dataset


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dataset_config(args: argparse.Namespace, classes, envelope, *, seed, num_messages):
    return SpanDatasetConfig(
        classes=classes,
        sample_rate=args.sample_rate,
        envelope=envelope,
        num_messages=num_messages,
        tokens_per_message_range=(args.tokens_min, args.tokens_max),
        junk_per_positive=args.junk_per_positive,
        pad_ms=args.pad_ms,
        seed=seed,
        wpm_range=(args.wpm_min, args.wpm_max),
        element_jitter_range=(0.0, args.jitter_max),
        gap_jitter_range=(0.0, args.jitter_max),
        dash_dot_ratio_range=(args.dash_dot_ratio_min, args.dash_dot_ratio_max),
        gap_inflation_range=(args.gap_inflation_min, args.gap_inflation_max),
        snr_db_range=(args.snr_min, args.snr_max) if args.snr_min is not None else None,
        qrn_rate_per_sec_range=(args.qrn_rate_min, args.qrn_rate_max),
    )


def _loader(config: SpanDatasetConfig, batch_size: int, *, shuffle: bool):
    x, y, classes = build_span_dataset(config)
    ds = TensorDataset(torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle), classes, y


@torch.no_grad()
def _evaluate(model, loader, junk_index, device):
    model.eval()
    correct = total = 0
    junk_total = junk_hit = 0
    junk_leak = 0  # true junk predicted as a real class (the harmful error)
    real_total = real_correct = 0
    for x, y in loader:
        pred = torch.argmax(model(x.to(device)), dim=1).cpu()
        correct += int((pred == y).sum())
        total += y.numel()
        is_junk = y == junk_index
        junk_total += int(is_junk.sum())
        junk_hit += int((pred[is_junk] == junk_index).sum())
        junk_leak += int((pred[is_junk] != junk_index).sum())
        real = ~is_junk
        real_total += int(real.sum())
        real_correct += int((pred[real] == y[real]).sum())
    return {
        "accuracy": correct / max(1, total),
        "real_accuracy": real_correct / max(1, real_total),
        "junk_recall": junk_hit / max(1, junk_total),
        "junk_leak_rate": junk_leak / max(1, junk_total),
    }


def main() -> None:
    args = _parse_args()
    device = _device(args.device)
    classes = parse_class_spec(args.classes, preset=args.class_preset)
    envelope = EnvelopeConfig(
        length=args.length, scale_mode="unit", canonical_units=args.canonical_units
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, full_classes, _ = _loader(
        _dataset_config(args, classes, envelope, seed=args.seed, num_messages=args.num_messages),
        args.batch_size,
        shuffle=True,
    )
    val_loader, _c, _y = _loader(
        _dataset_config(
            args, classes, envelope, seed=args.seed + 1, num_messages=args.val_messages
        ),
        args.batch_size,
        shuffle=False,
    )
    junk_index = full_classes.index(JUNK_TOKEN)
    print(f"classes={len(full_classes)} (incl {JUNK_TOKEN}) train_batches={len(train_loader)}")

    model = MorseGlyphCnn1d(num_classes=len(full_classes), dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_score = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * x.size(0)
            seen += x.size(0)
        scheduler.step()
        m = _evaluate(model, val_loader, junk_index, device)
        # select on real accuracy minus junk leakage: we want real glyphs right
        # AND merges rejected.
        score = m["real_accuracy"] - m["junk_leak_rate"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch={epoch:02d} loss={total_loss / max(1, seen):.4f} "
            f"acc={m['accuracy']:.3f} real_acc={m['real_accuracy']:.3f} "
            f"junk_recall={m['junk_recall']:.3f} junk_leak={m['junk_leak_rate']:.3f}"
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": best_state or model.state_dict(),
                "classes": full_classes,
                "envelope": envelope,
                "args": {**vars(args), "model": "glyph"},
            },
            out,
        )
        print(f"wrote {out}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--class-preset", choices=tuple(CLASS_PRESETS), default="real")
    p.add_argument("--classes", default=None)
    p.add_argument("--num-messages", type=int, default=4000)
    p.add_argument("--val-messages", type=int, default=600)
    p.add_argument("--tokens-min", type=int, default=2)
    p.add_argument("--tokens-max", type=int, default=8)
    p.add_argument("--junk-per-positive", type=float, default=1.5)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--canonical-units", type=float, default=32.0)
    p.add_argument("--pad-ms", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wpm-min", type=float, default=12.0)
    p.add_argument("--wpm-max", type=float, default=30.0)
    p.add_argument("--jitter-max", type=float, default=0.12)
    p.add_argument("--dash-dot-ratio-min", type=float, default=2.7)
    p.add_argument("--dash-dot-ratio-max", type=float, default=3.4)
    p.add_argument("--gap-inflation-min", type=float, default=0.9)
    p.add_argument("--gap-inflation-max", type=float, default=1.25)
    p.add_argument("--snr-min", type=float, default=None)
    p.add_argument("--snr-max", type=float, default=None)
    p.add_argument("--qrn-rate-min", type=float, default=0.0)
    p.add_argument("--qrn-rate-max", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.08)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="outputs/glyph_cnn_spans.pt")
    return p.parse_args()


if __name__ == "__main__":
    main()
