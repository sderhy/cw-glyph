"""Train the CRNN-CTC Morse recogniser on synthetic sequences."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morse_char_recognizer.ctc.dataset import (
    CachedSyntheticCtcDataset,
    CtcDatasetConfig,
    SyntheticCtcDataset,
    collate_ctc,
)
from morse_char_recognizer.ctc.decode import greedy_decode
from morse_char_recognizer.ctc.frames import FrameConfig
from morse_char_recognizer.ctc.metrics import aggregate_cer
from morse_char_recognizer.ctc.model import CrnnCtcConfig, CrnnCtcRecognizer
from morse_char_recognizer.ctc.vocab import CtcVocab


def main() -> None:
    args = _parse_args()
    device = _device(args.device)
    torch.manual_seed(args.seed)

    vocab = CtcVocab()
    frame_config = FrameConfig(frame_rate_hz=args.frame_rate_hz)
    train_config = _dataset_config(args, frame_config, vocab, seed=args.seed, count=args.train_size)
    val_config = _dataset_config(
        args,
        frame_config,
        vocab,
        seed=args.seed + 1,
        count=args.val_size,
        clean=True,
    )

    dataset_cls = CachedSyntheticCtcDataset if args.cache else SyntheticCtcDataset
    train_ds = dataset_cls(train_config)
    val_ds = dataset_cls(val_config)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_ctc,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_ctc,
        num_workers=args.num_workers,
    )

    model_config = CrnnCtcConfig(
        num_classes=vocab.num_classes,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        dropout=args.dropout,
    )
    model = CrnnCtcRecognizer(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ctc_loss = nn.CTCLoss(blank=vocab.blank_index, zero_infinity=True)

    best_cer: float | None = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, ctc_loss, device)
        val_cer, val_examples = _evaluate(model, val_loader, vocab, device)
        scheduler.step()
        if best_cer is None or val_cer < best_cer:
            best_cer = val_cer
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} val_cer={val_cer:.4f} "
            f"val_acc={1.0 - val_cer:.4f}"
        )
        for ref, hyp in val_examples[:3]:
            print(f"  ref={ref!r:>32}  hyp={hyp!r}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": best_state or model.state_dict(),
                "model_config": asdict(model_config),
                "vocab_chars": vocab.chars,
                "frame_config": asdict(frame_config),
                "args": vars(args),
                "best_cer": best_cer,
            },
            out,
        )
        print(f"wrote {out} (best CER {best_cer:.4f})")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=400)
    parser.add_argument("--min-chars", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=10)
    parser.add_argument("--word-prob", type=float, default=0.25)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--frame-rate-hz", type=float, default=200.0)
    parser.add_argument("--wpm-min", type=float, default=14.0)
    parser.add_argument("--wpm-max", type=float, default=28.0)
    parser.add_argument("--snr-min", type=float, default=None)
    parser.add_argument("--snr-max", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rnn-hidden", type=int, default=128)
    parser.add_argument("--rnn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", default="outputs/ctc_recognizer.pt")
    return parser.parse_args()


def _dataset_config(
    args: argparse.Namespace,
    frame_config: FrameConfig,
    vocab: CtcVocab,
    *,
    seed: int,
    count: int,
    clean: bool = False,
) -> CtcDatasetConfig:
    snr_range: tuple[float, float] | None = None
    if not clean and args.snr_min is not None and args.snr_max is not None:
        snr_range = (args.snr_min, args.snr_max)
    return CtcDatasetConfig(
        vocab=vocab,
        frames=frame_config,
        num_examples=count,
        sample_rate=args.sample_rate,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        word_probability=args.word_prob,
        seed=seed,
        wpm_range=(args.wpm_min, args.wpm_max),
        snr_db_range=snr_range,
    )


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ctc_loss: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        input_lengths = model.output_lengths(batch["input_lengths"]).to(device)
        target_lengths = batch["target_lengths"].to(device)
        optimizer.zero_grad(set_to_none=True)
        log_probs = model(inputs)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += float(loss.detach().cpu()) * inputs.size(0)
        count += inputs.size(0)
    return total / max(1, count)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    vocab: CtcVocab,
    device: torch.device,
) -> tuple[float, list[tuple[str, str]]]:
    model.eval()
    pairs: list[tuple[str, str]] = []
    for batch in loader:
        inputs = batch["inputs"].to(device)
        input_lengths = model.output_lengths(batch["input_lengths"]).to(device)
        log_probs = model(inputs)
        decoded = greedy_decode(log_probs, input_lengths, vocab)
        for reference, hypothesis in zip(batch["texts"], decoded):
            pairs.append((reference, hypothesis))
    return aggregate_cer(pairs).cer, pairs


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
