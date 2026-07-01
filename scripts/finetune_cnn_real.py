#!/usr/bin/env python3
"""Fine-tune the span glyph CNN on REAL harvested spans to close the domain gap.

Initialises from a synthetic-trained span checkpoint (which already has the
<JUNK> class) and continues training on the real spans harvested by
``scripts/harvest_real_spans.py``, optionally mixed with fresh synthetic spans
to keep coverage of classes that are rare or absent in the real audio. The
validation set is a held-out slice of the REAL spans, so its metrics finally
measure the target domain rather than the synthetic distribution.
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

from morse_char_recognizer.classes import REAL_CLASSES
from morse_char_recognizer.inference import load_checkpoint_model
from morse_char_recognizer.span_dataset import JUNK_TOKEN, SpanDatasetConfig, build_span_dataset


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _evaluate(model, loader, junk_index, device):
    model.eval()
    correct = total = real_total = real_correct = junk_total = junk_hit = junk_leak = 0
    for x, y in loader:
        pred = torch.argmax(model(x.to(device)), dim=1).cpu()
        correct += int((pred == y).sum()); total += y.numel()
        is_junk = y == junk_index
        junk_total += int(is_junk.sum()); junk_hit += int((pred[is_junk] == junk_index).sum())
        junk_leak += int((pred[is_junk] != junk_index).sum())
        real = ~is_junk
        real_total += int(real.sum()); real_correct += int((pred[real] == y[real]).sum())
    return {
        "acc": correct / max(1, total),
        "real_acc": real_correct / max(1, real_total),
        "junk_recall": junk_hit / max(1, junk_total),
        "junk_leak": junk_leak / max(1, junk_total),
    }


def main() -> None:
    args = _parse_args()
    device = _device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, classes, envelope, _a = load_checkpoint_model(args.checkpoint, device=device)
    class_index = {c: k for k, c in enumerate(classes)}
    junk_index = class_index[JUNK_TOKEN]
    print(f"init from {args.checkpoint}  classes={len(classes)}")

    real = np.load(args.real_npz, allow_pickle=True)
    x_real = real["x"].astype(np.float32)
    y_real = np.array([class_index[str(t)] for t in real["y"]], dtype=np.int64)
    print(f"real spans: {x_real.shape[0]}  (junk={int((y_real == junk_index).sum())})")

    xs = [np.repeat(x_real, args.real_oversample, axis=0)]
    ys = [np.repeat(y_real, args.real_oversample, axis=0)]
    if args.synth_messages > 0:
        cfg = SpanDatasetConfig(
            classes=REAL_CLASSES, envelope=envelope, num_messages=args.synth_messages,
            seed=args.seed, snr_db_range=(8.0, 30.0), qrn_rate_per_sec_range=(0.0, 0.4),
        )
        xs_s, ys_s, synth_classes = build_span_dataset(cfg)
        assert tuple(synth_classes) == tuple(classes), "class order mismatch"
        xs.append(xs_s); ys.append(ys_s)
        print(f"synthetic spans: {xs_s.shape[0]}")

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(x))
    x, y = x[perm], y[perm]
    n_val = int(len(x) * args.val_frac)
    xv, yv, xt, yt = x[:n_val], y[:n_val], x[n_val:], y[n_val:]

    def loader(xx, yy, shuffle):
        ds = TensorDataset(torch.from_numpy(xx).unsqueeze(1), torch.from_numpy(yy))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader, val_loader = loader(xt, yt, True), loader(xv, yv, False)
    print(f"train={len(xt)} val={len(xv)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_score, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = seen = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward(); optimizer.step()
            tot += float(loss.detach().cpu()) * xb.size(0); seen += xb.size(0)
        scheduler.step()
        m = _evaluate(model, val_loader, junk_index, device)
        score = m["real_acc"] - m["junk_leak"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch:02d} loss={tot / max(1, seen):.4f} real_acc={m['real_acc']:.3f} "
              f"junk_recall={m['junk_recall']:.3f} junk_leak={m['junk_leak']:.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": best_state or model.state_dict(), "classes": classes,
             "envelope": envelope, "args": {**vars(args), "model": "glyph"}},
            args.out,
        )
        print(f"wrote {args.out}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("real_npz")
    p.add_argument("--out", default="outputs/glyph_cnn_real.pt")
    p.add_argument("--real-oversample", type=int, default=1)
    p.add_argument("--synth-messages", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.12)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p.parse_args()


if __name__ == "__main__":
    main()
