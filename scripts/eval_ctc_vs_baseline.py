"""Compare CTC continuous-audio decoding to the per-character CNN baseline.

The evaluation generates a synthetic continuous-audio test set and decodes
every example with:

1. the CTC recogniser (no per-character segmentation);
2. the per-character CNN baseline through ``predict_audio_segments`` +
   ``join_segment_predictions`` (the existing pipeline).

Aggregate CER is reported for both pipelines so the CTC pipeline can be
gated on "≥ baseline weighted accuracy".
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morse_char_recognizer.ctc.dataset import CtcDatasetConfig, SyntheticCtcDataset
from morse_char_recognizer.ctc.frames import FrameConfig
from morse_char_recognizer.ctc.inference import decode_audio, load_ctc_checkpoint
from morse_char_recognizer.ctc.metrics import aggregate_cer
from morse_char_recognizer.ctc.vocab import CtcVocab
from morse_char_recognizer.inference import (
    join_segment_predictions,
    load_checkpoint_model,
    predict_audio_segments,
)
from morse_char_recognizer.segment import (
    SegmentConfig,
    detect_active_regions,
    estimate_unit_samples,
)


def main() -> None:
    args = _parse_args()
    device = _device(args.device)
    ctc = load_ctc_checkpoint(args.ctc_checkpoint, device=device)

    baseline = None
    if args.baseline_checkpoint:
        model, classes, envelope, _baseline_args = load_checkpoint_model(
            args.baseline_checkpoint,
            device=device,
        )
        if args.scale_mode != "checkpoint":
            envelope = replace(
                envelope,
                scale_mode=args.scale_mode,
                canonical_units=args.canonical_units,
            )
        baseline = (model, classes, envelope)

    dataset = SyntheticCtcDataset(
        CtcDatasetConfig(
            vocab=ctc.vocab,
            frames=FrameConfig(frame_rate_hz=args.frame_rate_hz),
            num_examples=args.num_examples,
            sample_rate=args.sample_rate,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            word_probability=args.word_prob,
            seed=args.seed,
            wpm_range=(args.wpm_min, args.wpm_max),
            snr_db_range=_snr_range(args),
        )
    )

    ctc_pairs: list[tuple[str, str]] = []
    base_pairs: list[tuple[str, str]] = []
    rows: list[dict] = []
    for index, example in enumerate(dataset):
        ctc_hyp = decode_audio(ctc, example.audio, example.sample_rate, device=device)
        base_hyp = ""
        if baseline is not None:
            base_hyp = _baseline_decode(baseline, example.audio, example.sample_rate, device)
        ctc_pairs.append((example.text, ctc_hyp))
        if baseline is not None:
            base_pairs.append((example.text, base_hyp))
        if index < args.print_examples:
            print(f"  ref={example.text!r}")
            print(f"  ctc={ctc_hyp!r}")
            if baseline is not None:
                print(f"  cnn={base_hyp!r}")
            print("  ---")
        rows.append({"reference": example.text, "ctc": ctc_hyp, "baseline": base_hyp})

    ctc_cer = aggregate_cer(ctc_pairs)
    print(
        f"ctc       cer={ctc_cer.cer:.4f} acc={ctc_cer.accuracy:.4f} "
        f"edits={ctc_cer.distance}/{ctc_cer.reference_length}"
    )
    if baseline is not None:
        base_cer = aggregate_cer(base_pairs)
        print(
            f"baseline  cer={base_cer.cer:.4f} acc={base_cer.accuracy:.4f} "
            f"edits={base_cer.distance}/{base_cer.reference_length}"
        )
        delta = base_cer.cer - ctc_cer.cer
        print(f"delta_cer (baseline - ctc) = {delta:+.4f}")
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rows": rows, "ctc": ctc_cer.__dict__}, indent=2))
        print(f"wrote {out}")


def _baseline_decode(
    baseline,
    audio,
    sample_rate: int,
    device: torch.device | str,
) -> str:
    model, classes, envelope = baseline
    segment_config = SegmentConfig()
    unit_samples = estimate_unit_samples(detect_active_regions(audio, sample_rate, segment_config))
    predictions = predict_audio_segments(
        model,
        audio,
        sample_rate,
        classes=classes,
        envelope=envelope,
        segment_config=segment_config,
        allowed_classes=tuple(c for c in CtcVocab().chars if c != " " and c in classes),
        device=device,
    )
    return join_segment_predictions(
        predictions,
        sample_rate=sample_rate,
        unit_samples=unit_samples,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctc-checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--min-chars", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=12)
    parser.add_argument("--word-prob", type=float, default=0.25)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--frame-rate-hz", type=float, default=200.0)
    parser.add_argument("--wpm-min", type=float, default=14.0)
    parser.add_argument("--wpm-max", type=float, default=28.0)
    parser.add_argument("--snr-min", type=float, default=None)
    parser.add_argument("--snr-max", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--scale-mode",
        choices=("checkpoint", "stretch", "unit"),
        default="checkpoint",
    )
    parser.add_argument("--canonical-units", type=float, default=32.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--print-examples", type=int, default=5)
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def _snr_range(args: argparse.Namespace) -> tuple[float, float] | None:
    if args.snr_min is None and args.snr_max is None:
        return None
    if args.snr_min is None or args.snr_max is None:
        raise SystemExit("--snr-min and --snr-max must be provided together")
    return (args.snr_min, args.snr_max)


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
