"""Compare greedy CTC against prefix-beam-search + LM on noisy synth.

The script generates a held-out synthetic continuous-audio test set, runs
the encoder once, and decodes the frame log-probs with two strategies:

1. greedy CTC;
2. prefix beam search with a character n-gram LM trained from a small
   Morse corpus.

It reports aggregate CER for both decoders. Phase 3 of the CTC plan
requires the LM-rescored beam search to beat greedy by at least one
absolute CER point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morse_char_recognizer.ctc.corpus import build_corpus, train_test_split
from morse_char_recognizer.ctc.dataset import CtcDatasetConfig, SyntheticCtcDataset
from morse_char_recognizer.ctc.decode import (
    BeamSearchConfig,
    greedy_decode_numpy,
    prefix_beam_search,
)
from morse_char_recognizer.ctc.frames import FrameConfig
from morse_char_recognizer.ctc.inference import frame_log_probs, load_ctc_checkpoint
from morse_char_recognizer.ctc.lm import CharNgramLm
from morse_char_recognizer.ctc.metrics import aggregate_cer


def main() -> None:
    args = _parse_args()
    device = _device(args.device)
    loaded = load_ctc_checkpoint(args.checkpoint, device=device)

    train_texts, test_texts = _build_split(args)
    lm = CharNgramLm(vocab=loaded.vocab, order=args.lm_order, smoothing=args.lm_smoothing)
    lm.train(train_texts)

    dataset = SyntheticCtcDataset(
        CtcDatasetConfig(
            vocab=loaded.vocab,
            frames=FrameConfig(frame_rate_hz=args.frame_rate_hz),
            num_examples=args.num_examples,
            sample_rate=args.sample_rate,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            word_probability=args.word_prob,
            text_pool=tuple(test_texts) if test_texts else None,
            seed=args.seed,
            wpm_range=(args.wpm_min, args.wpm_max),
            snr_db_range=(args.snr_min, args.snr_max),
        )
    )

    beam_config = BeamSearchConfig(
        beam_width=args.beam_width,
        lm_weight=args.lm_weight,
        insertion_bonus=args.insertion_bonus,
        prune_log_prob=args.prune_log_prob,
    )

    greedy_pairs: list[tuple[str, str]] = []
    beam_pairs: list[tuple[str, str]] = []
    rows: list[dict] = []
    for index, example in enumerate(dataset):
        log_probs = frame_log_probs(loaded, example.audio, example.sample_rate, device=device)
        greedy = greedy_decode_numpy(log_probs, loaded.vocab) if log_probs.size else ""
        beam = (
            prefix_beam_search(log_probs, loaded.vocab, config=beam_config, lm=lm)
            if log_probs.size
            else ""
        )
        greedy_pairs.append((example.text, greedy))
        beam_pairs.append((example.text, beam))
        rows.append({"reference": example.text, "greedy": greedy, "beam_lm": beam})
        if index < args.print_examples:
            print(f"  ref={example.text!r}")
            print(f"  greedy={greedy!r}")
            print(f"  beam  ={beam!r}")
            print("  ---")

    greedy_result = aggregate_cer(greedy_pairs)
    beam_result = aggregate_cer(beam_pairs)
    delta = greedy_result.cer - beam_result.cer
    print(
        f"greedy   cer={greedy_result.cer:.4f} acc={greedy_result.accuracy:.4f} "
        f"edits={greedy_result.distance}/{greedy_result.reference_length}"
    )
    print(
        f"beam+lm  cer={beam_result.cer:.4f} acc={beam_result.accuracy:.4f} "
        f"edits={beam_result.distance}/{beam_result.reference_length}"
    )
    print(f"delta_cer (greedy - beam_lm) = {delta:+.4f}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "greedy": greedy_result.__dict__,
                    "beam_lm": beam_result.__dict__,
                    "delta": delta,
                    "config": vars(args),
                    "rows": rows,
                },
                indent=2,
            )
        )
        print(f"wrote {out}")


def _build_split(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if args.corpus:
        all_lines = [line for line in Path(args.corpus).read_text().splitlines() if line.strip()]
    else:
        all_lines = build_corpus(seed=args.corpus_seed, num_sentences=args.corpus_size)
    return train_test_split(all_lines, test_ratio=args.test_ratio, seed=args.corpus_seed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--corpus-size", type=int, default=600)
    parser.add_argument("--corpus-seed", type=int, default=0)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--min-chars", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=12)
    parser.add_argument("--word-prob", type=float, default=0.25)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--frame-rate-hz", type=float, default=200.0)
    parser.add_argument("--wpm-min", type=float, default=14.0)
    parser.add_argument("--wpm-max", type=float, default=28.0)
    parser.add_argument("--snr-min", type=float, default=-2.0)
    parser.add_argument("--snr-max", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--lm-weight", type=float, default=0.4)
    parser.add_argument("--insertion-bonus", type=float, default=0.0)
    parser.add_argument("--prune-log-prob", type=float, default=-8.0)
    parser.add_argument("--lm-order", type=int, default=4)
    parser.add_argument("--lm-smoothing", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--print-examples", type=int, default=5)
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
