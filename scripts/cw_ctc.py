"""Test the CTC Morse recogniser on a WAV file or a synthesised string.

Examples:

    # decode an existing WAV with the trained checkpoint
    python scripts/cw_ctc.py outputs/ctc_phase2_noisy.pt path/to/audio.wav

    # synthesise "CQ DE F4ABC" and decode it (useful without real audio)
    python scripts/cw_ctc.py outputs/ctc_phase2_noisy.pt \\
        --synth "CQ DE F4ABC" --wpm 20 --snr 6

    # beam search + character LM rescoring
    python scripts/cw_ctc.py outputs/ctc_phase2_noisy.pt audio.wav \\
        --beam --lm-weight 0.4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from morse_char_recognizer.audio_io import read_wav_mono
from morse_char_recognizer.ctc.corpus import build_corpus, train_test_split
from morse_char_recognizer.ctc.decode import (
    BeamSearchConfig,
    greedy_decode_numpy,
    prefix_beam_search,
)
from morse_char_recognizer.ctc.inference import frame_log_probs, load_ctc_checkpoint
from morse_char_recognizer.ctc.lm import CharNgramLm
from morse_char_recognizer.ctc.metrics import edit_distance
from morse_synth.channel import ChannelConfig
from morse_synth.core import render
from morse_synth.keying import KeyingConfig
from morse_synth.operator import OperatorConfig


def main() -> None:
    args = _parse_args()
    if not args.wav and not args.synth:
        raise SystemExit("provide a WAV path or --synth TEXT")

    device = _device(args.device)
    loaded = load_ctc_checkpoint(args.checkpoint, device=device)

    sample_rate, audio, reference = _load_audio(args)

    started = time.perf_counter()
    log_probs = frame_log_probs(loaded, audio, sample_rate, device=device)
    encode_ms = (time.perf_counter() - started) * 1000.0

    if log_probs.size == 0:
        print("(no frames decoded)")
        return

    greedy = greedy_decode_numpy(log_probs, loaded.vocab)

    beam_decoded: str | None = None
    if args.beam:
        lm = _build_lm(args, loaded.vocab) if args.lm_weight > 0 else None
        beam_config = BeamSearchConfig(
            beam_width=args.beam_width,
            lm_weight=args.lm_weight,
            insertion_bonus=args.insertion_bonus,
            prune_log_prob=args.prune_log_prob,
        )
        beam_decoded = prefix_beam_search(log_probs, loaded.vocab, config=beam_config, lm=lm)

    _print_results(args, reference, greedy, beam_decoded, audio, sample_rate, log_probs, encode_ms)


def _load_audio(args: argparse.Namespace) -> tuple[int, np.ndarray, str | None]:
    if args.synth:
        audio = _synthesise(args)
        return args.sample_rate, audio, args.synth.upper()
    sample_rate, audio = read_wav_mono(args.wav)
    return sample_rate, audio, None


def _synthesise(args: argparse.Namespace) -> np.ndarray:
    channel = (
        ChannelConfig(seed=args.seed)
        if args.snr is None
        else ChannelConfig(snr_db=args.snr, seed=args.seed)
    )
    audio = render(
        args.synth,
        operator=OperatorConfig(wpm=args.wpm, seed=args.seed),
        keying=KeyingConfig(rise_ms=5.0),
        channel=channel,
        freq=args.freq,
        sample_rate=args.sample_rate,
        amplitude=0.6,
        tail_ms=50.0,
    )
    pad = np.zeros(int(round(0.1 * args.sample_rate)), dtype=np.float32)
    return np.concatenate([pad, audio.astype(np.float32), pad])


def _build_lm(args: argparse.Namespace, vocab) -> CharNgramLm:
    if args.lm_corpus:
        lines = [line for line in Path(args.lm_corpus).read_text().splitlines() if line.strip()]
    else:
        corpus = build_corpus(seed=0, num_sentences=args.corpus_size)
        lines, _ = train_test_split(corpus, test_ratio=0.0001, seed=0)
        if not lines:
            lines = corpus
    lm = CharNgramLm(vocab=vocab, order=args.lm_order, smoothing=args.lm_smoothing)
    lm.train(lines)
    return lm


def _print_results(
    args: argparse.Namespace,
    reference: str | None,
    greedy: str,
    beam: str | None,
    audio: np.ndarray,
    sample_rate: int,
    log_probs: np.ndarray,
    encode_ms: float,
) -> None:
    audio_s = audio.size / sample_rate
    print(f"audio    {audio_s:.2f}s  frames={log_probs.shape[0]}  encode={encode_ms:.0f}ms")
    if reference is not None:
        print(f"ref      {reference!r}")
    print(f"greedy   {greedy!r}")
    if reference is not None:
        print(f"  edits {edit_distance(reference, greedy)} / {len(reference)}")
    if beam is not None:
        print(f"beam+lm  {beam!r}")
        if reference is not None:
            print(f"  edits {edit_distance(reference, beam)} / {len(reference)}")

    if args.show_frames:
        _dump_frame_predictions(log_probs, args.show_frames)


def _dump_frame_predictions(log_probs: np.ndarray, top_k: int) -> None:
    import math

    print(f"\nframe-by-frame argmax (top-{top_k}, blank suppressed):")
    last_label = -1
    for frame_index, frame in enumerate(log_probs):
        top_indices = np.argsort(frame)[::-1][:top_k]
        labels = []
        for index in top_indices:
            label = int(index)
            if label == 0:
                continue
            prob = math.exp(float(frame[label]))
            labels.append(f"idx{label}:{prob:.2f}")
        argmax_label = int(np.argmax(frame))
        if argmax_label == last_label:
            continue
        last_label = argmax_label
        marker = "_" if argmax_label == 0 else "*"
        print(f"  t={frame_index:04d} {marker} top={','.join(labels)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="path to a CTC checkpoint (.pt)")
    parser.add_argument("wav", nargs="?", default=None, help="WAV file to decode (optional)")
    parser.add_argument("--synth", default=None, help="synthesise this text instead of loading a WAV")
    parser.add_argument("--wpm", type=float, default=20.0, help="speed for --synth")
    parser.add_argument("--freq", type=float, default=600.0, help="carrier frequency for --synth")
    parser.add_argument("--snr", type=float, default=None, help="SNR dB for --synth (none = clean)")
    parser.add_argument("--sample-rate", type=int, default=8000, help="sample rate for --synth")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--beam", action="store_true", help="also run prefix beam search")
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--lm-weight", type=float, default=0.4)
    parser.add_argument("--insertion-bonus", type=float, default=0.0)
    parser.add_argument("--prune-log-prob", type=float, default=-8.0)
    parser.add_argument("--lm-order", type=int, default=4)
    parser.add_argument("--lm-smoothing", type=float, default=0.1)
    parser.add_argument("--lm-corpus", default=None, help="path to a text corpus (one line per sentence)")
    parser.add_argument("--corpus-size", type=int, default=600)

    parser.add_argument("--show-frames", type=int, default=0, metavar="K",
                        help="dump frame argmax (top-K) on state change")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
