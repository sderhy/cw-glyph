"""CTC decoders.

Greedy decoding picks the argmax token per frame and then collapses
repeats and removes blanks. Prefix beam search keeps the top ``beam_width``
partial hypotheses, optionally rescored with a character n-gram language
model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from morse_char_recognizer.ctc.lm import CharNgramLm
from morse_char_recognizer.ctc.vocab import BLANK_INDEX, CtcVocab

NEG_INF = float("-inf")


def greedy_decode(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor | None,
    vocab: CtcVocab,
) -> list[str]:
    """Greedy CTC decode.

    ``log_probs`` is shaped ``(time, batch, num_classes)`` (PyTorch CTC
    convention). Returns one decoded string per batch element.
    """
    if log_probs.ndim != 3:
        raise ValueError(f"log_probs must be 3D, got shape {tuple(log_probs.shape)}")
    time_steps, batch, _ = log_probs.shape
    if input_lengths is None:
        input_lengths = torch.full((batch,), time_steps, dtype=torch.long)

    argmax = log_probs.argmax(dim=-1).transpose(0, 1).cpu().numpy()  # [B, T]
    lengths = input_lengths.cpu().numpy()
    out: list[str] = []
    for row, length in zip(argmax, lengths):
        usable = row[: int(length)]
        out.append(_collapse(usable, vocab))
    return out


def greedy_decode_numpy(
    log_probs: np.ndarray,
    vocab: CtcVocab,
) -> str:
    """Numpy variant for offline / single-sequence use.

    ``log_probs`` is shaped ``(time, num_classes)``.
    """
    if log_probs.ndim != 2:
        raise ValueError(f"log_probs must be 2D, got shape {log_probs.shape}")
    return _collapse(log_probs.argmax(axis=-1), vocab)


def _collapse(indices: np.ndarray, vocab: CtcVocab) -> str:
    out: list[str] = []
    previous = -1
    for index in indices:
        value = int(index)
        if value == BLANK_INDEX:
            previous = value
            continue
        if value != previous:
            out.append(vocab.index_to_char(value))
        previous = value
    return "".join(out).strip()


@dataclass
class BeamSearchConfig:
    """Parameters for CTC prefix beam search with optional LM rescoring."""

    beam_width: int = 16
    prune_log_prob: float = -8.0  # ignore frame tokens below this log prob
    lm_weight: float = 0.0
    insertion_bonus: float = 0.0


@dataclass
class _Beam:
    """One partial hypothesis carrying its blank/non-blank probability split."""

    text: str = ""
    log_pb: float = NEG_INF   # log prob that this prefix ends in blank
    log_pnb: float = NEG_INF  # log prob that this prefix ends in non-blank
    log_lm: float = 0.0       # accumulated LM score
    last_char: str = ""

    def total(self) -> float:
        return _log_sum_exp(self.log_pb, self.log_pnb)

    def score(self, lm_weight: float, insertion_bonus: float) -> float:
        # Beam ordering uses acoustic + lm_weight*lm + insertion_bonus*|text|.
        return self.total() + lm_weight * self.log_lm + insertion_bonus * len(self.text)


def prefix_beam_search(
    log_probs: np.ndarray,
    vocab: CtcVocab,
    *,
    config: BeamSearchConfig | None = None,
    lm: CharNgramLm | None = None,
) -> str:
    """Decode one frame-level log-prob matrix with prefix beam search.

    ``log_probs`` has shape ``(time, num_classes)`` with the CTC blank at
    index ``vocab.blank_index``.
    """
    if log_probs.ndim != 2:
        raise ValueError(f"log_probs must be 2D, got shape {log_probs.shape}")
    config = config or BeamSearchConfig()
    if config.beam_width <= 0:
        raise ValueError("beam_width must be positive")

    initial = _Beam(text="", log_pb=0.0, log_pnb=NEG_INF, log_lm=0.0, last_char="")
    beams: dict[str, _Beam] = {"": initial}

    blank_index = vocab.blank_index
    for frame in log_probs:
        frame = np.asarray(frame, dtype=np.float32)
        candidate_indices = _candidate_indices(frame, blank_index, config.prune_log_prob)
        new_beams: dict[str, _Beam] = {}
        for prefix, beam in beams.items():
            blank_lp = float(frame[blank_index])
            entry = _ensure(new_beams, prefix, beam)
            entry.log_pb = _log_sum_exp(entry.log_pb, beam.total() + blank_lp)

            for index in candidate_indices:
                token = vocab.index_to_char(int(index))
                if not token:
                    continue
                token_lp = float(frame[index])
                if token == beam.last_char:
                    # Same character: extending from a non-blank stays in the
                    # current prefix; extending from a blank actually appends.
                    same = _ensure(new_beams, prefix, beam)
                    same.log_pnb = _log_sum_exp(same.log_pnb, beam.log_pnb + token_lp)

                    new_prefix = prefix + token
                    extended = _ensure_new(new_beams, beams, new_prefix, beam, token)
                    extended.log_pnb = _log_sum_exp(extended.log_pnb, beam.log_pb + token_lp)
                    _maybe_update_lm(extended, beam, token, lm)
                else:
                    new_prefix = prefix + token
                    extended = _ensure_new(new_beams, beams, new_prefix, beam, token)
                    extended.log_pnb = _log_sum_exp(extended.log_pnb, beam.total() + token_lp)
                    _maybe_update_lm(extended, beam, token, lm)
        beams = _prune(new_beams, config)

    best = max(beams.values(), key=lambda b: b.score(config.lm_weight, config.insertion_bonus))
    return best.text.strip()


def _ensure(beams: dict[str, _Beam], key: str, source: _Beam) -> _Beam:
    if key not in beams:
        beams[key] = _Beam(
            text=key,
            log_pb=NEG_INF,
            log_pnb=NEG_INF,
            log_lm=source.log_lm,
            last_char=source.last_char,
        )
    return beams[key]


def _ensure_new(
    new_beams: dict[str, _Beam],
    old_beams: dict[str, _Beam],
    key: str,
    source: _Beam,
    last_char: str,
) -> _Beam:
    if key not in new_beams:
        new_beams[key] = _Beam(
            text=key,
            log_pb=NEG_INF,
            log_pnb=NEG_INF,
            log_lm=source.log_lm,
            last_char=last_char,
        )
    return new_beams[key]


def _maybe_update_lm(
    new_beam: _Beam,
    source: _Beam,
    token: str,
    lm: CharNgramLm | None,
) -> None:
    if lm is None:
        return
    # Only set once per (prefix, token) — the LM transition is a property
    # of the prefix transition rather than acoustics, so subsequent updates
    # would double-count.
    if new_beam.log_lm != source.log_lm:
        return
    context = source.text[-(lm.order - 1) :] if lm.order > 1 else ""
    new_beam.log_lm = source.log_lm + lm.log_prob(token, context)


def _candidate_indices(
    frame: np.ndarray,
    blank_index: int,
    prune_log_prob: float,
) -> list[int]:
    indices = np.flatnonzero(frame >= prune_log_prob)
    return [int(index) for index in indices if int(index) != blank_index]


def _prune(beams: dict[str, _Beam], config: BeamSearchConfig) -> dict[str, _Beam]:
    if len(beams) <= config.beam_width:
        return beams
    ranked = sorted(
        beams.values(),
        key=lambda b: b.score(config.lm_weight, config.insertion_bonus),
        reverse=True,
    )[: config.beam_width]
    return {beam.text: beam for beam in ranked}


def _log_sum_exp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    high = max(a, b)
    return high + math.log(math.exp(a - high) + math.exp(b - high))

