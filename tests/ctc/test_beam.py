"""Tests for the CTC prefix beam search decoder."""

from __future__ import annotations

import math

import numpy as np

from morse_char_recognizer.ctc.decode import (
    BeamSearchConfig,
    greedy_decode_numpy,
    prefix_beam_search,
)
from morse_char_recognizer.ctc.lm import CharNgramLm, default_corpus
from morse_char_recognizer.ctc.vocab import CtcVocab


def _confident_log_probs(text: str, vocab: CtcVocab) -> np.ndarray:
    indices = []
    last = -1
    for char in text:
        idx = vocab.char_to_index(char)
        if idx == last:
            indices.append(vocab.blank_index)
        indices.append(idx)
        last = idx
    frames = np.full((len(indices), vocab.num_classes), -50.0, dtype=np.float32)
    for t, idx in enumerate(indices):
        frames[t, idx] = 0.0
    return frames


def test_beam_search_matches_greedy_for_confident_inputs():
    vocab = CtcVocab()
    log_probs = _confident_log_probs("CQ DE", vocab)
    greedy = greedy_decode_numpy(log_probs, vocab)
    beam = prefix_beam_search(log_probs, vocab, config=BeamSearchConfig(beam_width=4))
    assert beam == greedy


def test_lm_disambiguates_tied_acoustic():
    vocab = CtcVocab()
    # Build a frame sequence where the first frame is split 50/50 between A
    # and E; the rest of the sequence spells " BC".
    log_probs = np.full((6, vocab.num_classes), -50.0, dtype=np.float32)
    log_probs[0, vocab.char_to_index("A")] = math.log(0.5)
    log_probs[0, vocab.char_to_index("E")] = math.log(0.5)
    log_probs[1, vocab.blank_index] = 0.0
    log_probs[2, vocab.char_to_index("B")] = 0.0
    log_probs[3, vocab.blank_index] = 0.0
    log_probs[4, vocab.char_to_index("C")] = 0.0
    log_probs[5, vocab.blank_index] = 0.0

    lm = CharNgramLm(order=3, smoothing=0.05)
    lm.train(["ABC ABC ABC ABC", "ABCD ABCD", "ABCDEF"])

    no_lm = prefix_beam_search(log_probs, vocab, config=BeamSearchConfig(beam_width=8))
    with_lm = prefix_beam_search(
        log_probs,
        vocab,
        config=BeamSearchConfig(beam_width=8, lm_weight=2.0),
        lm=lm,
    )
    assert with_lm == "ABC"
    # without LM, the acoustic tie breaks either way -- check beam still has
    # 3 characters
    assert len(no_lm) == 3
