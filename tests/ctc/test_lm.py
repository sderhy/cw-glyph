"""Tests for the character n-gram LM."""

from __future__ import annotations

import math

from morse_char_recognizer.ctc.lm import CharNgramLm, default_corpus


def test_train_then_known_transition_scores_higher():
    lm = CharNgramLm(order=3, smoothing=0.05)
    lm.train(["THE QUICK BROWN FOX", "THE BROWN FOX JUMPS"])
    seen = lm.log_prob("H", "T")
    unseen = lm.log_prob("Z", "T")
    assert seen > unseen


def test_score_orders_natural_text_above_garbage():
    lm = CharNgramLm(order=3, smoothing=0.1)
    lm.train(default_corpus())
    natural = lm.score("CQ DE F4ABC")
    garbage = lm.score("XQVB UJKZ")
    assert natural > garbage


def test_empty_text_scores_zero():
    lm = CharNgramLm(order=3, smoothing=0.1)
    lm.train(default_corpus())
    assert lm.score("") == 0.0


def test_log_prob_is_finite_for_unseen_context():
    lm = CharNgramLm(order=3, smoothing=0.1)
    lm.train(["AAA"])
    value = lm.log_prob("Z", "QQ")
    assert math.isfinite(value)
