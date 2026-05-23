"""Tests for the Morse-realistic corpus generator."""

from __future__ import annotations

from morse_char_recognizer.ctc.corpus import build_corpus, train_test_split


def test_build_corpus_is_deterministic():
    a = build_corpus(seed=42, num_sentences=120)
    b = build_corpus(seed=42, num_sentences=120)
    assert a == b
    assert len(a) == 120


def test_corpus_contains_idiomatic_phrases():
    corpus = build_corpus(seed=0, num_sentences=400)
    assert any("CQ" in line for line in corpus)
    assert any("73" in line for line in corpus)


def test_train_test_split_is_disjoint():
    corpus = build_corpus(seed=0, num_sentences=200)
    train, test = train_test_split(corpus, test_ratio=0.25, seed=0)
    assert len(train) + len(test) == len(corpus)
    assert set(train).isdisjoint(set(test))


def test_train_test_split_is_deterministic():
    corpus = build_corpus(seed=0, num_sentences=200)
    a = train_test_split(corpus, test_ratio=0.25, seed=7)
    b = train_test_split(corpus, test_ratio=0.25, seed=7)
    assert a == b
