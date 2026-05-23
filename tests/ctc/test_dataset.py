"""Tests for the synthetic CTC sequence dataset."""

from __future__ import annotations

import numpy as np

from morse_char_recognizer.ctc.dataset import (
    CtcDatasetConfig,
    SyntheticCtcDataset,
    collate_ctc,
)
from morse_char_recognizer.ctc.vocab import CtcVocab


def test_dataset_length_matches_config():
    ds = SyntheticCtcDataset(CtcDatasetConfig(num_examples=12, seed=0))
    assert len(ds) == 12


def test_examples_are_deterministic():
    config = CtcDatasetConfig(num_examples=4, seed=42)
    a = SyntheticCtcDataset(config)[2]
    b = SyntheticCtcDataset(config)[2]
    assert a.text == b.text
    assert np.array_equal(a.frames, b.frames)
    assert a.targets == b.targets


def test_targets_match_text():
    vocab = CtcVocab()
    ds = SyntheticCtcDataset(CtcDatasetConfig(num_examples=8, seed=1, vocab=vocab))
    for example in ds:
        assert vocab.decode(example.targets) == example.text.upper()
        assert example.frames.ndim == 1
        assert example.frames.size > len(example.targets)


def test_collate_pads_and_concatenates():
    ds = SyntheticCtcDataset(CtcDatasetConfig(num_examples=6, seed=3))
    batch = collate_ctc([ds[i] for i in range(4)])
    assert batch["inputs"].shape[0] == 4
    assert batch["inputs"].shape[1] == 1
    assert batch["inputs"].shape[2] == int(batch["input_lengths"].max().item())
    assert batch["targets"].numel() == int(batch["target_lengths"].sum().item())


def test_text_pool_restricts_sampled_texts():
    vocab = CtcVocab()
    pool = ("CQ DE F4ABC", "TNX FB 73")
    ds = SyntheticCtcDataset(
        CtcDatasetConfig(
            vocab=vocab,
            num_examples=20,
            seed=0,
            text_pool=pool,
        )
    )
    seen = {example.text for example in ds}
    assert seen <= set(pool)
    assert len(seen) >= 1
