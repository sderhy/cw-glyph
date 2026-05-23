"""Tests for the CTC inference helpers.

These tests use a tiny on-the-fly trained model so they stay
self-contained — the gigabytes of real audio referenced in livetests/ are
not committed.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from morse_char_recognizer.ctc.frames import FrameConfig
from morse_char_recognizer.ctc.inference import decode_audio, load_ctc_checkpoint
from morse_char_recognizer.ctc.model import CrnnCtcConfig, CrnnCtcRecognizer
from morse_char_recognizer.ctc.vocab import CtcVocab


def test_load_ctc_checkpoint_roundtrip(tmp_path: Path) -> None:
    vocab = CtcVocab()
    config = CrnnCtcConfig(num_classes=vocab.num_classes)
    model = CrnnCtcRecognizer(config)
    frame_config = FrameConfig()
    out = tmp_path / "tiny.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(config),
            "vocab_chars": vocab.chars,
            "frame_config": asdict(frame_config),
            "args": {"note": "tiny"},
        },
        out,
    )

    loaded = load_ctc_checkpoint(out)
    assert loaded.vocab.chars == vocab.chars
    assert loaded.frame_config.frame_rate_hz == frame_config.frame_rate_hz
    assert loaded.args["note"] == "tiny"


def test_decode_audio_returns_string_for_empty(tmp_path: Path) -> None:
    vocab = CtcVocab()
    config = CrnnCtcConfig(num_classes=vocab.num_classes)
    model = CrnnCtcRecognizer(config)
    frame_config = FrameConfig()
    out = tmp_path / "tiny.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(config),
            "vocab_chars": vocab.chars,
            "frame_config": asdict(frame_config),
            "args": {},
        },
        out,
    )
    loaded = load_ctc_checkpoint(out)
    text = decode_audio(loaded, np.zeros(0, dtype=np.float32), 8000)
    assert text == ""
