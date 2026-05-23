"""Tests for the audio → envelope-frame encoder."""

from __future__ import annotations

import numpy as np
import pytest

from morse_char_recognizer.ctc.frames import FrameConfig, audio_to_frames
from morse_synth.core import synthesize


def test_frame_count_matches_frame_rate():
    sample_rate = 8000
    audio = synthesize("E", wpm=20.0, sample_rate=sample_rate)
    frames = audio_to_frames(audio, sample_rate, FrameConfig(frame_rate_hz=200.0))
    expected = round(audio.size * 200.0 / sample_rate)
    assert abs(frames.size - expected) <= 1


def test_silent_audio_returns_zero_frames():
    frames = audio_to_frames(np.zeros(800, dtype=np.float32), 8000)
    assert frames.size > 0
    assert np.all(frames == 0.0)


def test_peak_normalisation_clamps_at_one():
    audio = synthesize("PARIS", wpm=20.0, sample_rate=8000)
    frames = audio_to_frames(audio, 8000, FrameConfig(frame_rate_hz=200.0))
    assert frames.max() <= 1.0 + 1e-6
    assert frames.min() >= 0.0


def test_empty_audio_returns_empty():
    frames = audio_to_frames(np.zeros(0, dtype=np.float32), 8000)
    assert frames.size == 0


def test_invalid_sample_rate_raises():
    with pytest.raises(ValueError):
        audio_to_frames(np.zeros(10, dtype=np.float32), 0)
