"""Tests for end-to-end baseline decoding."""

from __future__ import annotations

from morse_char_recognizer.baseline import TemplateConfig
from morse_char_recognizer.decode import decode_audio, decoded_text
from morse_char_recognizer.features import EnvelopeConfig
from morse_synth.core import synthesize


def test_decode_audio_decodes_clean_sos() -> None:
    envelope = EnvelopeConfig(length=128)
    audio = synthesize("SOS", wpm=20, sample_rate=8000, rise_ms=3.0)

    decoded = decode_audio(
        audio,
        8000,
        envelope_config=envelope,
        template_config=TemplateConfig(classes=("S", "O"), envelope=envelope),
    )

    assert decoded_text(decoded) == "SOS"
    assert all(item.score > 0.7 for item in decoded)
