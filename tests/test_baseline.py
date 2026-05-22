"""Tests for the template-matching baseline."""

from __future__ import annotations

from morse_char_recognizer.baseline import TemplateClassifier, TemplateConfig
from morse_char_recognizer.features import EnvelopeConfig, extract_envelope
from morse_synth.core import synthesize


def test_template_classifier_predicts_clean_isolated_character() -> None:
    config = TemplateConfig(classes=("A", "B", "S"), envelope=EnvelopeConfig(length=128))
    classifier = TemplateClassifier(config)
    audio = synthesize("S", wpm=20, sample_rate=8000, rise_ms=3.0)
    envelope = extract_envelope(audio, 8000, config.envelope)

    prediction = classifier.predict_one(envelope)

    assert prediction.char == "S"
    assert prediction.score > 0.95
