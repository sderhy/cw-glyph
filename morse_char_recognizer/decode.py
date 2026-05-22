"""End-to-end baseline decoding for continuous Morse audio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from morse_char_recognizer.baseline import Prediction, TemplateClassifier, TemplateConfig
from morse_char_recognizer.features import EnvelopeConfig
from morse_char_recognizer.segment import (
    Region,
    SegmentConfig,
    extract_segment_envelopes,
    segment_characters,
)


@dataclass(frozen=True)
class DecodedCharacter:
    """One decoded character candidate."""

    char: str
    label: int
    score: float
    segment: Region


def decode_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    segment_config: SegmentConfig | None = None,
    envelope_config: EnvelopeConfig | None = None,
    template_config: TemplateConfig | None = None,
    classifier: TemplateClassifier | None = None,
) -> list[DecodedCharacter]:
    """Segment continuous audio and classify each candidate character."""
    segments = segment_characters(audio, sample_rate, segment_config)
    envelopes = extract_segment_envelopes(audio, sample_rate, segments, envelope_config)

    clf = classifier or TemplateClassifier(template_config)
    predictions = clf.predict(envelopes)
    return [
        _decoded_character(prediction, segment)
        for prediction, segment in zip(predictions, segments)
    ]


def decoded_text(decoded: list[DecodedCharacter]) -> str:
    """Join decoded character candidates into text without word spacing."""
    return "".join(item.char for item in decoded)


def _decoded_character(prediction: Prediction, segment: Region) -> DecodedCharacter:
    return DecodedCharacter(
        char=prediction.char,
        label=prediction.label,
        score=prediction.score,
        segment=segment,
    )
