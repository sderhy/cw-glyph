"""Template-matching baseline for Morse character glyphs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from morse_char_recognizer.classes import DEFAULT_CLASSES
from morse_char_recognizer.features import EnvelopeConfig, extract_envelope
from morse_synth.core import synthesize


@dataclass(frozen=True)
class TemplateConfig:
    """Configuration for ideal synthetic character templates."""

    classes: tuple[str, ...] = DEFAULT_CLASSES
    sample_rate: int = 8000
    envelope: EnvelopeConfig = field(default_factory=EnvelopeConfig)
    wpm: float = 20.0
    freq: float = 600.0
    rise_ms: float = 3.0
    amplitude: float = 0.8
    tail_ms: float = 50.0


@dataclass(frozen=True)
class Prediction:
    """Nearest-template prediction result."""

    char: str
    label: int
    score: float


class TemplateClassifier:
    """Classify fixed-length envelopes by normalized dot product."""

    def __init__(self, config: TemplateConfig | None = None) -> None:
        self.config = config or TemplateConfig()
        self.classes = tuple(self.config.classes)
        self.templates = build_templates(self.config)
        self._normalized_templates = _normalize_rows(self.templates)

    def predict_one(self, envelope: np.ndarray) -> Prediction:
        x = _normalize_vector(np.asarray(envelope, dtype=np.float32))
        scores = self._normalized_templates @ x
        label = int(np.argmax(scores))
        return Prediction(char=self.classes[label], label=label, score=float(scores[label]))

    def predict(self, envelopes: list[np.ndarray] | np.ndarray) -> list[Prediction]:
        return [self.predict_one(envelope) for envelope in envelopes]


def build_templates(config: TemplateConfig | None = None) -> np.ndarray:
    """Build one ideal envelope template per configured class."""
    config = config or TemplateConfig()
    templates = []
    for char in config.classes:
        audio = synthesize(
            char,
            wpm=config.wpm,
            freq=config.freq,
            sample_rate=config.sample_rate,
            rise_ms=config.rise_ms,
            amplitude=config.amplitude,
            tail_ms=config.tail_ms,
        )
        templates.append(extract_envelope(audio, config.sample_rate, config.envelope))
    return np.stack(templates).astype(np.float32)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    return np.stack([_normalize_vector(row) for row in x]).astype(np.float32)


def _normalize_vector(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32, copy=False)
    y = y - float(np.mean(y))
    norm = float(np.linalg.norm(y))
    if norm <= 1e-12:
        return np.zeros_like(y, dtype=np.float32)
    return (y / norm).astype(np.float32)
