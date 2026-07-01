"""Tests for the CNN-scored element-level beam decoder."""

import numpy as np
import torch
from torch import nn

from morse_char_recognizer.beam_decode import BeamConfig, beam_decode_regions
from morse_char_recognizer.cnn_beam import (
    CnnBeamConfig,
    build_cnn_span_scorer,
    cnn_beam_decode_audio,
)
from morse_char_recognizer.features import EnvelopeConfig
from morse_char_recognizer.segment import SegmentConfig

from tests.test_beam_decode import UNIT, regions_from_text


class ConstLogitModel(nn.Module):
    """Fake CNN that always favours one class, ignoring its input."""

    def __init__(self, num_classes: int, favored_idx: int, strength: float = 10.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.favored_idx = favored_idx
        self.strength = strength

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        logits = torch.zeros(x.shape[0], self.num_classes)
        logits[:, self.favored_idx] = self.strength
        return logits


def tone_audio(regions, sample_rate: int = 8000, freq: float = 600.0) -> np.ndarray:
    """Fill each key-down region with a carrier tone, silence elsewhere."""
    length = regions[-1].end + sample_rate // 4
    audio = np.zeros(length, dtype=np.float32)
    t = np.arange(length) / sample_rate
    tone = np.sin(2 * np.pi * freq * t).astype(np.float32)
    for region in regions:
        audio[region.start : region.end] = tone[region.start : region.end]
    return audio


def test_span_score_flips_ambiguous_segmentation():
    # ".-.." is a single "L" by timing, but an external scorer that rewards the
    # A + I split (spans (0,2) and (2,4)) must be able to override that.
    regions = regions_from_text("L")
    assert beam_decode_regions(regions, UNIT) == "L"

    def prefer_ai(j: int, i: int, _char: str) -> float:
        return 10.0 if (j, i) in {(0, 2), (2, 4)} else 0.0

    assert beam_decode_regions(regions, UNIT, span_score=prefer_ai) == "AI"


def test_cnn_scorer_prefers_favored_glyph():
    classes = ("E", "T", "I", "A", "L")
    model = ConstLogitModel(len(classes), favored_idx=classes.index("L"))
    regions = regions_from_text("L")
    audio = tone_audio(regions)
    scorer = build_cnn_span_scorer(
        audio,
        8000,
        regions,
        UNIT,
        model,
        classes,
        EnvelopeConfig(),
        cnn_config=CnnBeamConfig(w_cnn=1.0),
    )
    # the full-span "L" (favored) must score above the "A" sub-span
    assert scorer(0, 4, "L") > scorer(0, 2, "A")
    # a near-zero log-prob for the favored, strongly-preferred class
    assert scorer(0, 4, "L") > -0.5


def test_cnn_scorer_uses_neutral_for_out_of_vocab_glyph():
    classes = ("E", "T", "A", "L")  # deliberately excludes "I"
    model = ConstLogitModel(len(classes), favored_idx=classes.index("L"))
    regions = regions_from_text("L")
    audio = tone_audio(regions)
    scorer = build_cnn_span_scorer(
        audio,
        8000,
        regions,
        UNIT,
        model,
        classes,
        EnvelopeConfig(),
        cnn_config=CnnBeamConfig(w_cnn=1.0, neutral_logprob=-4.0),
    )
    # span (2, 4) is "..", i.e. "I", which is not a CNN class -> neutral floor
    assert scorer(2, 4, "I") == -4.0


def test_cnn_beam_decode_audio_end_to_end():
    classes = ("E", "T", "I", "A", "L")
    model = ConstLogitModel(len(classes), favored_idx=classes.index("L"))
    regions = regions_from_text("L")
    audio = tone_audio(regions)
    segment_config = SegmentConfig(min_unit_ms=0.0, merge_gap_ms=4.0)
    decoded, detected, unit = cnn_beam_decode_audio(
        audio,
        8000,
        model,
        classes,
        EnvelopeConfig(),
        segment_config=segment_config,
        beam_config=BeamConfig(),
        cnn_config=CnnBeamConfig(w_cnn=1.0),
        unit_samples=UNIT,
    )
    assert len(detected) == 4
    assert decoded == "L"
