"""CNN-scored element-level beam decoding.

The pure-timing beam decoder (:mod:`morse_char_recognizer.beam_decode`) groups
detected dit/dah elements into valid Morse glyphs by exact DP, using only the
inter-element gaps as a soft prior. When an operator's spacing is ambiguous,
several *valid* segmentations of the same elements score alike (e.g. ``.-..``
is a single ``L`` but also ``A`` + ``I``), and timing alone cannot tell them
apart.

This module folds a trained glyph CNN into that search: every candidate span the
DP would consider is scored by the CNN, and the log-probability the CNN assigns
to the span's glyph is added (weighted) to the DP score. The Morse-validity
constraint is untouched — the CNN only *reweights* among segmentations that are
already legal, so it resolves ambiguity (``EDIKE`` -> ``LIKE``) without inventing
illegal glyphs.

Glyphs outside the CNN's class set (e.g. digits or punctuation when the model was
trained on letters) are scored with a neutral floor rather than vetoed, so they
stay available to the timing decoder.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .beam_decode import (
    BeamConfig,
    beam_decode_regions,
    classify_elements,
    enumerate_valid_spans,
    filter_regions,
)
from .features import EnvelopeConfig
from .segment import (
    Region,
    SegmentConfig,
    detect_active_regions,
    estimate_unit_samples,
    extract_segment_envelopes,
)


@dataclass
class CnnBeamConfig:
    """Weights for folding CNN glyph scores into the beam search."""

    w_cnn: float = 1.0  # weight on the CNN log-prob term (0 = pure timing)
    char_bonus: float = 0.0  # per-emitted-char reward; counters the merge/length bias
    neutral_logprob: float = -4.0  # score for glyphs outside the CNN class set
    pad_ms: float = 20.0  # symmetric padding around each span (matches training)
    batch_size: int = 4096  # forward-pass chunk size over candidate spans


@torch.no_grad()
def _log_softmax_batch(
    model: nn.Module,
    envelopes: list[np.ndarray],
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 4096,
) -> np.ndarray:
    """Return the ``(num_spans, num_classes)`` log-softmax matrix for envelopes."""
    if not envelopes:
        return np.zeros((0, 0), dtype=np.float32)
    stacked = np.asarray(envelopes, dtype=np.float32)
    rows: list[np.ndarray] = []
    for start in range(0, len(stacked), batch_size):
        chunk = torch.as_tensor(stacked[start : start + batch_size], dtype=torch.float32)
        if chunk.ndim == 2:
            chunk = chunk.unsqueeze(1)
        logits = model(chunk.to(device))
        rows.append(torch.log_softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(rows, axis=0)


def build_cnn_span_scorer(
    audio: np.ndarray,
    sample_rate: int,
    regions: list[Region],
    unit_samples: int,
    model: nn.Module,
    classes: tuple[str, ...],
    envelope: EnvelopeConfig,
    *,
    beam_config: BeamConfig | None = None,
    cnn_config: CnnBeamConfig | None = None,
    device: torch.device | str = "cpu",
) -> Callable[[int, int, str], float]:
    """Build a ``span_score(j, i, char)`` callback backed by the glyph CNN.

    ``regions`` must be the raw detected elements; the same ``min_element_units``
    filter the DP applies is applied here so the returned callback's ``(j, i)``
    indices line up with :func:`beam_decode_regions` on those regions.
    """
    beam_config = beam_config or BeamConfig()
    cnn_config = cnn_config or CnnBeamConfig()

    filtered = filter_regions(regions, unit_samples, beam_config.min_element_units)
    symbols = classify_elements(
        filtered, unit_samples, dot_dash_split=beam_config.dot_dash_split
    )
    spans = enumerate_valid_spans(
        symbols,
        max_elements_per_char=beam_config.max_elements_per_char,
        prefer_letters=beam_config.prefer_letters,
    )

    pad = int(round(cnn_config.pad_ms / 1000.0 * sample_rate))
    audio_len = len(audio)
    span_regions = [
        Region(
            max(0, filtered[j].start - pad),
            min(audio_len, filtered[i - 1].end + pad),
        )
        for j, i, _char in spans
    ]
    envelopes = extract_segment_envelopes(
        audio, sample_rate, span_regions, envelope, unit_samples=unit_samples
    )
    logprobs = _log_softmax_batch(
        model, envelopes, device=device, batch_size=cnn_config.batch_size
    )

    class_index = {char: idx for idx, char in enumerate(classes)}
    span_logprob: dict[tuple[int, int], float] = {}
    for (j, i, char), row in zip(spans, logprobs):
        idx = class_index.get(char)
        span_logprob[(j, i)] = (
            float(row[idx]) if idx is not None else cnn_config.neutral_logprob
        )

    w_cnn = cnn_config.w_cnn
    neutral = cnn_config.neutral_logprob
    char_bonus = cnn_config.char_bonus

    def scorer(j: int, i: int, _char: str) -> float:
        # Summing per-character log-probs across segmentations of different
        # length biases the search toward merging (fewer, hence fewer negative
        # terms); char_bonus is the per-character insertion reward that cancels
        # it, exactly as a word-insertion penalty does in ASR beam search.
        return w_cnn * span_logprob.get((j, i), neutral) + char_bonus

    return scorer


def cnn_beam_decode_audio(
    audio: np.ndarray,
    sample_rate: int,
    model: nn.Module,
    classes: tuple[str, ...],
    envelope: EnvelopeConfig,
    *,
    segment_config: SegmentConfig | None = None,
    beam_config: BeamConfig | None = None,
    cnn_config: CnnBeamConfig | None = None,
    unit_samples: int | None = None,
    device: torch.device | str = "cpu",
) -> tuple[str, list[Region], int]:
    """Detect elements, score glyph candidates with the CNN, and DP-decode.

    Returns ``(text, regions, unit_samples)`` like
    :func:`morse_char_recognizer.beam_decode.beam_decode_audio`.
    """
    segment_config = segment_config or SegmentConfig()
    beam_config = beam_config or BeamConfig()
    regions = detect_active_regions(audio, sample_rate, segment_config)
    if unit_samples is None:
        min_unit_ms = segment_config.min_unit_ms or 35.0
        unit_samples = estimate_unit_samples(
            regions,
            sample_rate,
            min_unit_ms=min_unit_ms,
            max_unit_ms=segment_config.max_unit_ms,
            histogram_bin_ms=segment_config.unit_histogram_bin_ms,
        )
    if not regions or unit_samples <= 0:
        return "", regions, unit_samples

    scorer = build_cnn_span_scorer(
        audio,
        sample_rate,
        regions,
        unit_samples,
        model,
        classes,
        envelope,
        beam_config=beam_config,
        cnn_config=cnn_config,
        device=device,
    )
    text = beam_decode_regions(regions, unit_samples, beam_config, span_score=scorer)
    return text, regions, unit_samples
