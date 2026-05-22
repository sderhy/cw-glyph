"""Inference helpers for trained Morse glyph CNN checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from morse_char_recognizer.features import EnvelopeConfig
from morse_char_recognizer.model import MorseGlyphCnn1d, SmallCnn1d
from morse_char_recognizer.segment import (
    Region,
    SegmentConfig,
    detect_active_regions,
    estimate_unit_samples,
    extract_segment_envelopes,
    segment_characters,
)


def load_checkpoint_model(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, tuple[str, ...], EnvelopeConfig, dict[str, Any]]:
    """Load a trained CNN checkpoint written by ``scripts/train_cnn.py``."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = tuple(checkpoint["classes"])
    args = checkpoint.get("args", {})
    envelope = checkpoint.get("envelope", EnvelopeConfig())

    model_name = args.get("model", "glyph")
    if model_name == "small":
        model: nn.Module = SmallCnn1d(num_classes=len(classes))
    else:
        model = MorseGlyphCnn1d(
            num_classes=len(classes),
            dropout=float(args.get("dropout", 0.08)),
        )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, classes, envelope, args


@torch.no_grad()
def predict_envelopes(
    model: nn.Module,
    envelopes: list[np.ndarray] | np.ndarray,
    *,
    classes: tuple[str, ...] | None = None,
    allowed_classes: tuple[str, ...] | None = None,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels and confidence scores for fixed-length envelopes."""
    x = torch.as_tensor(np.asarray(envelopes), dtype=torch.float32)
    if x.numel() == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    if x.ndim == 2:
        x = x.unsqueeze(1)
    logits = model(x.to(device))
    if allowed_classes is not None:
        if classes is None:
            raise ValueError("classes must be provided when allowed_classes is used")
        logits = _mask_logits(logits, classes, allowed_classes)
    probabilities = torch.softmax(logits, dim=1)
    scores, labels = torch.max(probabilities, dim=1)
    return labels.cpu().numpy(), scores.cpu().numpy()


@torch.no_grad()
def predict_envelopes_topk(
    model: nn.Module,
    envelopes: list[np.ndarray] | np.ndarray,
    *,
    classes: tuple[str, ...],
    allowed_classes: tuple[str, ...] | None = None,
    k: int = 3,
    device: torch.device | str = "cpu",
) -> list[list[tuple[str, float]]]:
    """Predict top-k labels and confidence scores for fixed-length envelopes."""
    x = torch.as_tensor(np.asarray(envelopes), dtype=torch.float32)
    if x.numel() == 0:
        return []
    if x.ndim == 2:
        x = x.unsqueeze(1)
    logits = model(x.to(device))
    if allowed_classes is not None:
        logits = _mask_logits(logits, classes, allowed_classes)
    probabilities = torch.softmax(logits, dim=1)
    top_k = min(max(1, k), len(classes))
    scores, labels = torch.topk(probabilities, k=top_k, dim=1)
    out = []
    for row_labels, row_scores in zip(labels.cpu().numpy(), scores.cpu().numpy()):
        out.append(
            [
                (classes[int(label)], float(score))
                for label, score in zip(row_labels, row_scores)
            ]
        )
    return out


@torch.no_grad()
def predict_audio_segments(
    model: nn.Module,
    audio: np.ndarray,
    sample_rate: int,
    *,
    classes: tuple[str, ...],
    envelope: EnvelopeConfig,
    segment_config: SegmentConfig | None = None,
    allowed_classes: tuple[str, ...] | None = None,
    device: torch.device | str = "cpu",
) -> list[tuple[str, float, Region]]:
    """Segment continuous audio and classify each candidate with a CNN."""
    segments = segment_characters(audio, sample_rate, segment_config)
    if not segments:
        return []
    unit_samples = None
    if envelope.scale_mode == "unit":
        active = detect_active_regions(audio, sample_rate, segment_config)
        unit_samples = estimate_unit_samples(active)
    envelopes = extract_segment_envelopes(
        audio,
        sample_rate,
        segments,
        envelope,
        unit_samples=unit_samples,
    )
    labels, scores = predict_envelopes(
        model,
        envelopes,
        classes=classes,
        allowed_classes=allowed_classes,
        device=device,
    )
    return [
        (classes[int(label)], float(score), segment)
        for label, score, segment in zip(labels, scores, segments)
    ]


def join_segment_predictions(
    predictions: list[tuple[str, float, Region]],
    *,
    sample_rate: int,
    word_gap_ms: float = 250.0,
    unit_samples: int | None = None,
    word_gap_units: float = 4.5,
) -> str:
    """Join segment predictions, inserting spaces for long inter-character gaps."""
    if not predictions:
        return ""

    out = [predictions[0][0]]
    if unit_samples is not None and unit_samples > 0:
        threshold = unit_samples * word_gap_units
    else:
        threshold = word_gap_ms / 1000.0 * sample_rate
    previous = predictions[0][2]
    for char, _score, segment in predictions[1:]:
        if segment.start - previous.end >= threshold:
            out.append(" ")
        out.append(char)
        previous = segment
    return "".join(out)


def _mask_logits(
    logits: torch.Tensor,
    classes: tuple[str, ...],
    allowed_classes: tuple[str, ...],
) -> torch.Tensor:
    allowed = set(allowed_classes)
    unknown = allowed.difference(classes)
    if unknown:
        raise ValueError(f"allowed_classes not present in checkpoint classes: {sorted(unknown)}")
    mask = torch.full_like(logits, float("-inf"))
    indices = [index for index, char in enumerate(classes) if char in allowed]
    if not indices:
        raise ValueError("allowed_classes produced an empty class mask")
    mask[:, indices] = 0.0
    return logits + mask
