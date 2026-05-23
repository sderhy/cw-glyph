"""Run a trained CRNN-CTC recogniser on continuous audio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from morse_char_recognizer.ctc.decode import greedy_decode_numpy
from morse_char_recognizer.ctc.frames import FrameConfig, audio_to_frames
from morse_char_recognizer.ctc.model import CrnnCtcConfig, CrnnCtcRecognizer
from morse_char_recognizer.ctc.vocab import CtcVocab


@dataclass
class LoadedCtcModel:
    """Bundle of a loaded CTC checkpoint."""

    model: CrnnCtcRecognizer
    vocab: CtcVocab
    frame_config: FrameConfig
    args: dict[str, Any]


def load_ctc_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedCtcModel:
    """Load a checkpoint produced by ``scripts/train_ctc.py``."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    vocab = CtcVocab(tuple(checkpoint["vocab_chars"]))
    model_config = CrnnCtcConfig(**checkpoint["model_config"])
    model = CrnnCtcRecognizer(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    frame_config = FrameConfig(**checkpoint["frame_config"])
    return LoadedCtcModel(
        model=model,
        vocab=vocab,
        frame_config=frame_config,
        args=dict(checkpoint.get("args", {})),
    )


@torch.no_grad()
def frame_log_probs(
    loaded: LoadedCtcModel,
    audio: np.ndarray,
    sample_rate: int,
    *,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Run the encoder on one audio file and return ``(time, num_classes)`` log probs."""
    frames = audio_to_frames(audio, sample_rate, loaded.frame_config)
    if frames.size == 0:
        return np.zeros((0, loaded.vocab.num_classes), dtype=np.float32)
    inputs = torch.from_numpy(frames).unsqueeze(0).unsqueeze(0).to(device)
    log_probs = loaded.model(inputs)  # [T, B=1, V]
    return log_probs.squeeze(1).cpu().numpy()


def decode_audio(
    loaded: LoadedCtcModel,
    audio: np.ndarray,
    sample_rate: int,
    *,
    device: torch.device | str = "cpu",
) -> str:
    """Decode a single continuous audio buffer with greedy CTC."""
    log_probs = frame_log_probs(loaded, audio, sample_rate, device=device)
    if log_probs.size == 0:
        return ""
    return greedy_decode_numpy(log_probs, loaded.vocab)
