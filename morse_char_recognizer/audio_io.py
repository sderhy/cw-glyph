"""Audio file loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile


def read_wav_mono(path: str | Path) -> tuple[int, np.ndarray]:
    """Read a WAV file as normalized float32 mono audio."""
    sample_rate, audio = wavfile.read(path)
    x = np.asarray(audio)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    if np.issubdtype(x.dtype, np.integer):
        peak = float(np.iinfo(x.dtype).max)
        x = x.astype(np.float32) / peak
    else:
        x = x.astype(np.float32)
    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    if max_abs > 0:
        x = x / max_abs
    return int(sample_rate), x.astype(np.float32, copy=False)
