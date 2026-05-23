"""Sequence-level CW recognition with CTC.

This subpackage holds a sequence-aware alternative to the per-character
classifier pipeline. The character glyph CNN treats each audio segment as an
independent EMNIST-like image and depends on a separate envelope segmenter to
find character boundaries. The CTC recogniser instead consumes a single
contiguous envelope stream and learns to align labels with frames internally.

The pieces are intentionally small:

- ``vocab``: a fixed ``A-Z 0-9 <space>`` vocabulary with a CTC ``<blank>``.
- ``frames``: deterministic audio → fixed-rate envelope-frame conversion.
- ``dataset``: synthetic multi-character sequence dataset.
- ``model``: a compact CRNN (1D conv frontend + bidirectional GRU + linear
  head over ``len(vocab) + 1``).
- ``decode``: greedy CTC decode and prefix beam search with optional LM.
- ``lm``: a character n-gram language model trained from a small corpus.
- ``inference``: helpers that load a checkpoint and decode a continuous WAV.
"""

__all__: list[str] = []
