# CW Glyph

Experimental character-level CW/Morse recognizer.

The project treats isolated Morse characters as audio glyphs, by analogy with
handwritten-character recognition:

```text
continuous CW audio
  -> envelope segmentation
  -> normalized 1D character glyphs
  -> 1D CNN classifier
  -> decoded text
```

The current recognizer includes synthetic data generation, real-WAV preview
tools, labelled livetest evaluation, and CNN models trained on 52 Morse glyph
classes.

## Highlights

- Parametric CW synthesis and HF channel simulation in `morse_synth/`.
- Character vocabularies:
  - `basic`: `A-Z0-9`;
  - `copy`: letters, digits, common punctuation;
  - `real`: copy classes plus selected prosigns/run-on glyphs.
- Envelope extraction with optional carrier bandpass and noise-floor removal.
- Time normalization by estimated Morse dot-unit, similar to scaling a
  handwritten character before classification.
- CNN inference on continuous WAV files, including labelled real-audio
  evaluation.

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
pytest
```

## Train

Train the current real-vocabulary unit-scaled CNN:

```bash
python scripts/train_cnn.py \
  --class-preset real \
  --scale-mode unit \
  --canonical-units 32 \
  --samples-per-class 45 \
  --val-samples-per-class 12 \
  --epochs 8 \
  --snr-min 5 \
  --snr-max 25 \
  --out outputs/glyph52_real_unit32_cnn_snr5_25.pt
```

## Decode A WAV

```bash
python scripts/decode_wav.py \
  outputs/glyph52_real_unit32_cnn_snr5_25.pt \
  path/to/audio.wav \
  --allowed-class-preset real \
  --verbose
```

## Preview Real Audio

```bash
python scripts/preview_wav_decode.py \
  outputs/glyph52_real_unit32_cnn_snr5_25.pt \
  path/to/audio.wav \
  --auto-center \
  --duration 30 \
  --out outputs/preview.png
```

## Evaluate Labelled Live Tests

For clean grouped-copy training audio, constrain outputs to copy characters:

```bash
python scripts/eval_livetest.py \
  outputs/glyph52_real_unit32_cnn_snr5_25.pt \
  path/to/audio.wav \
  --auto-center \
  --allowed-class-preset copy \
  --threshold 0.18 \
  --char-gap-units 2.3 \
  --window 30 \
  --hop 30
```

## Data And Artifacts

`livetests/` and `outputs/` are intentionally ignored. Real recordings,
generated checkpoints, and preview images should be managed separately from the
source repository unless their license and size are appropriate for publication.

## License

Apache-2.0.
