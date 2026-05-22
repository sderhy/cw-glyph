# CW Glyph

Experimental glyph-level CW/Morse symbol recognizer.

The goal of CW Glyph is to understand and recognize the Morse symbol itself.
It treats each Morse character, punctuation mark, prosign, or run-on rhythm as
an audio glyph: a compact temporal shape that can be segmented, normalized, and
classified.

This is deliberately inspired by handwritten-character recognition:

```text
handwritten recognition:
  image of one written symbol
  -> normalization
  -> classifier
  -> symbol

CW Glyph:
  audio segment of one Morse symbol
  -> envelope glyph normalization
  -> classifier
  -> symbol
```

The continuous-audio pipeline exists to produce candidate glyphs, but the core
object of study is still the character-level Morse symbol:

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

## Project Scope

CW Glyph is:

- a glyph engine for Morse audio;
- a Morse symbol classifier;
- a research tool for comparing Morse glyph representations;
- a way to evaluate character-by-character recognition before attempting a
  complete conversational decoder.

CW Glyph is not currently:

- a full HF receiver;
- a production-grade live CW decoder;
- a language model for reconstructing operator intent;
- a complete QSO parser;
- a replacement for end-to-end decoders trained on full audio transcripts.

This character-first scope is intentional. Segmenting and recognizing symbols
one at a time makes the errors inspectable: `E/T`, `Q/Y`, `? / 2`, timing
mistakes, prosign collisions, and operator-specific rhythm all remain visible
instead of being hidden inside an end-to-end text decoder.

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
uv sync --extra dev
uv run pytest
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
  --min-segment-units 1.4 \
  --max-character-units 19 \
  --word-gap-mode unit \
  --word-gap-units 4.5 \
  --window 30 \
  --hop 30 \
  --json-out outputs/livetest_eval.json
```

The headline metric is full character error rate (CER) over the decoded file,
including substitutions, insertions, and deletions. The older best-substring
mode is available only as a diagnostic for partial windows. JSON output includes
the command, git commit, checkpoint arguments, per-window predictions, and
per-file CER breakdown so runs can be reproduced and compared.

For short labelled fragments, prefer a full-file pass (`--duration 0`) over
fixed windows so artificial window boundaries do not create extra segmentation
errors.

## Evaluate Segmentation Alone

Synthetic segmentation tests report missed, split, merged, and false-positive
character regions independently from CNN classification:

```bash
python scripts/eval_segmentation_synth.py \
  --text "CQ TEST 599" \
  --wpm 15 20 25 \
  --seeds 0 1 2 \
  --snr-db 12 \
  --qsb-rate-hz 0.25 \
  --qsb-depth-db 12 \
  --json-out outputs/segmentation_eval.json
```

Segmentation supports two threshold modes:

- `fixed`: global envelope threshold, currently still the baseline to beat;
- `adaptive`: local percentile threshold for recordings with changing level or
  QSB. It is experimental and must be selected explicitly with
  `--threshold-mode adaptive`.

## Current Results

CW Glyph reports full character error rate (CER) with substitutions,
insertions, and deletions. Segmentation can also be evaluated independently on
synthetic references.

Current local measurements:

```text
clean synthetic glyph validation: >= 0.99 accuracy
noisy synthetic glyph validation (5-25 dB SNR): >= 0.99 accuracy
synthetic segmentation smoke, fixed threshold with 12 dB SNR + 12 dB QSB:
  precision 1.000, recall 1.000
synthetic segmentation smoke, adaptive threshold on the same case:
  precision 0.988, recall 0.975
g3ses/C1.wav with stretch checkpoint: CER 0.000, decoded "R QRL?"
g3ses/C1.wav with unit32 v2 checkpoint: CER 0.000, decoded "R QRL?"
g6pz/G1.wav with unit32 v2 checkpoint, full-file pass, short-segment filter,
  and overlong-segment split: CER 0.089
```

These numbers are development measurements from local labelled audio under
`livetests/`, which is not published in this repository.

The real-audio target is met on one short labelled QSO fragment (`g6pz/G1.wav`)
but is not proven across the corpus yet. The next milestone is to reproduce
CER <= 0.05-0.10 on more labelled recordings, with segmentation quality
reported separately.

## Roadmap

- Improve real-audio segmentation with local WPM/dot-unit estimation, overlong
  candidate splitting, and better word-gap estimation. Adaptive thresholding
  exists as an explicit experimental mode, but the fixed threshold remains the
  measured baseline.
- Extend segmentation-specific metrics from synthetic references to labelled
  real-audio boundaries.
- Preserve the glyph-first design while improving continuous-audio front-end
  quality.
- Calibrate or replace naive multi-checkpoint ensembling. Raw CNN confidence
  scores are not comparable across differently normalized models.
- Expand and document labelled livetest evaluation, including alignment reports
  that identify weak segments and repeated confusions.
- Add a documented checkpoint release path. Model files are generated artifacts
  and are not committed to the source tree.
- Investigate a 2D spectrogram/image baseline for comparison with the current
  1D envelope glyph approach.

## Data And Artifacts

`livetests/` and `outputs/` are intentionally ignored. Real recordings,
generated checkpoints, and preview images should be managed separately from the
source repository unless their license and size are appropriate for publication.

## License

Apache-2.0.
