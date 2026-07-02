# Decoding HOWTO

This document lists the practical decoding commands for CW Glyph.

Current recommendation: use the checkpoint-free **beam/timing** decoder first.
Use the CNN commands only for comparison or older glyph-classifier experiments.

## Setup

From the repository root:

```bash
uv sync --extra dev
```

Run commands through `uv run` so they use the project environment:

```bash
uv run python scripts/decode_beam.py --help
```

## 1. Single WAV, Beam/Timing Decoder

This is the main decoder today. It needs no model checkpoint.

```bash
uv run python scripts/decode_beam.py path/to/audio.wav --auto-center
```

With a sibling transcript `path/to/audio.txt`, it also prints CER:

```bash
uv run python scripts/decode_beam.py livetests/g3ses/C1.wav --auto-center
```

Typical output includes:

```text
center_hz=592.2 elements=21 unit=3022smp (68.5ms)
decoded: R QRL?
reference: R QRL?
CER=0.000 ...
```

Useful options:

```bash
--auto-center                 estimate the dominant tone frequency
--bandpass-center-hz 700      force a known tone frequency
--bandpass-width-hz 100       narrow receiver bandpass used before envelope
--threshold 0.22              high threshold for keydown detection
--hysteresis-low 0.12         low threshold for hysteresis body recovery
--dot-dash-split 1.8          dah threshold in dot units
--allowed-class-preset NAME   restrict output vocabulary, e.g. ham-copy
--json-out outputs/file.json  save decoded text and metrics
```

## 2. Single WAV With PNG Spectrogram

For visual inspection, add `--preview-out`. The PNG is a grayscale spectrogram:
time on x, frequency on y, signal intensity in gray, and center frequency in
blue.

```bash
uv run python scripts/decode_beam.py livetests/g6pz/G12.wav \
  --auto-center \
  --preview-out outputs/beam_previews/G12.png \
  --preview-min-hz 400 \
  --preview-max-hz 900
```

Use this whenever a decoded line looks suspicious. It gives a quick view of
whether the recording is centered, drifting, noisy, or multi-station.

## 3. WebSDR Benchmark With Carrier Tracking

Use this for `livetests/websdr/*.wav`. It is the best path for WebSDR because
it estimates a local carrier track and splits files when the dominant station
frequency changes.

```bash
uv run python scripts/eval_websdr_beam.py livetests/websdr
```

Write JSON:

```bash
uv run python scripts/eval_websdr_beam.py livetests/websdr \
  --json-out outputs/websdr_beam_eval.json
```

Write grayscale previews:

```bash
uv run python scripts/eval_websdr_beam.py livetests/websdr \
  --preview-dir outputs/websdr_previews_gray
```

Compare against one global carrier estimate:

```bash
uv run python scripts/eval_websdr_beam.py livetests/websdr --no-track
```

Default WebSDR behavior:

- uses `--allowed-class-preset ham-copy`;
- tracks the carrier in short windows;
- emits per-file CER, decoded text, element count, unit estimate, centers, and
  top alignment errors.

The `ham-copy` profile allows:

```text
A-Z 0-9 / . , ? = + <SK> <SN>
```

It intentionally excludes noisy/polluting glyphs such as `É`, `<UR>`, `:`, and
`&`.

## 4. WebSDR Single File, Manual Beam Decode

For quick manual checks on a WebSDR file:

```bash
uv run python scripts/decode_beam.py \
  livetests/websdr/websdr_recording_start_2026-07-01T08_10_36Z_14011.9kHz.wav \
  --auto-center \
  --bandpass-width-hz 70 \
  --allowed-class-preset ham-copy
```

If the spectrogram shows two frequency plateaus, prefer
`scripts/eval_websdr_beam.py`; `decode_beam.py` uses only one center frequency
per invocation.

## 5. Batch Check A Labelled Directory

There is not yet a dedicated generic beam batch script. For now, use this small
inline runner to evaluate every sibling-labelled WAV in a directory:

```bash
uv run python - <<'PY'
from pathlib import Path
from morse_char_recognizer.audio_io import read_wav_mono
from morse_char_recognizer.beam_decode import BeamConfig, beam_decode_audio
from morse_char_recognizer.live import compare_text, default_label_path, estimate_carrier_hz, read_label
from morse_char_recognizer.segment import SegmentConfig

root = Path("livetests/g3ses")
rows = []
for wav in sorted(root.glob("*.wav")):
    label = default_label_path(wav)
    if label is None:
        print(f"{wav.name}: skip, missing .txt")
        continue
    sr, audio = read_wav_mono(wav)
    center = estimate_carrier_hz(audio, sr, 400.0, 900.0)
    segment_config = SegmentConfig(
        threshold_mode="hysteresis",
        threshold=0.22,
        hysteresis_low=0.12,
        bandpass_center_hz=center,
        bandpass_width_hz=100.0,
        min_keydown_ms=10.0,
        merge_gap_ms=8.0,
        min_unit_ms=35.0,
        max_unit_ms=140.0,
    )
    decoded, regions, unit = beam_decode_audio(
        audio,
        sr,
        segment_config=segment_config,
        beam_config=BeamConfig(dot_dash_split=1.8),
    )
    cmp = compare_text(read_label(label), decoded)
    rows.append((wav.name, cmp.cer, cmp.distance, cmp.reference_length))
    print(f"{wav.name:12} CER={cmp.cer:.3f} dist={cmp.distance}/{cmp.reference_length}")

dist = sum(row[2] for row in rows)
ref = sum(row[3] for row in rows)
print(f"global CER={dist / ref:.3f} dist={dist}/{ref}")
PY
```

Change `root = Path("livetests/g3ses")` to `livetests/g6pz`,
`livetests/FAV22`, or another labelled corpus.

## 6. CNN Single-WAV Decode

This is the older glyph-classifier path. It needs a trained checkpoint.

```bash
uv run python scripts/decode_wav.py \
  outputs/glyph52_real_unit32_cnn_snr5_25.pt \
  path/to/audio.wav \
  --auto-center \
  --allowed-class-preset real \
  --verbose
```

Use this mainly for comparison. Current evidence says the beam/timing decoder is
stronger on noisy real recordings.

## 7. CNN Labelled Livetest Evaluation

For labelled real-audio evaluation with a CNN checkpoint:

```bash
uv run python scripts/eval_livetest.py \
  outputs/glyph52_real_unit32_cnn_snr5_25.pt \
  livetests/g3ses \
  --auto-center \
  --allowed-class-preset copy \
  --threshold 0.18 \
  --char-gap-units 2.3 \
  --min-segment-units 1.4 \
  --max-character-units 19 \
  --word-gap-mode unit \
  --word-gap-units 4.5 \
  --duration 0 \
  --json-out outputs/livetest_eval.json
```

For short labelled fragments, prefer `--duration 0` so fixed windows do not add
boundary artifacts.

## 8. CNN Visual Preview

For the CNN segmentation/classification path:

```bash
uv run python scripts/preview_wav_decode.py \
  outputs/glyph52_real_unit32_cnn_snr5_25.pt \
  path/to/audio.wav \
  --auto-center \
  --duration 30 \
  --out outputs/preview.png
```

This preview includes waveform/envelope/segments/CNN labels. For beam/timing
debugging, prefer `decode_beam.py --preview-out`.

## 9. Which Command Should I Use?

Use this decision table:

```text
One normal WAV, no checkpoint:      decode_beam.py --auto-center
One WAV plus visual inspection:     decode_beam.py --preview-out ...
WebSDR corpus:                      eval_websdr_beam.py livetests/websdr
WebSDR visuals:                     eval_websdr_beam.py --preview-dir ...
Old CNN checkpoint comparison:      decode_wav.py checkpoint.pt wav
Old CNN labelled evaluation:        eval_livetest.py checkpoint.pt corpus
Old CNN visual segmentation view:   preview_wav_decode.py checkpoint.pt wav
```

## 10. Current Corpus Baselines

Recent local beam/timing measurements:

```text
FAV22:  CER 0.009
g3ses:  CER 0.120
g6pz:   CER 0.244
WebSDR: CER 0.263 with carrier tracking
```

Interpretation:

- FAV22 is almost solved and is an anti-regression set.
- g3ses is reasonably clean and useful for anti-regression.
- g6pz is difficult; use it as a stress test, not as the main tuning truth.
- WebSDR requires carrier tracking; a single global center is not enough.

## 11. Recommended Next Data

For progress, collect new labelled samples:

- 20-60 seconds each;
- one dominant station;
- transcript checked while listening;
- keep WAV/TXT sibling files;
- include a few difficult cases only after the easy baseline is stable.

Good new samples will move the project faster than more CNN training right now.
