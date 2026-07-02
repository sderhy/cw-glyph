# Session handoff - 2026-07-02 (WebSDR front-end, carrier tracking, visual previews)

Follows [`HANDOFF_SESSION_2026-07-01.md`](./HANDOFF_SESSION_2026-07-01.md).

## Headline

The next gain was not CNN training. It was WebSDR front-end observability:

- add an explicit WebSDR benchmark for the pure beam/timing path;
- restrict noisy WebSDR copy to a `ham-copy` vocabulary when requested;
- track local carrier frequency instead of relying on one global center;
- produce grayscale spectrogram PNGs for future beam decodes.

The main result is that the long WebSDR file is no longer decoded against the
wrong half-file carrier. Its deletion count collapses when the recording is
split into two detected frequency plateaus.

## What landed

Two commits were pushed to `origin/main`:

1. **`f64c95a`** `Add WebSDR beam benchmark and carrier tracking`
   - `scripts/eval_websdr_beam.py`: reproducible WebSDR benchmark.
   - `morse_char_recognizer/live.py`: short-window carrier track,
     stable/unstable decision, piecewise carrier segments.
   - `morse_char_recognizer/classes.py`: opt-in `ham-copy` preset.
   - `morse_char_recognizer/beam_decode.py`: optional
     `BeamConfig.allowed_tokens`.
   - `scripts/decode_beam.py`: optional `--allowed-class-preset`.

2. **`20025af`** `Add grayscale spectrogram previews for beam decoding`
   - `morse_char_recognizer/preview.py`: shared grayscale spectrogram renderer.
   - `scripts/decode_beam.py --preview-out`: PNG for any beam decode.
   - `scripts/eval_websdr_beam.py --preview-dir`: same renderer for WebSDR.

Current pushed code head before this handoff file: **`20025af`**.

## Current measurements

All measurements below use the pure beam/timing path, no CNN.

### WebSDR

Command:

```bash
uv run python scripts/eval_websdr_beam.py livetests/websdr
```

With the local correction to the first WebSDR transcript (`... DX K E E`):

```text
summary CER=0.263 dist=41/156 subs=24 ins=8 del=9

14011.9 kHz: CER=0.150
  decoded:   R6ANK F6ARC F6ARC T X K E E
  reference: F6ANC F6ARC F6ARC DX K E E

14017.0 kHz: CER=0.279
  centers: 728.2 Hz, 620.5 Hz
```

Important comparison:

```text
WebSDR without carrier tracking: CER=0.623
WebSDR with carrier tracking:    CER=0.263
```

On the long 14017.0 kHz file, tracking reduces deletions from `72` to `9`.
That confirms the frequency-variation bug was real and fixed.

### g6pz

Command shape:

```bash
uv run python scripts/decode_beam.py livetests/g6pz/G12.wav --auto-center
```

Batch result over sibling-labelled WAV files:

```text
12 files, 1 skipped (G10.wav has no G10.txt)
global CER=0.244 dist=289/1183
mean CER=0.226 median CER=0.235
subs=186 ins=85 del=18
```

Per file:

```text
G1  0.122
G2  0.292
G3  0.000
G4  0.212
G5  0.182
G6  0.145
G7  0.213
G8  0.257
G9  0.357
G11 0.313
G12 0.274
G14 0.344
```

Interpretation: `g6pz` is a useful stress test but too hard and too ambiguous
for fine parameter tuning. It should not be the primary truth source for the
next iteration.

### g3ses

```text
14 files, 1 skipped (C1O.wav has no C1O.txt)
global CER=0.120 dist=158/1321
mean CER=0.126 median CER=0.115
subs=91 ins=22 del=45
```

Per file:

```text
C1  0.000
C2  0.174
C3  0.143
C4  0.197
C5  0.216
C6  0.110
C7  0.101
C8  0.080
C9  0.120
C10 0.197
C11 0.134
C12 0.099
C13 0.090
C14 0.109
```

Interpretation: `g3ses` is much cleaner than `g6pz`, and should be used mainly
as an anti-regression set.

### FAV22

```text
FAV22-0515.wav: CER=0.009 dist=24/2652
subs=16 ins=7 del=1
center=689.5 Hz unit=58.4 ms elements=9153
```

Interpretation: FAV22 is essentially solved by the current beam/timing path and
is best treated as an anti-regression test.

## Data caveat

`livetests/` is ignored by Git. The local WebSDR transcript correction:

```text
F6ANC F6ARC F6ARC DX K E E
```

is present locally but was not committed or pushed. The previous version lacked
the final `E E`. Keep this in mind when reproducing WebSDR numbers elsewhere.

There is still uncertainty in the first WebSDR transcript:

- decoded says `R6ANK ... T X K E E`;
- reference says `F6ANC ... DX K E E`;
- remaining errors are `F->R`, `C->K`, `D->T`.

Do not overfit code to that file until the human transcript is rechecked.

## Useful commands

WebSDR benchmark with JSON:

```bash
uv run python scripts/eval_websdr_beam.py \
  livetests/websdr \
  --json-out outputs/websdr_beam_eval.json
```

WebSDR benchmark with grayscale previews:

```bash
uv run python scripts/eval_websdr_beam.py \
  livetests/websdr \
  --preview-dir outputs/websdr_previews_gray
```

Single beam decode with grayscale spectrogram:

```bash
uv run python scripts/decode_beam.py livetests/g6pz/G12.wav \
  --auto-center \
  --preview-out outputs/beam_previews/G12.png \
  --preview-min-hz 400 \
  --preview-max-hz 900
```

Single WebSDR decode with restricted copy vocabulary:

```bash
uv run python scripts/decode_beam.py \
  livetests/websdr/websdr_recording_start_2026-07-01T08_10_36Z_14011.9kHz.wav \
  --auto-center \
  --bandpass-width-hz 70 \
  --allowed-class-preset ham-copy
```

## Validation run

Before `20025af` was pushed:

```text
uv run pytest -> 94 passed
ruff on modified files -> passed
```

The full repo still has pre-existing lint issues in untouched files if running
`ruff check morse_char_recognizer scripts tests` globally.

## Recommended next step

Collect new labelled samples. `g6pz` is too difficult and transcription is too
ambiguous for the next fine-grained iteration.

Best next corpus shape:

- 20-60 second excerpts;
- one dominant station;
- no long QRM/QSB fades unless intentionally testing them;
- transcript checked immediately while listening;
- keep raw WAV + TXT sibling files under `livetests/<source>/`.

Then add an error-focused report:

```text
scripts/report_beam_errors.py
```

It should create one small PNG per alignment error, with:

- local grayscale spectrogram;
- envelope and detected keydowns;
- dit/dah durations in units;
- chosen Morse pattern;
- reference token vs decoded token.

That will tell whether each remaining error is transcript uncertainty, missed
dit, merged element, wrong dah/dit split, or bad DP grouping.

## CNN status

Keep the CNN parked. Current evidence still says:

- LCWO/synthetic is solved;
- CNN does not fix missed/noisy elements;
- the next improvement should come from real front-end diagnostics and
  segmentation, not synthetic retraining.
