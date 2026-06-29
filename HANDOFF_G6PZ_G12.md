# CW Glyph handoff - G6PZ/G12 rhythm failure

Date: 2026-06-29

## Current context

Project: `/home/user/projects/cw-glyph`

Heavy work must run on RunPod, not locally. The local Ormind container has
limited memory and no GPU. RunPod SSH alias is configured as:

```text
runpod-cw-glyph
```

Expected remote repo path:

```text
/workspace/cw-glyph
```

The original GPU pod stopped being available, so the project was migrated to a
new RunPod GPU instance.

## RunPod SSH state

New GPU pod provided by Serge and validated:

```text
ssh b42j3q78ifjm6v-64411299@ssh.runpod.io -i ~/.ssh/id_ed25519
ssh root@194.68.245.53 -p 22113 -i ~/.ssh/id_ed25519
```

New pod validation:

```text
host: 20fe29b3da14
gpu: NVIDIA A40, 46068 MiB
repo: /workspace/cw-glyph
venv: /workspace/cw-glyph/.venv
pytest: 67 passed
```

The project, local `livetests`, and recovered old `outputs` have been migrated
to the new pod.

Old CPU-only instance TCP endpoint provided by Serge:

```text
ssh root@203.57.40.168 -p 10014 -i ~/.ssh/id_ed25519
```

Local SSH aliases:

```text
runpod-cw-glyph       -> new direct TCP endpoint
runpod-cw-glyph-proxy -> new RunPod proxy endpoint
runpod-cw-glyph-old   -> old CPU-only direct TCP endpoint
```

Old CPU-only TCP access was fixed by adding `~/.ssh/id_ed25519.pub` to
`/root/.ssh/authorized_keys` from the RunPod web terminal.

Recovered old RunPod outputs locally:

```text
recovered/runpod-old-20260629/outputs
recovered/runpod-old-20260629/MANIFEST.md
```

The recovered data includes the reference checkpoint, experiment logs, JSON
reports, and G12 preview images. Remote `livetests` were not copied because the
local `livetests` directory is at least as complete; local has extra
`faits/R13-*` files that were absent from the old RunPod instance.

## Known baseline

Reference experiment on RunPod:

```text
/workspace/cw-glyph/outputs/experiments/20260625T200509Z-reference
```

Checkpoint:

```text
checkpoint.pt
```

Baseline memory:

- synthetic classification is effectively solved at normal SNRs;
- SNR 25/15/10/5 dB: accuracy 1.000;
- SNR 0 dB: accuracy 0.996;
- synthetic segmentation precision/recall: 1.000.

So the current weak point is not generic synthetic glyph classification.

## Important artefacts

Local labelled sample:

```text
livetests/G6PZ/G12.wav
livetests/G6PZ/G12.txt
```

Preview artefacts:

```text
/home/user/tmp/cw-glyph-previews/G6PZ_G12_bad.png
/home/user/tmp/cw-glyph-previews/G6PZ_G12_centered_000_030.png
/home/user/tmp/cw-glyph-previews/G6PZ_G12_centered_060_090.png
/home/user/tmp/cw-glyph-previews/G6PZ_G12_bw100_th018_msu22_000_030.png
/home/user/tmp/cw-glyph-previews/G6PZ_G12_bw100_th018_msu22_060_090.png
/home/user/tmp/cw-glyph-previews/G6PZ_G12_rhythm_060_090_small.jpg
```

Browser URL for the main failed preview:

```text
https://serge.v2.ormind.ai/tmp/cw-glyph-previews/G6PZ_G12_bad.png
```

## Key observation from Serge

Looking at `G6PZ_G12_bad.png`, the Morse is readable by eye. The beginning is:

```text
BK HI ...
```

In Morse rhythm terms:

```text
B = dah di di di
K = dah di dah
```

The model/decoder reads this badly even though the visual rhythm is clear. This
strongly suggests a scaling/rhythm/segmentation failure rather than an
unreadable signal.

## Working diagnosis

The failure mode is probably caused by rhythm mismatch:

- dot/dash durations are not stable enough for one global unit estimate;
- operator rhythm differs from the synthetic training distribution;
- character grouping uses thresholds that are too rigid;
- glyph normalization may scale real rhythm in a way the CNN did not see during
  training;
- the decoder emits many insertions and short symbols, which is consistent with
  over-splitting or bad unit estimation.

Do not start by retraining blindly. First separate:

1. active keydown detection;
2. local dot-unit estimation;
3. character segmentation/grouping;
4. glyph normalization;
5. CNN classification on correctly segmented glyphs.

## Hypotheses to test

### 1. Wider WPM training range

Current synthetic training may not expose the model to enough timing variation.
Test synthetic generation from roughly:

```text
5 WPM to 50 WPM
```

This should include slow, normal, and fast operators.

Status 2026-06-29: tested on the new A40 pod with an inhomogeneous synthetic
distribution covering 5-60 WPM, wider timing jitter, dash/dot ratio variation,
gap inflation, rise/fall variation, stroke amplitude jitter, SNR, QSB, and RX
filtering.

The analog inhomogeneous checkpoint:

```text
/workspace/cw-glyph/outputs/experiments/20260629T070149Z-rhythm-analog
```

Synthetic accuracy:

```text
SNR 25 dB: 0.997
SNR 15 dB: 0.996
SNR 10 dB: 0.993
SNR  5 dB: 0.980
SNR  0 dB: 0.842
```

G12 with the old segmentation starts correctly with `BK HI SOUNDS...`, but CER
is still high (`0.877`) due to many insertions.

### 2. Random intra-sample rhythm variation

Training should include rhythm perturbations, not only fixed WPM per sample:

- dot length jitter;
- dash length jitter;
- intra-character gap jitter;
- inter-character gap jitter;
- gradual WPM drift inside a phrase;
- operator-specific "swing" or hand-sent irregularity.

The goal is not just noisy audio, but noisy timing.

Implemented 2026-06-29:

- per-keydown stroke amplitude jitter in `morse_synth.keying.KeyingConfig`;
- train/eval CLI ranges for dash/dot ratio, gap inflation, rise/fall time, and
  stroke amplitude jitter;
- optional envelope binarization threshold for explicit ablation.

Binary ablation:

```text
/workspace/cw-glyph/outputs/experiments/20260629T070435Z-rhythm-binary035
```

It slightly improves G12 CER in this one setup (`0.864`) but hurts synthetic
classification badly versus analog (`best validation 0.949` versus `0.993`).
Conclusion: keep analog inhomogeneous training as the main path; use binary
only as an ablation.

### 3. Local scaling instead of global scaling

For real audio, estimate dot-unit locally by window or by active region cluster.
Avoid one global dot-unit for all of G12.

Candidate diagnostics:

- histogram of keydown durations per 15-30 s window;
- histogram of gaps per 15-30 s window;
- overlay predicted dot-unit on the envelope;
- compare expected `BK HI` boundaries against detected boundaries.

### 4. Evaluate segmentation before CNN

For G12, manually inspect the first phrase (`BK HI ...`) and verify whether
detected character segments correspond to visible Morse glyphs. If segmentation
is wrong there, CNN results are secondary.

## Local code state to remember

There are already local uncommitted changes aimed at this direction:

- configurable bandpass center/width in segmentation;
- noise-floor percentile passed through segmentation config;
- more robust `estimate_unit_samples(...)` with plausible min/max unit bounds
  and histogram binning;
- CLI options for `--min-unit-ms`, `--max-unit-ms`, `--bandpass-width-hz`;
- tests for configured bandpass center and ignoring implausibly short fragments;
- `scripts/run_reference_experiment.sh` for reproducible smoke/reference runs.

These changes have not been validated locally because the local environment is
not meant for the project dependencies. Validate on RunPod when a GPU pod is
available.

## Next practical program

When RunPod is available again:

1. Sync or pull the current repo state to `/workspace/cw-glyph`.
2. Validate the current test suite on RunPod.
3. Recreate the G12 previews with the current checkpoint and current local
   segmentation changes.
4. Generate a specific rhythm report for the first 30 seconds:
   - envelope;
   - active regions;
   - keydown duration histogram;
   - gap histogram;
   - estimated local dot-unit;
   - detected segments over `BK HI`.
5. If `BK HI` is segmented incorrectly, fix segmentation/local unit estimation
   before retraining.
6. If `BK HI` is segmented correctly but classified badly, then train a timing
   augmented checkpoint with 5-50 WPM and random rhythm perturbations.

## Rule of thumb

If a human can read the spectrogram/envelope but CW Glyph cannot, treat it as a
representation and timing-normalization problem first, not as a lack of model
capacity.

## G12 short `E` diagnosis

Serge noticed that in `GREAT ES LIKE`, some visually clear `E` glyphs were
swallowed by the decoder.

Diagnosis 2026-06-29:

- not primarily a sampling issue;
- estimated unit on the G12 preview is about `57.3 ms`;
- previous `--min-segment-units 2.2` rejects segments shorter than about
  `126 ms`;
- a real `E` is roughly one unit, so the anti-noise filter can remove it.

Sweeps showed:

- lowering `min_segment_units` to `1.2-1.4` recovers `E`, `GREAT`, `LIKE`;
- but it also reintroduces many false short segments and insertions;
- increasing `char_gap_units` from `2.3` to about `2.9-3.5` helps reduce splits
  such as `L` becoming `E + D`.

Best provisional visual preview:

```text
/home/user/tmp/cw-glyph-previews/G6PZ_G12_analog_msu14_cgu29_000_030.png
```

It decodes:

```text
BK HI SOU/S GREAT H LIKE TOSE ...
```

This is better on `GREAT` and `LIKE`, but still has insertion/split errors.

Implemented code support:

- score-aware short segment filter that can preserve short `E,T` candidates;
- tests for that filter.

Experiment result: score alone is not enough because false short `E/T`
candidates also get high CNN confidence. The next real fix should be
segmentation/rhythm-based: local gap modeling and intra-character gap tolerance,
not just classifier confidence.
