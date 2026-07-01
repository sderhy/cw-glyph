# Session handoff — 2026-07-01 (CNN-in-beam → detection is the real lever)

Follows [`HANDOFF_SESSION_2026-06-30.md`](./HANDOFF_SESSION_2026-06-30.md).

## Headline

Set out to fold the glyph CNN into the element-level beam decoder; ended up
**improving the decoder by fixing element detection instead**. Three CNN
iterations all failed to beat pure timing (the synthetic→real domain gap is the
wall), but the forced-alignment tooling built along the way revealed that the
real CER bottleneck is **missed elements in detection** — fixed with a
dual-threshold **hysteresis** detector.

Net decoder result (pure timing, no CNN):
- **g6pz (G\*): mean CER 0.271 → 0.226**, median 0.298 → 0.235 (9/12 better, 0 worse)
- **g3ses+QSOs (held out): mean 0.172 → 0.168**, median 0.126 → 0.115
- combined 34 files ~0.207 → 0.188 (~9% rel), no net regression.

## What landed (commits 909fc4a → f732785, 91 tests pass)

1. **`909fc4a`** CNN glyph scoring in the beam: `beam_decode_regions` gains a
   `span_score(j,i,char)` hook; `cnn_beam.py` + `scripts/decode_cnn_beam.py`.
   Finding: naive rescore is neutral/harmful (merge bias).
2. **`f436ddf`** Segmentation-aware CNN with a `<JUNK>` merge-rejection class:
   `span_dataset.py` (exact element→char alignment from `build_events`),
   `scripts/train_cnn_spans.py`. Kills the merge bias in-distribution but
   doesn't transfer to real audio.
3. **`440d737`** Real-audio pipeline: `real_align.py` (Needleman–Wunsch forced
   alignment), `scripts/harvest_real_spans.py`, `scripts/finetune_cnn_real.py`.
4. **`f732785`** ← the win. **Hysteresis element detection** in `segment.py`.

## CNN verdict (why it's parked)

Fine-tuned real model `outputs/glyph_cnn_real.pt` (from
`glyph_cnn_spans_v1.pt`) is now well-calibrated on real glyphs (held-out real
val: real_acc 0.993, junk_leak 0.017) and **safe** to fold in at `w_cnn≈0.05`
(keeps G3 perfect at all weights, rejects merges), but it does **not** lower CER
on held-out g6pz (0.267→0.266). Reason: the harvested val only contains cleanly
aligned spans — the easy cases timing already gets right. The residual errors
are hard cases (noisy detection) the CNN can't touch. See memory
`cnn-in-beam-domain-gap`.

## Detection fix (the win)

Forced-alignment error analysis on g6pz: element-level **deletions 4.6%**
(missed elements) dominate, then dah→dit subs 3.8% (`dot_dash_split=2.0` too
high — real dahs run light), insertions 2.4%. A single fixed threshold can't win
(lowering 0.18 helps noisy files, hurts clean ones). **Hysteresis** keeps an
element body wherever the envelope exceeds `hysteresis_low` (0.12) but only for
runs that also reach `threshold` (0.22). New defaults:
`SegmentConfig.threshold_mode="hysteresis"` (0.22/0.12),
`BeamConfig.dot_dash_split=1.8`. See memory `detection-hysteresis-win`.

## State

- Local `main` = `f732785` (+ this handoff), pushed to `origin` (GitHub).
- Pod (`root@194.68.245.211`, port changes on migration) was synced to
  `f436ddf`; it went unreachable before the last two commits could be pushed —
  **re-sync pod to `f732785`** next session (see memory `runpod-pod-access`).
- Models on disk (gitignored): `outputs/glyph_cnn_spans_v1.pt` (synthetic span),
  `outputs/glyph_cnn_real.pt` (real fine-tune). Trained on g3ses+QSOs+FAV22,
  evaluated on the held-out g6pz — no leakage.

## Open items

1. **R2 regression** (QSOs): 0.191 → 0.338 — hysteresis adds spurious elements
   on that file. A per-file automatic threshold (Otsu was miscalibrated) would
   fix it without losing the g6pz gains.
2. Remaining detection error: leftover deletions + unit estimation robustness.
3. CNN is parked, not dead: to make it help you'd need to train on the *hard*
   spans (not cleanly alignable) — bigger project.

## How to run

```
# improved pure-timing decoder (hysteresis is now the default)
python scripts/decode_beam.py livetests/g6pz/G4.wav --auto-center

# with the real fine-tuned CNN folded in at a safe low weight
python scripts/decode_cnn_beam.py outputs/glyph_cnn_real.pt \
    livetests/g6pz/G4.wav --auto-center --w-cnn 0.05

# harvest real training spans (forced alignment) from labeled audio
python scripts/harvest_real_spans.py outputs/glyph_cnn_spans_v1.pt \
    'livetests/g3ses/*.wav' 'livetests/QSOs/*.wav' --out real.npz
```
