# Session handoff — 2026-06-30 (element-level beam decoder)

Follows [`HANDOFF_SESSION_2026-06-29.md`](./HANDOFF_SESSION_2026-06-29.md).
Technical background on the G6PZ/G12 problem is in
[`HANDOFF_G6PZ_G12.md`](./HANDOFF_G6PZ_G12.md).

## Headline

Built a new **element-level beam/DP Morse decoder** that recognises characters
without trusting inter-element gap thresholds (the core difficulty: hand-sent
operators have irregular spacing). It detects dit/dah elements and finds, by
exact DP, the grouping into *valid Morse glyphs* that best fits the gaps used
only as a soft prior. Committed as `c36785e`.

Why it matters:
- **single dits (E) are never dropped** — the original failure mode (the
  `min-segment-units` noise filter swallowed E's) is gone;
- **Morse validity forces splits** a local gap threshold misses (e.g. `...--.`
  must become S+G — fixes Serge's `[SG]` observation);
- **no checkpoint needed** (pure timing + Morse table).

## Results

- Generalises across livetests (no CNN): **median CER ~0.178** over 31 files,
  `G3` at **0.000**, several `< 0.10` (R9 0.048, C8 0.055, C12 0.077, ...).
- The 3 worst files are **data problems, not decoder problems**:
  - **G12** — audio repeats a middle passage the truth omitted (see below);
  - **R12b** — 148-char truth vs 1741 detected elements (audio ≫ truth);
  - **R12c** — 1826-char truth vs 313 elements (truth ≫ audio).
- On a *clean* G12 window (0-40 s, no repeat): CER **~0.33** vs the contaminated
  CNN best of 0.78.

## Data finding: G12 audio is duplicated

`livetests/G6PZ/G12.wav` physically sends two passages **twice in a row**
(`R1 R1 R2 R2`), but `G12.txt` transcribed them once — so every decoder scored
~30-40% phantom insertions. Confirmed by the CNN baseline showing the identical
duplication. This means **all prior G12 CER baselines (0.877 / 0.778) were
inflated**. See memory `g12-audio-duplicated-section`.

`livetests/G6PZ/G12.txt` was **corrected** on disk (local + pod) to include both
repeats: CER 0.784 -> **0.315** (matches the clean-window estimate, two
independent validations). NOTE: `livetests/` is **gitignored**, so this fix is
on disk only, NOT in version control.

## State

- Code committed + pushed: `c36785e` on origin, local, and pod (all in sync).
  - `morse_char_recognizer/beam_decode.py`, `scripts/decode_beam.py`,
    `tests/test_beam_decode.py` (suite: **77 passed**).
- Model pulled to local: `outputs/experiments/20260629T070149Z-rhythm-analog/`
  (`checkpoint.pt`, sha256-verified vs pod). `outputs/` is gitignored.
- Pod SSH (changes on migration): `root@194.68.245.211 -p 22054`. See memory
  `runpod-pod-access`.

## Open items (not done)

1. **Version the corrected truth `.txt`?** `G12.txt` fix lives on disk only
   (gitignored). Decide: `git add -f` the `.txt` files, or keep as out-of-git
   data + a tracked corrections note.
2. **R12b / R12c** truth-vs-audio mismatches — flag/fix.
3. **Next decoder step (planned):** add CNN scoring into the beam to resolve
   ambiguous glyphs (e.g. `EDIKE`->`LIKE`, `IÉIT`->`BULIT`), and restrict the
   allowed character set (French accents like `É` currently leak in). Also fix
   missing word spaces (`TOSE` -> `TO SE`).

## How to run the beam decoder

```
python scripts/decode_beam.py livetests/G6PZ/G12.wav --auto-center
# reports decoded text + CER vs the sibling .txt truth
```
