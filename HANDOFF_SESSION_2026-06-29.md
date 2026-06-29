# Session handoff — 2026-06-29 (infra / git sync)

This session was **infrastructure + git reconciliation only**. No model or
pipeline experiments were run. For the technical state of the CW Glyph problem
(G6PZ/G12 rhythm failure, hypotheses, next program), see
[`HANDOFF_G6PZ_G12.md`](./HANDOFF_G6PZ_G12.md) — it is current and authoritative.

## What was done this session

1. **New RunPod GPU pod connected.**
   - `ssh root@194.68.245.53 -p 22113 -i ~/.ssh/id_ed25519`
   - host `20fe29b3da14`, GPU **NVIDIA A40 (46 GB)**, repo at `/workspace/cw-glyph`,
     venv at `/workspace/cw-glyph/.venv` (Python 3.12.3).
   - Access required adding `~/.ssh/id_ed25519.pub` to the pod's
     `~/.ssh/authorized_keys` via the RunPod **web terminal** (the pod has no
     GitHub key and cannot reach `git@github.com`).

2. **Test suite validated on the pod:** `71 passed` (was 67 in the prior handoff).

3. **Git reconciled across all three locations.** Investigation showed the pod's
   uncommitted changes were the **same code** as the already-pushed local commit
   `c49feaf`, but with a slightly **older** `HANDOFF_G6PZ_G12.md` (missing the 44-line
   "G12 short E diagnosis" section) and `.gitignore` (missing `recovered/`).
   So `origin/main = c49feaf` was a **strict superset** of the pod.
   - Pod was reset to `c49feaf` (the redundant pod commit was discarded — no work lost).
   - Sync was done by **pushing `c49feaf` from local to the pod** (local→pod), because
     the pod cannot pull from GitHub.

## Current state — fully synced

| Location | Commit | Tree |
|---|---|---|
| `origin/main` (GitHub) | `c49feaf` | `373a1f…` |
| Local `main` | `c49feaf` | `373a1f…` |
| Pod `/workspace/cw-glyph` `main` | `c49feaf` | `373a1f…` |

All three are identical. Working trees clean.

## To resume work

The next concrete program is in `HANDOFF_G6PZ_G12.md` → "Next practical program".
Short version:
1. Recreate the G12 previews with the current checkpoint + current segmentation.
2. Build a rhythm report for the first 30 s of G12 (envelope, active regions,
   keydown-duration histogram, gap histogram, local dot-unit, segments over `BK HI`).
3. If `BK HI` segments wrong → fix segmentation / local unit estimation before retraining.
4. If `BK HI` segments right but classifies wrong → train a timing-augmented checkpoint
   (5–50 WPM + random rhythm perturbations).

Reference checkpoints on the pod:
- analog inhomogeneous: `outputs/experiments/20260629T070149Z-rhythm-analog`
- reference: `outputs/experiments/20260625T200509Z-reference`

### Caveat for next sync
The pod cannot reach GitHub. To get new commits onto the pod, push from local:
```
GIT_SSH_COMMAND="ssh -p 22113 -i ~/.ssh/id_ed25519" \
  git push ssh://root@194.68.245.53/workspace/cw-glyph <ref>:refs/heads/tmp-sync
# then on the pod:  git reset --hard tmp-sync && git branch -D tmp-sync
```
