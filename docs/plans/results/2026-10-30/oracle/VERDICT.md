# WP3 gate verdict (hand-written, 2026-10-30)

Generated numbers: `README.md` / `summary.json` (default config, DINO on) and
`summary_nodino.json` (`HealthConfig.geometry_only()`). WP1 acceptance: `check_remove.json`.

## Verdict: gate FAILS on the raw issue-recall criterion only

| criterion | result | status |
|---|---|---|
| torpedo mean IoU >= 0.95 | 0.9662 | pass |
| torpedo <= 1 edit / 100 frames | 0 edits | pass |
| stress mean IoU >= 0.90 | sub4 0.9634, exit 0.9468, occl 0.9562, mid 0.9574 | pass |
| loss in exit/occl caught as issue | exit 5/5, occl 1/1 pass-1 loss events | pass |
| issue recall >= 0.9 (raw, pooled over rounds) | 0.69 (DINO) / 0.94 (geometry-only) | **fail** (default config) |

Why recall fails, in order of weight:

1. **GT label error, not a tracker error.** On source frames 393-394 the GT box also covers
   the stand legs (`label_error_393.png`); the neighbouring frames box only the board and
   the predicted mask is correct. These 2 frames appear in torpedo, exit, occl and mid = 8 of
   the 37 pass-1 "bad" frames. No mask-based signal can flag a correct mask, so they cap the
   raw recall. Excluding them: pass-1 recall **0.931** (both configs), pooled 0.88 (DINO) /
   0.96 (geometry-only).
2. **Two borderline exit frames** (396, 459, IoU 0.795 / 0.793 vs the 0.8 "bad" cut) are the
   only genuine misses. The pooled metric counts them again in each of the 3 exit rounds,
   which is why pooled-excluding-label-error is 0.88 < 0.931.
3. The geometry-only config reaches 0.94 raw only because its keyframe-area rule opens a
   166-frame issue over torpedo's far-away stretch that happens to contain 393-394. Price:
   precision 0.05 and the oracle hits its 40-view cap on torpedo, sub4 and occl. That is
   not a better detector, so it is not the default.

What this means for WP4: the engine and signals are good enough to move on. Every genuine
tracker failure in the data (exit ghost slivers, full occlusion under the patch) was raised
as an issue on the first pass. Recommendation: accept the gate with this documented
exception, or fix the two GT labels and rerun `scripts/oracle_agent.py` (~6 min).

## Is WP8 (DINO re-detection fallback) needed?

**No, not on this data.** SAM 3 never lost the board for good: hit rate 1.0 on every
sequence, every re-entry in `exit` was re-acquired by the tracker itself, and `occl` recovered
right after the patch. The remaining errors are the opposite case, a mask where GT has none
(a sliver at the border while < 15 % is visible, or the fully covered board under the patch),
and the agent fixes those with `mark_absent`. DINO is still useful as a *health signal*
(occlusion / wrong-object frames, AUC 0.84-0.96 bad vs good, `dino_cost.json`). Caveat: all
five sequences derive from one video, so a real second dataset may change this.

## Cost observations

- `exit` needed 10 absent marks (budget exhausted, 1.9 edits/100 frames, absent accuracy
  0.74 -> 0.92). The object leaves 13 times. WP5/WP6 should let the agent mark several
  absent ranges from one contact sheet, or the backend could pre-suggest `absent` for
  `exit`-kind issues whose worst frame is a border sliver.
- Thresholds were tuned and evaluated on the same 5 sequences (no held-out data). Treat the
  defaults as a starting point.
