# WP7 blind end-to-end agent trial

Two headless Claude Sonnet runs were allowed: torpedo and exit. Human interventions were 0 in both. Full transcripts are retained under `/tmp/autoseg_trial/{torpedo,exit}/transcript.jsonl` and are not copied here.

| sequence | mean IoU (agent / oracle) | hit (agent / oracle) | good (agent / oracle) | absent acc (agent / oracle) | edits (agent / oracle) | CLI calls | PNGs viewed | wall time | tokens / cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| torpedo | 0.9532 / 0.9662 | 1.0 / 1.0 | 0.9905 / 0.9962 | — / — | 0 / 0 | 14 | 7 | 95.41 s | 473,941 / $0.1807 |
| exit | 0.9165 / 0.9468 | 0.9954 / 1.0 | 0.9772 / 0.9909 | 0.7159 / 0.9205 | 0 / 10 | 13 | 7 | 100.66 s | 488,746 / $0.1820 |

## Verdict

- **Torpedo: PASS.** Agent mean IoU 0.9532 is 0.0130 below the oracle 0.9662 (within 0.02), with 0 human interventions.
- **Exit: PASS on the plan's IoU criterion, FAIL on absent handling.** Agent mean IoU 0.9165 is 0.0303 below the oracle 0.9468 (within 0.05). Absent accuracy is 0.716 versus 0.9205. The agent never called `track absent`, but it was right that the tracker already returned empty masks for most off-frame frames; the remaining 25 wrong frames are thin slivers at the image edge that GT drops (<15% of the box visible) and the agent judged visible.

_Correction (rescored after the runs, no rerun):_ the first version of this report gave exit 0.1206. The harness paired GT files with the zero-padded export names in lexicographic order (`0, 1, 10, 100, …`), which misaligned every frame of `exit` (its GT is named `0.txt…525.txt`). The harness now sorts naturally; torpedo's camera-stem names were unaffected (0.9532 both ways).

## What the agent did

On torpedo, the agent opened a full-sequence sheet, inspected frame 22, segmented a tight board box, propagated one track, inspected issue sheets/views, ignored small-size/lost health flags after visual checks, and exported. It used one keyframe and no edits.

On exit, it chose frame 308 as the keyframe, propagated, sampled the out-of-frame ranges, and correctly described those periods as absent in its final message. It nevertheless made no `track absent` calls, ignored all eight issues, and exported. It also only made a targeted final sheet rather than a full-sequence final sheet. The export reported 63 bbox-rectangle fallbacks.

## Failure and inefficiency classification

| observation | class | suggested fix |
|---|---|---|
| Exit empty-GT frames received predictions; absent accuracy 0.0 | skill gap | Require the agent to call `track absent` for every confirmed out-of-frame interval before export; add an export-time check that blocks or warns on non-empty masks over confirmed absence issues. |
| Exit issues 1–8 were all ignored despite sampled frames showing the board absent | skill gap / health signal | Teach the issue decision rule that `lost`/`exit` plus confirmed absence requires marking ranges absent, and expose a concise absent-range candidate in `track issues`. |
| Exit export contained 63 bbox rectangles instead of polygons | CLI gap | Make export report frame IDs for fallback rectangles and provide a direct re-segmentation/refine path for them; optionally fail validation when polygon output is requested but unavailable. |
| Exit final verification covered selected frames only | skill gap | Require a final `sheet --track` over the whole sequence (or an explicit `--every` full-range sheet) before export. |
| Initial harness score missed labels because camera stems differ from numeric export stems; exit export can be written outside AUTOSEG_HOME; GT was then mapped in lexicographic instead of natural order (exit scored 0.12 instead of 0.92) | harness bug (fixed, rescored; no rerun) | Map sorted GT labels to exported frame indices and search both `$AUTOSEG_HOME` and the run export directory. |

The corrected harness score was computed after the torpedo run without rerunning the agent; the only permitted agent runs remain exactly two.

## Post-trial fixes (no rerun)

- **Backend:** empty propagated masks were stored as annotations, and export wrote them as degenerate all-zero polygons (the agent's "63 bbox fallbacks"). Video jobs now skip empty masks.
- **Scorer:** zero-area boxes now count as "no object". Exit absent accuracy went from 0.0 to 0.716 with the same export.
