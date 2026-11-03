---
name: agentseg-label
description: Label or track an object through a video / image sequence in this repo with the `agentseg` CLI (SAM 3 tracking + health checks). Use for requests like "label the torpedo", "label the board in this folder", "track X through these frames", "fix the labels of track N", or exporting those labels.
---

# Labeling a sequence with `agentseg`

You segment the object once, SAM 3 tracks it through every frame, and the
backend flags frames that look wrong ("issues"). Your job is to give a good
first mask, then inspect issues and fix only what is actually wrong. Most
sequences need 0–3 fixes.

Run the CLI from the repo root as `.venv/bin/agentseg …` (or `uv run agentseg …`).
Every command prints one JSON object; read `ok`, `error` and `next`. Images are
written as PNG files and returned as `"image": "<path>"`. **Open that path with
your file-reading tool to see it.** You cannot judge a mask without looking.

Conventions: pixel coordinates, boxes `x0,y0,x1,y1`, points `x,y`, frames are
0-based indices in the project's image order. Frame ranges are written like
`120-139,150`.

## 0. Services

```
agentseg status            # backend + sam_server + open project
agentseg up                # starts whatever is down (sam_server takes ~1–2 min)
```

If `up` fails, read the log path it prints. Do not kill other GPU processes.

## 1. Open the project and find the object

```
agentseg project open <image_folder>     # or an existing project dir (has agentseg.db)
agentseg sheet                           # ~24 thumbnails across the whole sequence
```

`project open` returns `frames` and `image_size`. Look at the sheet. Pick a
prompt frame where the object is **fully visible, large and sharp**, not
necessarily frame 0. Tracking runs both forward and backward from keyframes.

## 2. Resolve the label before the first mask

Choose the label name before calling `segment` or creating a track. Use the
object name from the user's request; when it is not explicit, infer a short,
specific name from the image and state that choice in the run. Never leave the
label as `unknown`, an empty string, or a placeholder. If there are several
plausible objects and the choice changes the annotation target, ask the user
before running SAM. The resolved name is carried into the first `track create`
call and every later report.

For example, for “label the torpedo” resolve `torpedo`; for “label the board
in this folder” resolve `board` (or `torpedo_board` when that is the clear
object identity). Keep it stable for the entire sequence.

Register it in the project's label list before the first mask (this is the
same list a person picks from in the web UI; reuse an existing label when one
already names the object):

```
agentseg labels                     # existing labels
agentseg labels --add <resolved_label>
```

## 3. First mask

```
agentseg view <f> --grid                          # 50 px labelled grid for reading coordinates
agentseg segment <f> --box x0,y0,x1,y1            # preview only, nothing saved
```

- Read the box off the grid with a small margin (5–10 px) around the object.
- Open the preview image. The yellow fill is the proposed mask, with an outline.
- **Scope matters.** Include exactly the object the user means. If they say "the
  board" and the mask also covers the stand or legs, re-run `segment` with
  the same `--box` plus `--neg x,y` on the unwanted part. Add `--pos x,y` on a
  missed part. Both flags can be repeated. Keep the box as it was and fix the
  mask with clicks; change the box only if it missed part of the object.
- Iterate until the preview is right, at most 3 tries. Then:

```
agentseg track create --label <resolved_label> --preview <preview_id>
agentseg track propagate <track_id>               # ~1 min per 500 frames; returns issues
```

Use the label name the user gave, or a short descriptive one such as `torpedo_board`.

## 4. Review issues

`propagate` (and `refine`, `absent`, `keyframes --rm`) return the current
issue list:

```
{"id": 9, "start": 210, "end": 228, "worst_frame": 218, "kind": "drift", "n_frames": 19, "reasons": {...}}
```

For each issue, look at its `worst_frame`:

```
agentseg view <worst_frame> --track <T> --grid
```

In the view, the track mask is magenta with an outline. A yellow box marks a
keyframe and a cyan box a propagated frame.

| What you see | Action |
|---|---|
| Mask is correct; the object just got smaller, larger, rotated or moved fast | **Ignore.** Write one line why in your report. |
| Mask spills onto background or other objects, or has stray specks far away (huge bbox) | `segment` that frame with a tight box (+ `--neg` on the spill), then `track refine T <f> --preview P` |
| Mask covers only part of the object, or sits on the wrong object | Same: `segment` → `track refine` |
| Object is not visible at all (out of frame, fully occluded), but a mask exists | `track absent T <range>`. Find the range first with `sheet --track T --frames a-b`. |
| Object is visible but there is no mask (red "empty" in the sheet) | `segment` → `track refine` on a frame where it is clearly visible |

Rules:
- **Never refine without looking at the preview first.** A bad keyframe poisons
  the tracker for many frames.
- If a keyframe you added made things worse: `agentseg track keyframes T --rm <f>`.
- **Batch fixes:** add `--no-propagate` to all but the last `refine` or
  `absent`, or follow them with one `agentseg track propagate T`. Each
  propagation costs 30–60 s.
- For `absent`, check the boundaries. Use `sheet --track T --frames a-b`
  around the range and mark only frames where the object is truly not visible
  (a sliver at the image edge still counts as visible).
- Same issue still there after 2 fixes → stop fixing it and mention it in your
  report.
- Budget: about 10 fixes and about 40 viewed images per sequence. Prefer one
  `sheet` over many `view`s when scanning a range.

## 5. Final check and export

```
agentseg sheet --track T                 # border colours: yellow keyframe, green ok,
                                        # orange review, grey absent, red no mask
agentseg export                          # YOLO-seg (default); --format yolo-detect|coco
```

Look at the final sheet. Orange frames are fine if you inspected them and
decided to ignore them. Red frames need attention, unless the object is truly
absent there; then mark them `absent`.

## 6. Report to the user

Short:
- track id and label
- number of frames
- keyframes (frames and why)
- absent ranges
- issues ignored, with one-line reasons
- issues left unresolved
- export path
- final sheet path

## Reference

```
agentseg status | up [--no-sam] [--no-ui] | down [--all]
agentseg project open <dir> [--name N] [--project-dir D]
agentseg labels [--add NAME]
agentseg view <f> [--track T] [--grid] [--preview P]
agentseg sheet [--track T] [--every N | --frames a-b,c]
agentseg segment <f> --box x0,y0,x1,y1 [--pos x,y]... [--neg x,y]... [--no-grid]
agentseg track create --label L --preview P
agentseg track propagate T [--full] [--start a --end b] [--no-wait]
agentseg track issues T [--verbose]
agentseg track refine T <f> (--preview P | --box ... [--pos ..] [--neg ..]) [--no-propagate]
agentseg track keyframes T [--rm F] [--no-propagate]
agentseg track absent T <frames> [--no-propagate]
agentseg track list
agentseg export [--format yolo-seg|yolo-detect|coco] [--out DIR] [--approved-only] [--labels-only]
```

All commands accept `--help`. Errors print `{"ok": false, "error": "..."}` and
exit non-zero; read the message, since it usually names the fix.
