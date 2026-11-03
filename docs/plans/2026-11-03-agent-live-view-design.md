# Agent live view: design plan

Status: proposal, 2026-11-03. Prototype: `docs/design/agent-live-view-prototype.html`
(open it in Chrome; it replays a full labeling run with real frames and real SAM 3 masks
from `torpedo_image_label`. Pause, replay and 1×/2×/4× are at the bottom right of the canvas.)

## Goal

Someone who glances at the web UI while Muse labels a sequence should understand within a
few seconds what the agent is doing and why, and whether it is going well. Today they see
a green breathing border, a teal box that draws itself, and frames jumping around.

## What the agent actually does

From `.agent/skills/agentseg-label/SKILL.md` a run has five stages, and every step maps to a
CLI call that already publishes activity events (`cli/main.py`, `backend/api/activity.py`):

| Stage | Agent action | CLI | Event today |
|---|---|---|---|
| Survey | reads a 24-frame contact sheet, picks a prompt frame | `sheet` | `artifact.created` (inspect_sequence) |
| First mask | reads a frame with a 50 px grid | `view f --grid` | `frame.started`, `artifact.created` |
| | draws a box, adds +/− clicks, up to 3 tries | `segment f --box --pos --neg` | `segment.started` (bbox only), `mask.proposed` |
| | reads the preview PNG and judges it | (file read in Muse) | none |
| Track | creates the track, propagates | `track create`, `track propagate` | `track.created`, `track.progress` (sparse), `track.finished` (+issues) |
| Review | reads each issue's worst frame | `view f --track T` | as above |
| | fixes: new keyframe / absent / remove keyframe | `track refine`, `track absent`, `track keyframes --rm` | `mask.refine_started`, `track.absent`, none for `--rm` |
| Export | final sheet, export, report | `sheet --track`, `export` | `artifact.created`, `export.completed` |

Gaps: the agent's thinking, its image reads, and its reasons are invisible. The UI can't tell
"thinking" from "waiting on SAM" from "finished", so the green border is shown for all of them.

## Design principles

1. **Hand, eyes, voice.** The agent gets a cursor for everything it does to the image (box,
   clicks, edge drags), brackets with a dimmed surround for everything it looks at, and a short
   line of text for why it acts. Every visual is backed by a real event; nothing is decorative.
2. **Three kinds of busy, shown differently.**
   - Thinking (LLM call running): the top bar's status text shimmers; the cursor's tag shows `Muse ···`.
   - Acting (tool on the canvas): the cursor animation itself is the indicator.
   - Waiting on compute (SAM 3): a scan sweep inside the box, a spinner ring on the cursor, a progress bar in the top bar.
3. **One agent color.** Violet means "the agent did this" (cursor, box, focus, active step).
   All other colors keep the meaning they have in the agent's own PNGs and the sheet legend:
   yellow = preview / keyframe, magenta = track mask, cyan = propagated box, green/orange/grey/red
   = ok/review/absent/no mask. The user and the agent read the same color code.
4. **Paced, never late.** Animations have minimum durations so a 200 ms CLI call is still
   visible, but the view never runs more than ~1.5 s behind real events (see Director below).
5. **The image is the hero; the panel is the explanation.** No modal, no toast spam, no glowing border.

## Layout

```
┌ header ─────────────────────────────────────────────────────────────────────┐
│ agentseg · project · Manual tools ▾ │ ◉ Muse Spark 1.3 │ status… │ 1:23 │ Following · Stop │
├ stage ───────────────────────────────────────────┬ run panel ──────────────┤
│ Frame 310/526 [keyframe]       Agent's view ▦   │ Task: “Label the board…” │
│              (image + agent overlay)             │ ① Survey        frame 23 │
│                                                  │ ② First mask     2 tries │
│                                                  │ ③ Track       526 frames │
│                                                  │ ④ Review           1 / 2 │
│                                                  │ ⑤ Export                 │
├ timeline ────────────────────────────────────────┤ Activity   [Commands ○] │
│ ■ torpedo_board  track 1     ◆2  ok 511  review 15│  feed: actions, thoughts,│
│        [1 fixed]            [2]                  │  snapshots, report card  │
│ ◆██████████████████◆█████████████████████▒▒▒████ │                          │
└──────────────────────────────────────────────────┴──────────────────────────┘
```

- **Top bar (presence).** Replaces the breathing border. Shows agent and model name, the
  current step in plain words ("Drawing a box on frame 23", "SAM 3 · tracking 312 / 526 frames"),
  elapsed time, and a progress bar during compute. Turns green with a summary when done.
  The manual toolbar collapses into "Manual tools ▾" while a run is live.
- **Stage.** Frame chip with keyframe/propagated/flagged badge; "Agent's view · 50 px grid"
  chip when the agent asked for `--grid` (we draw the same grid it sees); "Muse is looking
  at frame N" chip while it reads an image.
- **Timeline.** New. One cell per frame, colored by health status; yellow diamonds for
  keyframes; numbered pins over flagged ranges (orange → green "fixed" / grey "ignored");
  violet playheads during propagation (forward and backward from each keyframe, like
  TrackEngine); stale frames dim while they are being regenerated; a small frame preview
  rides the forward playhead; white marker for the frame on screen. Clicking a cell or pin
  jumps there (and detaches Follow).
- **Run panel.** The user's task, the resolved label chip, a 5-step stepper with per-step
  results, and an activity feed. Feed rows: action (icon, plain title, frame chip, result,
  live spinner while running), thought (agent's one-line reason, violet rule), snapshot (what
  the agent saw, same overlay colors), issue list, final report card (track, frames, keyframes
  and why, absent ranges, ignored issues, export path). Raw `agentseg …` commands are hidden
  behind a "Commands" toggle.

## Event → visual mapping

| Event | Canvas | Timeline | Panel / top bar |
|---|---|---|---|
| `agent.session` started/ended (new) | — | — | run panel opens / report; top bar on/off |
| `agent.thinking` start/stop (new) | cursor tag `Muse ···` | — | shimmer status |
| `agent.thought` (new) | — | — | thought row |
| `artifact.created` sheet | frame shrinks into a 6×4 sheet, rows highlight as it reads, chosen tile zooms back to full | — | "Contact sheet" row |
| `frame.started` / `artifact.created` view | crossfade to frame; grid if `--grid` | current-frame marker moves | "Inspect frame N" row + snapshot |
| `agent.looked` (new) | brackets converge on the region of interest, surround dims | — | "Muse is looking at…" chip |
| `segment.started` | cursor moves (curved, eased), presses, drags the box with a live size readout; then +/− clicks with ripples | — | "Box prompt · try k" row |
| (waiting on SAM) | scan sweep inside the box, spinner ring on cursor | — | "SAM 3 · segmenting" |
| `mask.proposed` | yellow mask revealed radially from the box center; score chip | — | result "score 0.97" + snapshot |
| next `segment.started` on the same frame | cursor drags the edge that changed; old mask crossfades to new | — | "try 2" |
| `track.created` | preview turns magenta, box and points fade | track appears, first diamond drops | label chip appears in task card |
| `track.progress` / `track.frames` | stays on keyframe (Follow can show live frames) | playheads fill cells; preview rides along | progress in top bar |
| `track.finished` | — | pins drop in | issue list row |
| `mask.refine_started` + `keyframe.added` | same box/click choreography on the issue frame | diamond drops; affected range dims, then re-fills | "Add keyframe and re-track" |
| `track.absent` | — | range turns grey with a left-to-right wipe | row with range |
| `keyframe.removed` (new) | — | diamond fades out; range re-fills | row |
| `export.completed` | — | — | export row + report card; top bar "Done" |
| `command.failed` | cursor shakes once, red chip | — | red row with the error text |

## Data contract changes

CLI / backend:
1. `segment.started`: add `pos`, `neg`, `try` (count of previews on this frame) and `prev_preview_id`.
2. `mask.refine_started`: add `box/pos/neg` when given. New `keyframe.added` after the POST, and `keyframe.removed` for `track keyframes --rm` (emits nothing today).
3. `track.progress` only fires on the first frame, last frame and frames with a render URL. Add a throttled `track.frames` batch (~4/s) with `[frame, status]` pairs and `direction` / `from_keyframe`.
4. `GET /api/video/tracks/{id}/frames` → per-frame status array (the logic exists as `_frame_statuses` in `backend/api/video_tracks.py`), so the timeline can load and replay without events.
5. A `session_id` on every event (env `AGENTSEG_SESSION`, set by the Muse launcher/hook). Today each CLI call gets its own `run_id`, so the UI can't group a run.

Muse bridge (hooks in the project's Muse config; Muse 1.4.3 has `sessionStart`, `sessionEnd`, `stop`, `preLLMCall`, `postLLMCall`, `preToolUse`, `postToolUse`, `userPromptSubmit`):
6. `sessionStart` / `sessionEnd` / `stop` → `agent.session` (replaces the guessing in `App.tsx` that toggles `agentActive` on any `*.started` / `*.finished`).
7. `preLLMCall` / `postLLMCall` → `agent.thinking`.
8. `postToolUse` on a file read under `.agentseg/views/` → `agent.looked` with the frame and region parsed from the artifact metadata.
9. `userPromptSubmit` → `agent.task`; reasoning summaries / short agent messages → `agent.thought`. If hook payloads lack the reasoning text, take it from `muse exec --json` (`reasoning` and `agentMessage` items) when the backend launches the run.

## Frontend architecture

- `agentDirector` (zustand store + rAF loop): turns events into "beats" with minimum
  durations (cursor travel 360–1050 ms by distance, box drag ~800 ms, mask reveal ~650 ms,
  look ~1.5 s). When the backlog exceeds ~1.5 s it speeds beats up, then collapses them to
  their end states. History replay at 1/2/4/8× comes from the same queue, since events are
  persisted in `activity_events`.
- Components: `AgentPresence` (header), `AgentOverlayLayer` (Konva layer: cursor, box,
  handles, points, ripples, sweep, brackets, spotlight; coordinates are image pixels like the
  CLI), `TrackTimeline` (canvas strip + DOM diamonds/pins/playheads), `AgentRunPanel` (task,
  stepper, feed, report). The stepper stage is derived from events with fixed rules (sheet
  before any track → Survey; segment/view before `track.created` → First mask; propagate →
  Track; inspections after issues → Review; final sheet/export → Export).
- Follow mode: on by default; any manual navigation detaches it and shows a
  "Return to Muse" chip.
- Delete `.agent-canvas-active` and the teal self-drawing box in `ImageCanvas.tsx`.
- Reduced motion: the cursor jumps instead of travelling, no sweeps or zooms; text states stay.
- Frame numbers are 1-based in the UI and 0-based in the CLI. The feed shows 1-based; raw
  commands (behind the toggle) stay as the agent typed them.

## Phases

1. Data contract: CLI event fields, `keyframe.*`, `track.frames`, frames endpoint, `session_id`, Muse hooks.
2. Top bar + run panel (stepper, feed, report), border removed. Biggest clarity gain for the effort.
3. Track timeline.
4. Canvas choreography: cursor, box, clicks, sweep, brackets, sheet transition.
5. Director pacing + replay.
6. Polish: reduced motion, failure states, long runs (virtualized feed), tests.

## Open questions

- Where does a run start? From a terminal `muse` session (hooks only), or from a prompt box
  in the web UI that runs `muse exec --json` (full event stream, working Stop button)?
  Recommendation: support both, with the web prompt box as the main path.
- Show the agent's reasoning lines in the feed? Recommendation: yes, one line per decision.
- Violet as the agent color: OK?
