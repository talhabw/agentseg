#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <seq_name> <image_dir> <gt_labels_dir> \"<instruction>\"" >&2
  exit 2
fi

SEQ=$1
IMAGE_DIR=$2
GT_DIR=$3
INSTRUCTION=$4
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TRIAL_ROOT=/tmp/autoseg_trial
RUN_DIR="$TRIAL_ROOT/$SEQ"
BLIND_DIR="$RUN_DIR/images"
STATE_DIR="$RUN_DIR/state"
TRANSCRIPT="$RUN_DIR/transcript.jsonl"
RESULT="$RUN_DIR/result.json"

rm -rf "$RUN_DIR"
mkdir -p "$BLIND_DIR" "$STATE_DIR"
cp "$IMAGE_DIR"/* "$BLIND_DIR"/

export AUTOSEG_HOME="$STATE_DIR"
cd "$ROOT"

if ! .venv/bin/autoseg status >/tmp/autoseg_trial_${SEQ}_status.json 2>/tmp/autoseg_trial_${SEQ}_status.err; then
  .venv/bin/autoseg up >/tmp/autoseg_trial_${SEQ}_up.json 2>/tmp/autoseg_trial_${SEQ}_up.err
fi

START_NS=$(date +%s%N)
set +e
timeout --signal=TERM --kill-after=30s 2700s /home/emin/.nvm/current/bin/claude -p "$INSTRUCTION" --model sonnet --dangerously-skip-permissions --output-format stream-json --verbose >"$TRANSCRIPT"
CLAUDE_EXIT=$?
set -e
END_NS=$(date +%s%N)

EXPORT_DIR=$(find "$RUN_DIR/export" "$STATE_DIR" -type f -name '*.txt' -not -path '*/classes/*' -printf '%h\n' 2>/dev/null | sort -u | while read -r d; do [[ -n "$d" ]] && echo "$d"; done | tail -n 1)
if [[ -z "${EXPORT_DIR:-}" ]]; then EXPORT_DIR="$STATE_DIR/export"; fi

SCORE_JSON="$RUN_DIR/score.json"
# Exports use zero-padded frame indices, while source labels retain camera stems.
MAPPED_GT="$RUN_DIR/gt_mapped"
mkdir -p "$MAPPED_GT"
.venv/bin/python - "$GT_DIR" "$MAPPED_GT" <<'PY'
import shutil, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for i, p in enumerate(sorted(src.glob('*.txt')), 1):
    shutil.copy2(p, dst / f'{i:06d}.txt')
PY
.venv/bin/python "$ROOT/scripts/score_export.py" "$EXPORT_DIR" "$MAPPED_GT" --json "$SCORE_JSON" >/tmp/autoseg_trial_${SEQ}_score.json || true

.venv/bin/python - "$SEQ" "$RUN_DIR" "$STATE_DIR" "$TRANSCRIPT" "$RESULT" "$CLAUDE_EXIT" "$START_NS" "$END_NS" "$EXPORT_DIR" "$SCORE_JSON" <<'PY'
import json, sys, re
from pathlib import Path

seq, run, state, transcript, result, exit_code, start_ns, end_ns, export_dir, score_path = sys.argv[1:]
run, state = Path(run), Path(state)
score = json.loads(Path(score_path).read_text()) if Path(score_path).exists() else {"error": "score unavailable"}
counts = {}
log = state / "cli_log.jsonl"
if log.exists():
    for line in log.read_text().splitlines():
        try:
            argv = json.loads(line).get("argv", [])
            if argv: counts[argv[0]] = counts.get(argv[0], 0) + 1
        except Exception: pass

viewed = set(); final_message = ""; usage = {}; cost = None
for line in Path(transcript).read_text(errors="replace").splitlines():
    try: ev = json.loads(line)
    except Exception: continue
    # Claude stream-json tool events have either direct name/input or nested content blocks.
    blocks = ev.get("content") if isinstance(ev.get("content"), list) else []
    candidates = [ev] + [b for b in blocks if isinstance(b, dict)]
    for c in candidates:
        name = c.get("name") or c.get("tool_name")
        inp = c.get("input") or c.get("tool_input") or {}
        if name == "Read" and isinstance(inp, dict):
            p = inp.get("file_path") or inp.get("path") or ""
            if str(p).lower().endswith(".png"): viewed.add(str(p))
    if ev.get("type") == "result" or ev.get("subtype") == "success":
        final_message = ev.get("result") or ev.get("message") or ev.get("text") or final_message
        usage = ev.get("usage") or usage
        cost = ev.get("cost_usd", ev.get("total_cost_usd", cost))
    if ev.get("type") == "assistant" and isinstance(ev.get("message"), dict):
        for c in ev["message"].get("content", []):
            if c.get("type") == "tool_use" and c.get("name") == "Read":
                p = (c.get("input") or {}).get("file_path", "")
                if str(p).lower().endswith(".png"): viewed.add(str(p))
            if c.get("type") == "text": final_message = c.get("text", final_message)
        usage = ev["message"].get("usage", usage) or usage
    if isinstance(ev.get("usage"), dict): usage = ev["usage"]
    if cost is None and isinstance(ev.get("message"), dict): cost = ev["message"].get("cost_usd")

pngs = [p for p in state.rglob("*.png")]
final_sheet = max(pngs, key=lambda p: p.stat().st_mtime) if pngs else None
out = {"sequence": seq, "export_dir": export_dir, "score": score,
       "cli_calls": counts, "png_images_opened": len(viewed), "wall_seconds": round((int(end_ns)-int(start_ns))/1e9, 2),
       "tokens": usage, "cost_usd": cost, "final_message": final_message,
       "human_interventions": 0, "claude_exit": int(exit_code),
       "transcript": transcript, "final_sheet": str(final_sheet) if final_sheet else None}
(run / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(out, indent=2, ensure_ascii=False))
PY
cat "$RESULT"
