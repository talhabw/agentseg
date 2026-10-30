#!/usr/bin/env python3
"""Headless tracking client for the backend job API (issue #3). Stdlib only.

Examples:
  python scripts/track.py --project-dir /data/proj --load-models --start 0
  python scripts/track.py --start 120 --end 300 --failure-mode skip
  python scripts/track.py --attach <job_id>          # follow an existing job

When a job pauses (failure mode 'stop'), supply a new label with
  POST /api/tracks/{track_id}/reanchor {"image_id": ..., "bbox": [x1,y1,x2,y2]}
(or draw it in the web UI), then press Enter here to resume ('q' stops).
Use --on-pause stop|resume for non-interactive runs.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request


def call(base, method, path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path} -> {e.code}: {e.read().decode()}")


def stream(base, job_id, after=0):
    """Yield events from the SSE stream (reconnects from the last seq)."""
    while True:
        req = urllib.request.Request(f"{base}/api/track/jobs/{job_id}/stream?after={after}",
                                     headers={"Accept": "text/event-stream"})
        with urllib.request.urlopen(req, timeout=60) as r:
            for raw in r:
                line = raw.decode().rstrip("\n")
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    after = ev["seq"]
                    yield ev
                    if ev["type"] in ("finished", "error"):
                        return
        if call(base, "GET", f"/api/track/jobs/{job_id}")["status"] in ("finished", "stopped", "error"):
            return


def fmt(ev):
    t = ev["type"]
    if t == "started":
        return (f"started frames {ev['start_index']}..{ev['end_index']} with tracks "
                f"{[x['track_id'] for x in ev['tracks']]} (failure_mode={ev['settings']['failure_mode']})")
    if t == "frame_done":
        p = ev["progress"]
        return (f"[{p['done']}/{p['total']}] frame {ev['image_index']}: created={len(ev['created'])} "
                f"existing={len(ev['existing'])} skipped={len(ev['skipped'])} lost={ev['lost']} "
                f"uncertain={ev['uncertain']}")
    if t == "object_lost":
        return (f"LOST track {ev['track_id']} ({ev.get('label_name')}) at frame {ev['image_index']}: "
                f"{ev['reason']} - {ev['message']}")
    if t == "uncertain":
        return f"uncertain track {ev['track_id']} at frame {ev['image_index']}: {ev['reasons']}"
    if t == "possible_id_switch":
        return (f"possible ID switch: track {ev['track_id']} overlaps track "
                f"{ev['conflicting_track_id']} at frame {ev['image_index']}")
    if t == "paused_waiting_for_label":
        names = ", ".join(f"{x['track_id']}:{x['label_name']}" for x in ev["tracks"])
        return f"PAUSED at frame {ev['image_index']} (image_id {ev['image_id']}): waiting for label for {names}"
    if t == "finished":
        return f"finished ({ev['reason']}) at frame {ev['current_index']}: {ev['counts']}"
    if t == "error":
        return f"ERROR: {ev['message']}"
    return f"{t}: " + json.dumps({k: v for k, v in ev.items() if k not in ('seq', 'type', 'job_id', 'ts')})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--project-dir", help="open this project first")
    ap.add_argument("--load-models", action="store_true", help="POST /api/ml/propagate/load first")
    ap.add_argument("--embed-model", default="vith16")
    ap.add_argument("--start", type=int, default=0, help="start frame index")
    ap.add_argument("--end", type=int, help="end frame index (inclusive, default: last)")
    ap.add_argument("--tracks", type=int, nargs="*", help="only these track ids")
    ap.add_argument("--labels", type=int, nargs="*", help="only these label ids")
    ap.add_argument("--failure-mode", default="stop", choices=["stop", "skip", "continue"])
    ap.add_argument("--mode", default="auto", choices=["peak", "dense", "auto"])
    ap.add_argument("--on-pause", default="ask", choices=["ask", "stop", "resume"])
    ap.add_argument("--attach", help="follow an existing job id instead of starting one")
    ap.add_argument("--json", action="store_true", help="print raw JSON events")
    a = ap.parse_args()
    base = a.url.rstrip("/")

    if a.project_dir:
        call(base, "POST", "/api/projects/open", {"project_dir": a.project_dir})
    if a.load_models:
        print("loading models...", file=sys.stderr)
        call(base, "POST", "/api/ml/propagate/load", {"embed_model": a.embed_model}, timeout=900)

    if a.attach:
        job_id = a.attach
    else:
        body = {"start_index": a.start, "end_index": a.end, "failure_mode": a.failure_mode,
                "mode": a.mode, "track_ids": a.tracks, "label_ids": a.labels}
        job_id = call(base, "POST", "/api/track/jobs", body)["id"]
    print(f"job {job_id}", file=sys.stderr)

    try:
        for ev in stream(base, job_id):
            print(json.dumps(ev) if a.json else fmt(ev), flush=True)
            if ev["type"] == "paused_waiting_for_label":
                action = a.on_pause
                if action == "ask":
                    ans = input("re-anchor the lost track(s), then Enter to resume / 'q' to stop: ")
                    action = "stop" if ans.strip().lower().startswith("q") else "resume"
                call(base, "POST", f"/api/track/jobs/{job_id}/{action}")
    except KeyboardInterrupt:
        call(base, "POST", f"/api/track/jobs/{job_id}/stop")
        print("stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
