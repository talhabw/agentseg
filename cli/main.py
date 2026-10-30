"""`autoseg` CLI: a thin client over the backend's video tracking API.

Every command prints exactly one JSON object on stdout. Images are written as
PNG files (default `.autoseg/views/`) and their paths are returned in the JSON.
Errors exit non-zero and print `{"ok": false, "error": "..."}`. Long jobs print
progress on stderr. Every invocation is appended to `.autoseg/cli_log.jsonl`.

Coordinates are always pixels, boxes are `x0,y0,x1,y1`, frames are 0-based
order indices of the project's images.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
HOME = Path(os.getenv("AUTOSEG_HOME", ".autoseg"))
BACKEND = os.getenv("AUTOSEG_BACKEND", "http://localhost:8000")
SAM_URL = os.getenv("SAM_SERVER_URL", "http://localhost:8770")


class CliError(Exception):
    def __init__(self, message: str, code: int = 1, **extra):
        super().__init__(message)
        self.code, self.extra = code, extra


# ---------------------------------------------------------------- HTTP helpers

class Api:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.http = httpx.Client(base_url=self.base, timeout=600.0)

    def _call(self, method: str, path: str, **kw):
        try:
            r = self.http.request(method, path, **kw)
        except httpx.TransportError as exc:
            raise CliError(f"backend not reachable at {self.base} ({exc.__class__.__name__}); run `autoseg up`")
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise CliError(f"{method} {path} -> {r.status_code}: {detail}", status=r.status_code)
        return r

    def get(self, path, **kw):
        return self._call("GET", path, **kw).json()

    def post(self, path, body=None):
        return self._call("POST", path, json=body or {}).json()

    def delete(self, path):
        return self._call("DELETE", path).json()

    def png(self, path, params, out: Path) -> Path:
        r = self._call("GET", path, params=params)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(r.content)
        return out.resolve()


def _alive(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2.0).status_code < 400
    except httpx.HTTPError:
        return False


def _sam_ready() -> dict | None:
    try:
        r = httpx.get(f"{SAM_URL}/health", timeout=2.0)
        return r.json() if r.status_code < 400 else None
    except (httpx.HTTPError, ValueError):
        return None


# ---------------------------------------------------------------- parsing

def _floats(s: str, n: int, what: str) -> list[float]:
    try:
        v = [float(x) for x in s.split(",")]
    except ValueError:
        v = []
    if len(v) != n:
        raise CliError(f"{what} must be {n} comma-separated numbers, got {s!r}", 2)
    return v


def _box(s: str) -> list[float]:
    b = _floats(s, 4, "--box")
    if b[2] <= b[0] or b[3] <= b[1]:
        raise CliError("--box must be x0,y0,x1,y1 with x1>x0 and y1>y0", 2)
    return b


def _points(items) -> list[list[float]]:
    return [_floats(p, 2, "point") for p in (items or [])]


def _frames(spec: str) -> list[int]:
    """'3,10-14,20' -> [3,10,11,12,13,14,20]."""
    out: set[int] = set()
    try:
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                a, b = (int(x) for x in part.split("-", 1))
                out.update(range(min(a, b), max(a, b) + 1))
            elif part:
                out.add(int(part))
    except ValueError:
        raise CliError(f"bad frame spec {spec!r}; use e.g. 3,10-14", 2)
    if not out:
        raise CliError("empty frame spec", 2)
    return sorted(out)


def _compact_issue(i: dict) -> dict:
    out = {"id": i.get("id"), "start": i.get("start_frame", i.get("start")),
           "end": i.get("end_frame", i.get("end"))}
    keep = ("worst_frame", "kind", "n_frames", "n_lost", "reasons")
    return {**out, **{k: i[k] for k in keep if k in i}}


# ---------------------------------------------------------------- commands

def cmd_status(api: Api, a) -> dict:
    out = {"backend": {"url": api.base, "up": _alive(f"{api.base}/api/health")},
           "sam_server": {"url": SAM_URL, "health": _sam_ready()}}
    if out["backend"]["up"]:
        proj = api.get("/api/projects/current")
        out["project"] = proj
        if proj:
            out["tracks"] = [_track_summary(t) for t in api.get("/api/video/tracks")]
    if not out["backend"]["up"] or not out["sam_server"]["health"]:
        out["hint"] = "run `autoseg up` to start the missing services"
    return out


def _spawn(cmd: list[str], cwd: Path, log: Path, env: dict | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "ab")
    p = subprocess.Popen(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, env={**os.environ, **(env or {})})
    return p.pid


def _wait(check, timeout: float, what: str):
    end = time.time() + timeout
    while time.time() < end:
        if check():
            return
        time.sleep(2)
    raise CliError(f"{what} did not become ready within {timeout:.0f}s")


def cmd_up(api: Api, a) -> dict:
    out = {}
    if _sam_ready():
        out["sam_server"] = "already running"
    elif a.no_sam:
        out["sam_server"] = "down (skipped)"
    else:
        log = (HOME / "sam_server.log").resolve()
        attrs = '{"batched_grounding_batch_size": 4, "postprocess_batch_size": 4}'
        port = SAM_URL.rsplit(":", 1)[-1].strip("/")
        pid = _spawn(["uv", "run", "sam-server", "--port", port, "--models", "sam3,image",
                      "--sam3-checkpoint", str(REPO / "models/sam3/sam3.pt"), "--video-model-attrs", attrs],
                     REPO / "sam_server", log, {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        print(f"starting sam_server (pid {pid}, log {log}) ...", file=sys.stderr)
        _wait(lambda: (_sam_ready() or {}).get("ready"), 600, "sam_server")
        out["sam_server"] = {"started": pid, "log": str(log)}
    if _alive(f"{api.base}/api/health"):
        out["backend"] = "already running"
    else:
        log = (HOME / "backend.log").resolve()
        port = api.base.rsplit(":", 1)[-1].strip("/")
        pid = _spawn([sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1",
                      "--port", port], REPO, log, {"PYTHONPATH": str(REPO), "SAM_SERVER_URL": SAM_URL,
                                        "VIDEO_CLOSE_STALE_SESSIONS": "1"})
        print(f"starting backend (pid {pid}, log {log}) ...", file=sys.stderr)
        _wait(lambda: _alive(f"{api.base}/api/health"), 120, "backend")
        out["backend"] = {"started": pid, "log": str(log)}
    return out


def _project_info(api: Api, proj: dict) -> dict:
    first = api.get("/api/images/by-index/0") if proj["image_count"] else None
    return {**proj, "frames": proj["image_count"],
            "last_frame": proj["image_count"] - 1,
            "image_size": [first["width"], first["height"]] if first else None}


def cmd_project_open(api: Api, a) -> dict:
    path = Path(a.path).expanduser().resolve()
    if not path.is_dir():
        raise CliError(f"not a directory: {path}", 2)
    if (path / "autoseg.db").exists():
        proj, created = api.post("/api/projects/open", {"project_dir": str(path)}), False
    else:
        name = a.name or path.name if path.name != "images" else (a.name or path.parent.name)
        pdir = Path(a.project_dir).expanduser().resolve() if a.project_dir else (HOME / "projects" / name).resolve()
        if (pdir / "autoseg.db").exists():
            proj, created = api.post("/api/projects/open", {"project_dir": str(pdir)}), False
        else:
            proj, created = api.post("/api/projects", {"project_dir": str(pdir), "image_dir": str(path),
                                                      "name": name}), True
    return {"created": created, **_project_info(api, proj)}


def cmd_labels(api: Api, a) -> dict:
    if a.add:
        api.post("/api/labels", {"name": a.add})
    return {"labels": api.get("/api/labels")}


def _views() -> Path:
    return HOME / "views"


def cmd_view(api: Api, a) -> dict:
    params = {"frame": a.frame, "grid": a.grid}
    name = f"frame_{a.frame:05d}"
    if a.track is not None:
        params["track_id"] = a.track; name += f"_t{a.track}"
    if a.preview:
        params["preview_id"] = a.preview; name += "_preview"
    if a.grid:
        name += "_grid"
    path = api.png("/api/video/render/frame", params, Path(a.out or _views()) / f"{name}.png")
    return {"frame": a.frame, "image": str(path),
            "legend": "track masks tinted, yellow box = keyframe, cyan box = propagated, "
                      "yellow fill = preview; grid lines every 50 px labelled in pixels"}


def cmd_sheet(api: Api, a) -> dict:
    n = api.get("/api/projects/current") or {}
    count = n.get("image_count", 0)
    params, tag = {}, ""
    if a.frames:
        frames = _frames(a.frames)
        params["frames"] = ",".join(map(str, frames)); tag = f"f{frames[0]}-{frames[-1]}_{len(frames)}"
    else:
        every = a.every or max(1, math.ceil(count / 24))
        params["every"] = every; frames = list(range(0, count, every)); tag = f"every{every}"
    if a.track is not None:
        params["track_id"] = a.track; tag = f"t{a.track}_{tag}"
    path = api.png("/api/video/render/sheet", params, Path(a.out or _views()) / f"sheet_{tag}.png")
    out = {"image": str(path), "frames": frames}
    if a.track is not None:
        out["legend"] = ("thumbnail border: yellow=keyframe, green=ok, orange=review (health flagged), "
                         "grey=absent, red=no mask")
    return out


def cmd_segment(api: Api, a) -> dict:
    body = {"frame": a.frame, "box": _box(a.box), "pos": _points(a.pos), "neg": _points(a.neg)}
    res = api.post("/api/video/segment", body)
    params = {"frame": a.frame, "preview_id": res["preview_id"], "grid": not a.no_grid}
    path = api.png("/api/video/render/frame", params,
                   Path(a.out or _views()) / f"preview_{a.frame:05d}_{res['preview_id']}.png")
    return {**res, "image": str(path),
            "next": "look at the image; if the yellow mask is right use the preview_id with "
                    "`track create` or `track refine --preview`, otherwise re-run segment with "
                    "a better --box and/or --pos/--neg points"}


def _track_summary(t: dict) -> dict:
    job = t.get("job") or {}
    return {"id": t["id"], "label_id": t["label_id"], "keyframes": t["keyframes"],
            "absent": _ranges(t.get("absent") or []), "coverage": t["coverage"],
            "issue_count": t["issue_count"], "job_status": job.get("status")}


def _ranges(frames: list[int]) -> str:
    frames = sorted(frames); out = []; i = 0
    while i < len(frames):
        j = i
        while j + 1 < len(frames) and frames[j + 1] == frames[j] + 1:
            j += 1
        out.append(str(frames[i]) if i == j else f"{frames[i]}-{frames[j]}"); i = j + 1
    return ",".join(out)


def _get_track(api: Api, tid: int) -> dict:
    for t in api.get("/api/video/tracks"):
        if t["id"] == tid:
            return t
    raise CliError(f"track {tid} not found")


def cmd_track_create(api: Api, a) -> dict:
    res = api.post("/api/video/tracks", {"preview_id": a.preview, "label": a.label, "name": a.name})
    return {**res, "next": f"run `autoseg track propagate {res['track_id']}`"}


def _propagate(api: Api, tid: int, full=False, start=None, end=None, wait=True) -> dict:
    body = {"full": full}
    if start is not None: body["start"] = start
    if end is not None: body["end"] = end
    job = api.post(f"/api/video/tracks/{tid}/propagate", body)
    if not wait:
        return {"job": job}
    t0, seq, frames = time.time(), 0, 0
    lo, hi = job.get("frame_range") or (0, 0)
    total = hi - lo + 1
    while True:
        res = api.get(f"/api/video/jobs/{job['id']}/events", params={"after": seq, "wait": 15})
        for ev in res["events"]:
            seq = max(seq, ev["seq"])
            if ev["type"] == "frame":
                frames += 1
                if frames % 50 == 0:
                    print(f"  propagated {frames}/{total} frames ({time.time() - t0:.0f}s)", file=sys.stderr)
            elif ev["type"] == "error":
                raise CliError(f"propagation failed: {ev.get('message')}")
        if res["status"] in ("finished", "stopped", "error") and not res["events"]:
            break
    job = api.get(f"/api/video/jobs/{job['id']}")
    if job["status"] != "finished":
        raise CliError(f"propagation ended with status {job['status']}: {job.get('error')}")
    issues = [_compact_issue(i) for i in api.get(f"/api/video/tracks/{tid}/issues")]
    return {"track_id": tid, "frame_range": [lo, hi], "frames_written": job["counts"]["frames"],
            "seconds": round(time.time() - t0, 1), "issues": issues,
            "next": ("no issues: check `autoseg sheet --track %d` then export" % tid) if not issues else
                    "inspect each issue's worst_frame with `autoseg view <frame> --track %d --grid`" % tid}


def cmd_track_propagate(api: Api, a) -> dict:
    return _propagate(api, a.track, a.full, a.start, a.end, not a.no_wait)


def cmd_track_issues(api: Api, a) -> dict:
    raw = api.get(f"/api/video/tracks/{a.track}/issues")
    return {"track_id": a.track, "issues": raw if a.verbose else [_compact_issue(i) for i in raw]}


def cmd_track_refine(api: Api, a) -> dict:
    body = {"frame": a.frame}
    if a.preview:
        body["preview_id"] = a.preview
    elif a.box:
        body.update(box=_box(a.box), pos=_points(a.pos), neg=_points(a.neg))
    else:
        raise CliError("give --preview P or --box x0,y0,x1,y1", 2)
    res = api.post(f"/api/video/tracks/{a.track}/keyframes", body)
    out = {"keyframe": res}
    if not a.no_propagate:
        out["propagate"] = _propagate(api, a.track)
    return out


def cmd_track_keyframes(api: Api, a) -> dict:
    out = {}
    if a.rm is not None:
        out["removed"] = api.delete(f"/api/video/tracks/{a.track}/keyframes/{a.rm}")
        if not a.no_propagate:
            out["propagate"] = _propagate(api, a.track)
    t = _get_track(api, a.track)
    out["keyframes"] = t["keyframes"]
    return out


def cmd_track_absent(api: Api, a) -> dict:
    frames = _frames(a.frames)
    res = api.post(f"/api/video/tracks/{a.track}/absent", {"frames": frames})
    out = {"track_id": a.track, "absent": _ranges(res["frames"])}
    if not a.no_propagate:
        out["propagate"] = _propagate(api, a.track)
    return out


def cmd_track_list(api: Api, a) -> dict:
    return {"tracks": [_track_summary(t) for t in api.get("/api/video/tracks")]}


def cmd_export(api: Api, a) -> dict:
    proj = api.get("/api/projects/current")
    if not proj:
        raise CliError("no project open; run `autoseg project open <dir>`")
    out_dir = Path(a.out or HOME / "export" / f"{proj['name']}_{a.format}").expanduser().resolve()
    common = {"output_dir": str(out_dir), "train_split": 1.0, "approved_only": a.approved_only,
              "include_negative": True, "labels_colocate": False, "labels_only": a.labels_only}
    if a.format == "yolo-seg":
        res = api.post("/api/export/yolo", common)
    else:
        res = api.post("/api/export/bbox", {**common, "format": a.format})
    return {"format": a.format, "output_dir": str(out_dir), **res}


# ---------------------------------------------------------------- argparse

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autoseg", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default=BACKEND, help=f"backend URL (default {BACKEND}, env AUTOSEG_BACKEND)")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    s = sub.add_parser("status", help="backend, sam_server and project state")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("up", help="start sam_server and/or the backend if they are down")
    s.add_argument("--no-sam", action="store_true", help="do not start sam_server")
    s.set_defaults(fn=cmd_up)

    s = sub.add_parser("project", help="open or create a project")
    ps = s.add_subparsers(dest="sub", required=True, metavar="SUB")
    o = ps.add_parser("open", help="open a project dir (has autoseg.db) or create one from an image folder")
    o.add_argument("path", help="project dir or image folder")
    o.add_argument("--name", help="project name when creating (default: folder name)")
    o.add_argument("--project-dir", help="where to create the project (default .autoseg/projects/<name>)")
    o.set_defaults(fn=cmd_project_open)

    s = sub.add_parser("labels", help="list labels; --add NAME creates one")
    s.add_argument("--add", metavar="NAME")
    s.set_defaults(fn=cmd_labels)

    s = sub.add_parser("view", help="render one frame with overlays to a PNG")
    s.add_argument("frame", type=int)
    s.add_argument("--track", type=int, help="only this track's masks")
    s.add_argument("--grid", action="store_true", help="draw a 50 px pixel-coordinate grid")
    s.add_argument("--preview", help="overlay a segment preview")
    s.add_argument("--out", help="output folder (default .autoseg/views)")
    s.set_defaults(fn=cmd_view)

    s = sub.add_parser("sheet", help="contact sheet of many frames (status-coloured with --track)")
    s.add_argument("--track", type=int)
    s.add_argument("--every", type=int, help="every Nth frame (default: about 24 thumbnails)")
    s.add_argument("--frames", help="explicit frames, e.g. 100-110,140")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_sheet)

    s = sub.add_parser("segment", help="preview a mask from a box (+ optional clicks); nothing is saved")
    s.add_argument("frame", type=int)
    s.add_argument("--box", required=True, help="x0,y0,x1,y1 in pixels")
    s.add_argument("--pos", action="append", metavar="X,Y", help="positive click (repeatable)")
    s.add_argument("--neg", action="append", metavar="X,Y", help="negative click (repeatable)")
    s.add_argument("--no-grid", action="store_true", help="render the preview without the grid")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_segment)

    s = sub.add_parser("track", help="track commands")
    ts = s.add_subparsers(dest="sub", required=True, metavar="SUB")

    t = ts.add_parser("create", help="create a track from an approved preview (first keyframe)")
    t.add_argument("--label", required=True, help="label name (created if missing)")
    t.add_argument("--preview", required=True)
    t.add_argument("--name")
    t.set_defaults(fn=cmd_track_create)

    t = ts.add_parser("propagate", help="propagate keyframes through the video, then list issues")
    t.add_argument("track", type=int)
    t.add_argument("--full", action="store_true", help="regenerate every frame, not just the dirty range")
    t.add_argument("--start", type=int); t.add_argument("--end", type=int)
    t.add_argument("--no-wait", action="store_true", help="return right after starting the job")
    t.set_defaults(fn=cmd_track_propagate)

    t = ts.add_parser("issues", help="list health issues of a track")
    t.add_argument("track", type=int)
    t.add_argument("--verbose", action="store_true", help="include raw signals")
    t.set_defaults(fn=cmd_track_issues)

    t = ts.add_parser("refine", help="add/replace a keyframe at FRAME, then re-propagate")
    t.add_argument("track", type=int); t.add_argument("frame", type=int)
    t.add_argument("--preview", help="preview id from `segment` on the same frame")
    t.add_argument("--box", help="x0,y0,x1,y1 (segments directly, no preview check)")
    t.add_argument("--pos", action="append", metavar="X,Y"); t.add_argument("--neg", action="append", metavar="X,Y")
    t.add_argument("--no-propagate", action="store_true", help="batch several edits, propagate later")
    t.set_defaults(fn=cmd_track_refine)

    t = ts.add_parser("keyframes", help="list keyframes; --rm FRAME deletes one and re-propagates")
    t.add_argument("track", type=int)
    t.add_argument("--rm", type=int, metavar="FRAME")
    t.add_argument("--no-propagate", action="store_true")
    t.set_defaults(fn=cmd_track_keyframes)

    t = ts.add_parser("absent", help="mark frames where the object is not visible, e.g. 120-139,150")
    t.add_argument("track", type=int); t.add_argument("frames")
    t.add_argument("--no-propagate", action="store_true")
    t.set_defaults(fn=cmd_track_absent)

    t = ts.add_parser("list", help="tracks with keyframes, absent ranges, coverage, issue count")
    t.set_defaults(fn=cmd_track_list)

    s = sub.add_parser("export", help="export labels")
    s.add_argument("--format", choices=["yolo-seg", "yolo-detect", "coco"], default="yolo-seg")
    s.add_argument("--out", help="output folder (default .autoseg/export/<project>_<format>)")
    s.add_argument("--approved-only", action="store_true", help="skip frames health flagged for review")
    s.add_argument("--labels-only", action="store_true", help="do not copy images")
    s.set_defaults(fn=cmd_export)
    return p


def _log(argv, code, seconds):
    try:
        HOME.mkdir(parents=True, exist_ok=True)
        with open(HOME / "cli_log.jsonl", "a") as f:
            f.write(json.dumps({"ts": time.time(), "argv": argv, "exit": code, "seconds": round(seconds, 2)}) + "\n")
    except OSError:
        pass


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    t0 = time.time()
    parser = build_parser()
    try:
        a = parser.parse_args(argv)
    except SystemExit as e:
        _log(argv, e.code, time.time() - t0)
        raise
    code = 0
    try:
        result = a.fn(Api(a.backend), a)
        print(json.dumps({"ok": True, **result}, indent=1))
    except CliError as e:
        code = e.code
        print(json.dumps({"ok": False, "error": str(e), **e.extra}))
    except KeyboardInterrupt:
        code = 130
        print(json.dumps({"ok": False, "error": "interrupted"}))
    _log(argv, code, time.time() - t0)
    return code


if __name__ == "__main__":
    sys.exit(main())
