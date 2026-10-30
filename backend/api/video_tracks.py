"""SAM3 video tracking API (WP4).

The router deliberately keeps the HTTP surface small: previews live in a bounded
in-process cache, while keyframes, absent frames, propagated annotations and
health issues are persisted in the project database.
"""
from __future__ import annotations

import io
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from backend.api.projects import get_project, get_store
from core.masks import mask_area, mask_to_bbox, mask_to_rle, rle_to_mask
from core.polygons import mask_to_yolo_polygon
from ml.sam3_video import FrameSet, Sam3Client, TrackEngine
from ml.sam3_video.health import HealthConfig, assess

router = APIRouter()


@dataclass
class Preview:
    id: str
    project_id: int
    frame: int
    mask: np.ndarray
    score: float
    bbox: list[float]
    created: float = field(default_factory=time.time)


class VideoTrackJob:
    """Background TrackEngine run with the same polling event shape as track_jobs."""
    def __init__(self, project_id: int, track_id: int, engine: TrackEngine,
                 store_db: str, frame_range: Optional[tuple[int, int]] = None,
                 full: bool = False):
        self.id = uuid.uuid4().hex[:12]
        self.project_id, self.track_id = project_id, track_id
        self.engine, self.store_db = engine, store_db
        self.frame_range, self.full = frame_range, full
        self.status = "pending"
        self.error: Optional[str] = None
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.counts = {"frames": 0, "issues": 0}
        self._events: list[dict] = []
        self._seq = 0
        self._cond = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self.run, daemon=True, name=f"video-track-{self.id}")

    @property
    def is_active(self):
        return self.status not in {"finished", "stopped", "error"}

    @property
    def last_seq(self):
        return self._seq

    def emit(self, typ: str, **data):
        with self._cond:
            self._seq += 1
            ev = {"seq": self._seq, "type": typ, "job_id": self.id, "track_id": self.track_id,
                  "ts": time.time(), **data}
            self._events.append(ev)
            self._cond.notify_all()
        return ev

    def events_after(self, after=0):
        with self._cond:
            return [e for e in self._events if e["seq"] > after]

    def wait_events(self, after=0, timeout=15.0):
        end = time.time() + timeout
        with self._cond:
            while True:
                out = [e for e in self._events if e["seq"] > after]
                if out or not self.is_active or time.time() >= end:
                    return out
                self._cond.wait(max(0.01, end - time.time()))

    def snapshot(self):
        return {"id": self.id, "project_id": self.project_id, "track_id": self.track_id,
                "status": self.status, "frame_range": self.frame_range,
                "full": self.full, "counts": dict(self.counts), "error": self.error,
                "last_seq": self.last_seq, "created_at": self.started_at,
                "finished_at": self.finished_at}

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop = True

    def run(self):
        from core.store import ProjectStore
        store = ProjectStore(self.store_db)
        try:
            self.status = "running"
            lo, hi = self.frame_range or (0, self.engine.num_frames - 1)
            store.delete_propagated_annotations(self.track_id, lo, hi)
            store.clear_issues(self.track_id)
            self.emit("started", start_frame=lo, end_frame=hi)

            def on_event(ev):
                if self._stop:
                    return
                if ev.kind == "frame" and ev.mask is not None:
                    frame = int(ev.frame)
                    if frame < lo or frame > hi:
                        return
                    image = store.get_image_by_index(self.project_id, frame)
                    track = store.get_track(self.track_id)
                    if image is None or track is None:
                        return
                    m = np.asarray(ev.mask, dtype=bool)
                    ann = store.create_annotation(
                        image.id, track.label_id, mask_to_bbox(m), source="propagated",
                        status="approved", confidence=None, mask_rle=mask_to_rle(m),
                        polygon_norm=mask_to_yolo_polygon(m, image.width, image.height),
                        track_id=self.track_id,
                    )
                    self.counts["frames"] += 1
                    self.emit("frame", frame=frame, annotation_id=ann.id,
                              progress=self.counts["frames"])
                elif ev.kind in {"start", "segment"}:
                    self.emit(ev.kind, frame=getattr(ev, "frame", None),
                              done=ev.done, total=ev.total)

            self.engine.run(track_ids=[self.track_id], full=self.full,
                            frame_range=self.frame_range, on_event=on_event,
                            should_stop=lambda: self._stop)
            if self._stop:
                self.status = "stopped"
                self.emit("stopped")
                return

            health, issues = assess(self.engine, self.track_id, HealthConfig.geometry_only())
            health_by_frame = {h.frame: h for h in health}
            # A frame is approved only when health says ok; keyframe/absent are
            # deliberately excluded from the propagated table.
            for ann, image in store.list_track_annotations(self.track_id):
                if ann.source != "propagated" or not (lo <= image.order_index <= hi):
                    continue
                h = health_by_frame.get(image.order_index)
                if h is not None:
                    store.update_annotation(ann.id, status="approved" if h.status == "ok" else "needs_review")
            for issue in issues:
                d = issue.to_dict()
                store.create_issue(self.track_id, d, self.id)
                self.counts["issues"] += 1
                self.emit("issue", issue=d)
            self.status = "finished"
            self.finished_at = time.time()
            self.emit("finished", counts=dict(self.counts))
        except Exception as exc:  # pragma: no cover - exercised by integration failures
            self.error = str(exc)
            self.status = "error"
            self.finished_at = time.time()
            self.emit("error", message=self.error)
        finally:
            store.close()


_PREVIEWS: OrderedDict[str, Preview] = OrderedDict()
_PREVIEW_LIMIT = int(os.getenv("VIDEO_PREVIEW_LIMIT", "64"))
_PREVIEW_TTL = float(os.getenv("VIDEO_PREVIEW_TTL", "1800"))
_ENGINES: dict[str, TrackEngine] = {}
_ENGINE_LOCK = threading.RLock()
_JOBS: dict[str, VideoTrackJob] = {}
_JOBS_LOCK = threading.RLock()


def _client() -> Sam3Client:
    return Sam3Client(os.getenv("SAM_SERVER_URL", "http://localhost:8770"))


def _engine() -> TrackEngine:
    project, store = get_project(), get_store()
    with _ENGINE_LOCK:
        key = project.db_path
        eng = _ENGINES.get(key)
        if eng is not None and eng.num_frames == store.get_image_count(project.id):
            return eng
        if eng is not None:
            eng.close()
        images = store.list_images(project.id)
        frames = FrameSet([i.path for i in images], root=str(Path(project.root_dir) / ".autoseg" / "frames"))
        eng = TrackEngine(_client(), frames)
        _ENGINES[key] = eng
        return eng


def _track_loaded(engine: TrackEngine, track_id: int):
    if track_id in engine.tracks:
        return
    store = get_store()
    track = store.get_track(track_id)
    if track is None:
        raise HTTPException(404, "Track not found")
    engine.add_track(track_id)
    for ann, image in store.list_track_annotations(track_id):
        if ann.source == "propagated" or ann.mask_rle is None:
            continue
        try:
            engine.add_keyframe(track_id, image.order_index, rle_to_mask(ann.mask_rle).astype(bool), source=ann.source)
        except (ValueError, IndexError):
            continue
    absent = store.list_track_absences(track_id)
    if absent:
        engine.mark_absent(track_id, absent)


def _keyframe_dirty_range(engine: TrackEngine, track_id: int, frame: int) -> tuple[int, int]:
    """Frames owned by a newly inserted keyframe (inclusive)."""
    t = engine.track(track_id)
    positives = sorted(t.positive_keyframes())
    pos = positives.index(frame) if frame in positives else 0
    lo = ((positives[pos - 1] + frame) // 2 + 1) if pos else 0
    hi = ((frame + positives[pos + 1]) // 2) if pos + 1 < len(positives) else engine.num_frames - 1
    return lo, hi


def _get_preview(pid: str) -> Preview:
    p = _PREVIEWS.get(pid)
    if p is None or time.time() - p.created > _PREVIEW_TTL:
        _PREVIEWS.pop(pid, None)
        raise HTTPException(404, "Preview not found or expired")
    _PREVIEWS.move_to_end(pid)
    return p


class SegmentRequest(BaseModel):
    frame: int
    box: list[float] = Field(min_length=4, max_length=4)
    pos: list[list[float]] = Field(default_factory=list)
    neg: list[list[float]] = Field(default_factory=list)


class CreateTrackRequest(BaseModel):
    preview_id: str
    label_id: Optional[int] = None
    label: Optional[str] = None
    name: Optional[str] = None


class PropagateRequest(BaseModel):
    full: bool = False
    start: Optional[int] = None
    end: Optional[int] = None


class KeyframeRequest(BaseModel):
    frame: int
    preview_id: Optional[str] = None
    box: Optional[list[float]] = None
    pos: list[list[float]] = Field(default_factory=list)
    neg: list[list[float]] = Field(default_factory=list)


class AbsentRequest(BaseModel):
    frames: Optional[list[int]] = None
    start: Optional[int] = None
    end: Optional[int] = None


def _image_for_frame(frame: int):
    image = get_store().get_image_by_index(get_project().id, frame)
    if image is None:
        raise HTTPException(404, f"Frame {frame} not found")
    return image


@router.post("/segment")
def segment_preview(req: SegmentRequest):
    image = _image_for_frame(req.frame)
    client = _client()
    try:
        mask, score = client.segment(image.path, box=req.box,
                                     points=req.pos + req.neg,
                                     point_labels=[1] * len(req.pos) + [0] * len(req.neg))
    except Exception as exc:
        raise HTTPException(502, f"sam_server segmentation failed: {exc}")
    finally:
        close = getattr(client, "close", None)
        if close:
            close()
    p = Preview(uuid.uuid4().hex[:16], get_project().id, req.frame, mask.astype(bool), float(score), mask_to_bbox(mask))
    _PREVIEWS[p.id] = p
    while len(_PREVIEWS) > _PREVIEW_LIMIT:
        _PREVIEWS.popitem(last=False)
    return {"preview_id": p.id, "frame": p.frame, "bbox": p.bbox,
            "area": mask_area(p.mask), "score": p.score}


@router.post("/tracks")
def create_video_track(req: CreateTrackRequest):
    p = _get_preview(req.preview_id)
    project, store = get_project(), get_store()
    if p.project_id != project.id:
        raise HTTPException(400, "Preview belongs to another project")
    image = _image_for_frame(p.frame)
    label = store.get_label_by_id(req.label_id) if req.label_id else None
    if label is None and req.label:
        label = store.upsert_label(project.id, req.label)
    if label is None:
        raise HTTPException(400, "label_id or label is required")
    track = store.create_track(project.id, label.id, name=req.name)
    ann = store.create_annotation(image.id, label.id, p.bbox, source="sam", status="approved",
                                  mask_rle=mask_to_rle(p.mask),
                                  polygon_norm=mask_to_yolo_polygon(p.mask, image.width, image.height), track_id=track.id)
    eng = _engine(); _track_loaded(eng, track.id)
    if track.id not in eng.tracks:
        eng.add_track(track.id)
    eng.add_keyframe(track.id, p.frame, p.mask, source="sam")
    eng._video_dirty_range = (0, eng.num_frames - 1)
    return {"track_id": track.id, "keyframe": {"frame": p.frame, "annotation_id": ann.id,
            "bbox": p.bbox, "area": mask_area(p.mask)}, "label_id": label.id}


@router.post("/tracks/{track_id}/propagate")
def start_propagate(track_id: int, req: PropagateRequest = PropagateRequest()):
    project, store = get_project(), get_store()
    if store.get_track(track_id) is None:
        raise HTTPException(404, "Track not found")
    with _JOBS_LOCK:
        if any(j.is_active and j.project_id == project.id for j in _JOBS.values()):
            raise HTTPException(409, "A video tracking job is already running for this project")
        eng = _engine(); _track_loaded(eng, track_id)
        if not eng.track(track_id).positive_keyframes():
            raise HTTPException(400, "Track has no keyframes")
        dirty = getattr(eng, "_video_dirty_range", None)
        lo = req.start if req.start is not None else (0 if req.full or dirty is None else dirty[0])
        hi = req.end if req.end is not None else (eng.num_frames - 1 if req.full or dirty is None else dirty[1])
        if lo < 0 or hi >= eng.num_frames or lo > hi:
            raise HTTPException(400, "Invalid frame range")
        job = VideoTrackJob(project.id, track_id, eng, store.db_path,
                            frame_range=(lo, hi), full=req.full)
        _JOBS[job.id] = job
        eng._video_dirty_range = None
        job.start()
    return job.snapshot()


def _job_or_404(job_id: str) -> VideoTrackJob:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Video tracking job not found")
    return job


@router.get("/jobs/{job_id}")
def video_job_status(job_id: str):
    return _job_or_404(job_id).snapshot()


@router.get("/jobs/{job_id}/events")
def video_job_events(job_id: str, after: int = 0, wait: float = 0.0):
    job = _job_or_404(job_id)
    events = job.wait_events(after, min(wait, 30)) if wait > 0 else job.events_after(after)
    return {"events": events, "status": job.status, "last_seq": job.last_seq}


@router.get("/tracks/{track_id}/issues")
def track_issues(track_id: int):
    if get_store().get_track(track_id) is None:
        raise HTTPException(404, "Track not found")
    return get_store().list_issues(track_id)


@router.post("/tracks/{track_id}/keyframes")
def add_video_keyframe(track_id: int, req: KeyframeRequest):
    eng = _engine(); _track_loaded(eng, track_id)
    image = _image_for_frame(req.frame)
    if req.preview_id:
        p = _get_preview(req.preview_id)
        if p.frame != req.frame:
            raise HTTPException(400, "Preview frame does not match keyframe frame")
        mask, score = p.mask, p.score
    elif req.box:
        c = _client()
        try:
            mask, score = c.segment(image.path, box=req.box, points=req.pos + req.neg,
                                    point_labels=[1] * len(req.pos) + [0] * len(req.neg))
        finally:
            close = getattr(c, "close", None)
            if close:
                close()
        mask = mask.astype(bool)
    else:
        raise HTTPException(400, "preview_id or box is required")
    store = get_store(); track = store.get_track(track_id)
    store.delete_propagated_annotations(track_id, req.frame, req.frame)
    # Replace an existing non-propagated annotation at this frame.
    for ann, img in store.list_track_annotations(track_id):
        if img.order_index == req.frame and ann.source != "propagated":
            store.delete_annotation(ann.id)
    ann = store.create_annotation(image.id, track.label_id, mask_to_bbox(mask), source="sam", status="approved",
                                  confidence=float(score), mask_rle=mask_to_rle(mask),
                                  polygon_norm=mask_to_yolo_polygon(mask, image.width, image.height), track_id=track_id)
    eng.add_keyframe(track_id, req.frame, mask, source="sam")
    eng._video_dirty_range = _keyframe_dirty_range(eng, track_id, req.frame)
    return {"track_id": track_id, "frame": req.frame, "annotation_id": ann.id, "score": float(score),
            "bbox": mask_to_bbox(mask)}


@router.delete("/tracks/{track_id}/keyframes/{frame}")
def remove_video_keyframe(track_id: int, frame: int):
    eng = _engine(); _track_loaded(eng, track_id)
    if frame not in eng.track(track_id).keyframes:
        raise HTTPException(404, "Keyframe not found")
    eng.remove_keyframe(track_id, frame)
    eng._video_dirty_range = (0, eng.num_frames - 1)
    store = get_store()
    for ann, img in store.list_track_annotations(track_id):
        if img.order_index == frame and ann.source != "propagated":
            store.delete_annotation(ann.id)
    return {"track_id": track_id, "frame": frame, "removed": True}


@router.post("/tracks/{track_id}/absent")
def mark_track_absent(track_id: int, req: AbsentRequest):
    eng = _engine(); _track_loaded(eng, track_id)
    frames = req.frames or (list(range(req.start, req.end + 1)) if req.start is not None and req.end is not None else [])
    if not frames:
        raise HTTPException(400, "frames or start/end is required")
    try:
        eng.mark_absent(track_id, frames)
    except (ValueError, IndexError) as exc:
        raise HTTPException(400, str(exc))
    store = get_store(); store.add_track_absences(track_id, frames)
    store.delete_propagated_annotations(track_id, min(frames), max(frames))
    eng._video_dirty_range = (min(frames), max(frames))
    return {"track_id": track_id, "frames": sorted(set(frames)), "status": "absent"}


@router.get("/tracks")
def list_video_tracks():
    project, store = get_project(), get_store()
    out = []
    for track in store.list_tracks(project.id):
        anns = store.list_track_annotations(track.id)
        anchors = [img.order_index for ann, img in anns if ann.source != "propagated"]
        propagated = [img.order_index for ann, img in anns if ann.source == "propagated"]
        job = next((j for j in reversed(list(_JOBS.values())) if j.track_id == track.id), None)
        out.append({"id": track.id, "label_id": track.label_id, "status": track.status,
                    "keyframes": sorted(anchors), "absent": store.list_track_absences(track.id),
                    "coverage": {"first": min(propagated) if propagated else None,
                                 "last": max(propagated) if propagated else None,
                                 "count": len(propagated)},
                    "issue_count": len(store.list_issues(track.id)),
                    "job": job.snapshot() if job else None})
    return out


def _overlay(path: str, track_id: Optional[int], frame: int, grid: bool, preview_id: Optional[str]):
    base = Image.open(path).convert("RGBA")
    store = get_store(); anns = store.list_annotations(_image_for_frame(frame).id)
    if track_id is not None:
        anns = [a for a in anns if a.track_id == track_id]
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    colors = [(40, 180, 255, 100), (255, 90, 80, 100), (100, 230, 120, 100)]
    for i, ann in enumerate(anns):
        if not ann.mask_rle:
            continue
        m = rle_to_mask(ann.mask_rle).astype(bool)
        a = np.zeros((m.shape[0], m.shape[1], 4), np.uint8); a[m] = colors[i % len(colors)]
        overlay = Image.alpha_composite(overlay, Image.fromarray(a, "RGBA"))
    if preview_id:
        p = _get_preview(preview_id); m = p.mask
        a = np.zeros((m.shape[0], m.shape[1], 4), np.uint8); a[m] = (255, 220, 20, 130)
        overlay = Image.alpha_composite(overlay, Image.fromarray(a, "RGBA"))
    out = Image.alpha_composite(base, overlay)
    d = ImageDraw.Draw(out)
    d.rectangle((4, 4, 180, 28), fill=(0, 0, 0, 170)); d.text((8, 8), f"frame {frame}", fill="white")
    for ann in anns:
        if ann.bbox_xyxy:
            b = ann.bbox_xyxy; col = "yellow" if ann.source != "propagated" else "cyan"
            d.rectangle(b, outline=col, width=2)
    if grid:
        for x in range(0, out.width, 50):
            d.line((x, 0, x, out.height), fill=(255, 255, 255, 100), width=1); d.text((x + 2, 2), str(x), fill="white")
        for y in range(0, out.height, 50):
            d.line((0, y, out.width, y), fill=(255, 255, 255, 100), width=1); d.text((2, y + 2), str(y), fill="white")
    buf = io.BytesIO(); out.convert("RGB").save(buf, format="PNG"); return buf.getvalue()


@router.get("/render/frame")
def render_frame(frame: int, track_id: Optional[int] = None, grid: bool = False, preview_id: Optional[str] = None):
    image = _image_for_frame(frame)
    return Response(_overlay(image.path, track_id, frame, grid, preview_id), media_type="image/png")


@router.get("/render/sheet")
def render_sheet(track_id: Optional[int] = None, every: int = Query(10, ge=1), frames: Optional[str] = None,
                 grid: bool = False):
    n = get_store().get_image_count(get_project().id)
    selected = sorted({int(x) for x in frames.split(",")}) if frames else list(range(0, n, every))
    thumbs = []
    for f in selected:
        if 0 <= f < n:
            im = Image.open(io.BytesIO(_overlay(_image_for_frame(f).path, track_id, f, grid, None))).convert("RGB")
            im.thumbnail((320, 240)); thumbs.append((f, im.copy()))
    cols = 4; cell_w, cell_h = 320, 270
    sheet = Image.new("RGB", (cols * cell_w, max(1, (len(thumbs) + cols - 1) // cols) * cell_h), "#222")
    d = ImageDraw.Draw(sheet)
    for i, (f, im) in enumerate(thumbs):
        x, y = (i % cols) * cell_w, (i // cols) * cell_h
        sheet.paste(im, (x, y + 22)); d.text((x + 5, y + 4), f"frame {f}", fill="white")
    buf = io.BytesIO(); sheet.save(buf, format="PNG"); return Response(buf.getvalue(), media_type="image/png")
