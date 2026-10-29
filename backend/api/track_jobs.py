"""Tracking job API (issue #3): run frame-to-frame tracking in the backend.

POST /api/track/jobs                 start a job
GET  /api/track/jobs                 list jobs (?active=true for the running one)
GET  /api/track/jobs/{id}            status + progress
POST /api/track/jobs/{id}/stop       stop (also releases a paused job)
POST /api/track/jobs/{id}/resume     resume a paused job
GET  /api/track/jobs/{id}/events     polling: events with seq > ?after=
GET  /api/track/jobs/{id}/stream     SSE stream (resumes from ?after= / Last-Event-ID)
"""

import json
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.api import ml as ml_api
from backend.api.projects import get_project, get_store
from ml.propagate import (
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_SIZE_MAX_RATIO,
    DEFAULT_SIZE_MIN_RATIO,
)
from ml.tracking_job import (
    FAILURE_MODES,
    ServicePropagator,
    TrackingJob,
    TrackingSettings,
    get_job_manager,
)

router = APIRouter()


class StartJobRequest(BaseModel):
    # Range: by index or by image id; end is inclusive (default: last image)
    start_index: Optional[int] = None
    start_image_id: Optional[int] = None
    end_index: Optional[int] = None
    num_frames: Optional[int] = None  # alternative to end_index (1 = single step)
    # Object selection: explicit tracks, or all objects (optionally label-filtered)
    track_ids: Optional[list[int]] = None
    label_ids: Optional[list[int]] = None
    # Settings
    failure_mode: str = "stop"  # "stop" | "skip" | "continue"
    performance: bool = False
    mode: str = "auto"
    iou_verify: bool = True
    iou_threshold: float = 0.3
    use_cached_masks: bool = True
    use_bbox_hint: bool = True
    bbox_hint_scale: float = 1.15
    prune_thin_artifacts: bool = True
    size_min_ratio: float = DEFAULT_SIZE_MIN_RATIO
    size_max_ratio: float = DEFAULT_SIZE_MAX_RATIO
    top_k: int = 5
    min_similarity: float = DEFAULT_MIN_SIMILARITY
    duplicate_threshold: float = 0.75
    max_fallback_attempts: int = 2
    use_anchor: bool = True


def _settings(req: StartJobRequest) -> TrackingSettings:
    return TrackingSettings(
        mode=req.mode,
        iou_verify=req.iou_verify,
        iou_threshold=req.iou_threshold,
        use_cached_masks=req.use_cached_masks,
        use_bbox_hint=req.use_bbox_hint,
        bbox_hint_scale=req.bbox_hint_scale,
        prune_thin_artifacts=req.prune_thin_artifacts,
        size_min_ratio=req.size_min_ratio,
        size_max_ratio=req.size_max_ratio,
        top_k=req.top_k,
        min_similarity=req.min_similarity,
        duplicate_threshold=req.duplicate_threshold,
        failure_mode=req.failure_mode,
        performance=req.performance,
        max_fallback_attempts=req.max_fallback_attempts,
        use_anchor=req.use_anchor,
    )


def _job_or_404(job_id: str) -> TrackingJob:
    job = get_job_manager().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Tracking job not found")
    return job


def _propagate_fn():
    """Factory hook (tests replace this with a stub)."""
    service = ml_api._propagate_service
    if service is None:
        raise HTTPException(
            status_code=400,
            detail="Propagation not loaded. Call /api/ml/propagate/load first",
        )
    return ServicePropagator(service)


@router.post("/jobs")
def start_job(req: StartJobRequest):
    store = get_store()
    project = get_project()
    if req.failure_mode not in FAILURE_MODES:
        raise HTTPException(status_code=400, detail=f"failure_mode must be one of {sorted(FAILURE_MODES)}")

    start = req.start_index
    if req.start_image_id is not None:
        img = store.get_image_by_id(req.start_image_id)
        if img is None:
            raise HTTPException(status_code=404, detail="Start image not found")
        start = img.order_index
    if start is None:
        raise HTTPException(status_code=400, detail="Provide start_index or start_image_id")
    end = req.end_index
    if req.num_frames is not None:
        end = start + max(1, req.num_frames)

    job = TrackingJob(
        db_path=store.db_path,
        project_id=project.id,
        propagate_fn=_propagate_fn(),
        start_index=start,
        end_index=end,
        settings=_settings(req),
        track_ids=req.track_ids,
        label_ids=req.label_ids,
    )
    try:
        get_job_manager().submit(job)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return job.snapshot()


@router.get("/jobs")
def list_jobs(active: bool = False):
    mgr = get_job_manager()
    if active:
        try:
            db_path = get_store().db_path
        except HTTPException:
            db_path = None
        job = mgr.active(db_path)
        return [job.snapshot()] if job else []
    return [j.snapshot() for j in mgr.list()]


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    return _job_or_404(job_id).snapshot()


@router.post("/jobs/{job_id}/stop")
def stop_job(job_id: str):
    job = _job_or_404(job_id)
    job.stop()
    return job.snapshot()


@router.post("/jobs/{job_id}/resume")
def resume_job(job_id: str):
    job = _job_or_404(job_id)
    if not job.resume():
        raise HTTPException(status_code=409, detail=f"Job is not paused (status: {job.status})")
    return job.snapshot()


@router.get("/jobs/{job_id}/events")
def job_events(job_id: str, after: int = 0, wait: float = 0.0):
    """Polling fallback. ``wait`` > 0 long-polls up to that many seconds."""
    job = _job_or_404(job_id)
    events = (
        job.wait_events(after, timeout=min(wait, 30.0)) if wait > 0 else job.events_after(after)
    )
    return {"events": events, "status": job.status, "last_seq": job.last_seq}


@router.get("/jobs/{job_id}/stream")
def job_stream(
    job_id: str,
    after: int = 0,
    last_event_id: Optional[str] = Header(default=None, alias="Last-Event-ID"),
):
    job = _job_or_404(job_id)
    if last_event_id and last_event_id.isdigit():
        after = max(after, int(last_event_id))

    def gen():
        cursor = after
        while True:
            events = job.wait_events(cursor, timeout=15.0)
            for ev in events:
                cursor = ev["seq"]
                yield f"id: {ev['seq']}\nevent: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
            if not events:
                if not job.is_active:
                    break
                yield ": keepalive\n\n"
            elif not job.is_active and cursor >= job.last_seq:
                break

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
