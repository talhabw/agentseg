"""
Tracks API - object identity across frames (issue #4).

A track is one physical object. Annotations reference it via track_id.
Anchors are annotations with status='approved' (human/agent-approved); a track
re-initialises from its most recent anchor.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.annotations import AnnotationResponse, annotation_to_response
from backend.api.projects import get_project, get_store
from core.db import TRACK_STATUSES

router = APIRouter()


class TrackResponse(BaseModel):
    id: int
    project_id: int
    label_id: int
    name: Optional[str] = None
    notes: Optional[str] = None
    status: str
    lost_at_image_id: Optional[int] = None
    annotation_count: int = 0
    anchor_count: int = 0
    first_image_index: Optional[int] = None
    last_image_index: Optional[int] = None


class CreateTrackRequest(BaseModel):
    label_id: int
    name: Optional[str] = None
    notes: Optional[str] = None


class UpdateTrackRequest(BaseModel):
    name: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[str] = None
    label_id: Optional[int] = None


class MarkLostRequest(BaseModel):
    image_id: Optional[int] = None  # frame where the object was lost


class TrackFrame(BaseModel):
    image_id: int
    image_index: int
    annotation: AnnotationResponse


class TrackAnchorResponse(BaseModel):
    found: bool
    annotation: Optional[AnnotationResponse] = None
    image_index: Optional[int] = None


class ReanchorRequest(BaseModel):
    image_id: int
    # Either adopt an existing annotation on that frame...
    annotation_id: Optional[int] = None
    # ...or create a new anchor from geometry.
    bbox: Optional[list[float]] = None
    mask_rle: Optional[dict] = None
    polygon: Optional[list[float]] = None
    source: str = "manual"
    confidence: Optional[float] = None
    # Remove other annotations of this track on that frame (e.g. a drifted one).
    replace_existing: bool = True


def _track_response(track, summary: Optional[dict] = None) -> TrackResponse:
    summary = summary or {}
    return TrackResponse(
        id=track.id,
        project_id=track.project_id,
        label_id=track.label_id,
        name=track.name,
        notes=track.notes,
        status=track.status,
        lost_at_image_id=track.lost_at_image_id,
        annotation_count=summary.get("annotation_count", 0) or 0,
        anchor_count=summary.get("anchor_count", 0) or 0,
        first_image_index=summary.get("first_image_index"),
        last_image_index=summary.get("last_image_index"),
    )


def _get_track_or_404(store, track_id: int):
    track = store.get_track(track_id)
    if track is None:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")
    return track


def _single_summary(store, track) -> dict:
    return store.track_summaries(track.project_id).get(track.id, {})


@router.get("", response_model=list[TrackResponse])
async def list_tracks(
    project_id: Optional[int] = None,
    label_id: Optional[int] = None,
    status: Optional[str] = None,
):
    """List tracks (defaults to the current project)."""
    store = get_store()
    pid = project_id if project_id is not None else get_project().id
    summaries = store.track_summaries(pid)
    return [
        _track_response(t, summaries.get(t.id))
        for t in store.list_tracks(pid, label_id=label_id, status=status)
    ]


@router.post("", response_model=TrackResponse)
async def create_track(request: CreateTrackRequest):
    store = get_store()
    project = get_project()
    if store.get_label_by_id(request.label_id) is None:
        raise HTTPException(status_code=404, detail="Label not found")
    track = store.create_track(project.id, request.label_id, request.name, request.notes)
    return _track_response(track)


@router.get("/{track_id}", response_model=TrackResponse)
async def get_track(track_id: int):
    store = get_store()
    track = _get_track_or_404(store, track_id)
    return _track_response(track, _single_summary(store, track))


@router.patch("/{track_id}", response_model=TrackResponse)
async def update_track(track_id: int, request: UpdateTrackRequest):
    store = get_store()
    _get_track_or_404(store, track_id)
    fields = request.model_dump(exclude_none=True)
    if "status" in fields and fields["status"] not in TRACK_STATUSES:
        raise HTTPException(status_code=400, detail=f"status must be one of {TRACK_STATUSES}")
    if fields.get("status") == "active":
        fields["lost_at_image_id"] = None
    track = store.update_track(track_id, **fields)
    return _track_response(track, _single_summary(store, track))


@router.delete("/{track_id}")
async def delete_track(track_id: int):
    """Delete a track. Its annotations are kept but become untracked."""
    store = get_store()
    _get_track_or_404(store, track_id)
    store.delete_track(track_id)
    return {"status": "deleted", "id": track_id}


@router.get("/{track_id}/annotations", response_model=list[TrackFrame])
async def get_track_annotations(track_id: int):
    """A track's annotations ordered by frame (each has is_anchor)."""
    store = get_store()
    _get_track_or_404(store, track_id)
    return [
        TrackFrame(image_id=img.id, image_index=img.order_index, annotation=annotation_to_response(ann))
        for ann, img in store.list_track_annotations(track_id)
    ]


@router.post("/{track_id}/lost", response_model=TrackResponse)
async def mark_track_lost(track_id: int, request: MarkLostRequest):
    store = get_store()
    _get_track_or_404(store, track_id)
    track = store.update_track(track_id, status="lost", lost_at_image_id=request.image_id)
    return _track_response(track, _single_summary(store, track))


@router.post("/{track_id}/end", response_model=TrackResponse)
async def mark_track_ended(track_id: int):
    store = get_store()
    _get_track_or_404(store, track_id)
    track = store.update_track(track_id, status="ended")
    return _track_response(track, _single_summary(store, track))


@router.get("/{track_id}/anchor", response_model=TrackAnchorResponse)
async def find_track_anchor(
    track_id: int,
    before_image_index: Optional[int] = None,
    exclude_image_ids: Optional[str] = None,
):
    """Most recent anchor (approved annotation) of this track before a frame."""
    store = get_store()
    _get_track_or_404(store, track_id)
    excluded = (
        {int(x) for x in exclude_image_ids.split(",") if x.strip()}
        if exclude_image_ids
        else set()
    )
    found = store.find_track_anchor(track_id, before_image_index, excluded)
    if found is None:
        return TrackAnchorResponse(found=False)
    ann, img = found
    return TrackAnchorResponse(
        found=True, annotation=annotation_to_response(ann), image_index=img.order_index
    )


@router.post("/{track_id}/reanchor", response_model=AnnotationResponse)
async def reanchor_track(track_id: int, request: ReanchorRequest):
    """Re-anchor a track on a frame: the result is an approved anchor annotation
    with this track_id, and the track becomes active again."""
    store = get_store()
    track = _get_track_or_404(store, track_id)
    if store.get_image_by_id(request.image_id) is None:
        raise HTTPException(status_code=404, detail="Image not found")

    if request.annotation_id is not None:
        existing = store.get_annotation_by_id(request.annotation_id)
        if existing is None or existing.image_id != request.image_id:
            raise HTTPException(status_code=404, detail="Annotation not found on that image")
        keep_id = existing.id
    elif request.bbox is not None:
        keep_id = None
    else:
        raise HTTPException(status_code=400, detail="Provide annotation_id or bbox")

    if request.replace_existing:
        for other in store.list_annotations(request.image_id):
            if other.track_id == track_id and other.id != keep_id:
                store.delete_annotation(other.id)

    if keep_id is not None:
        ann = store.update_annotation(
            keep_id, track_id=track_id, status="approved", label_id=track.label_id
        )
    else:
        ann = store.create_annotation(
            image_id=request.image_id,
            label_id=track.label_id,
            bbox_xyxy=request.bbox,
            source=request.source,
            status="approved",
            confidence=request.confidence,
            mask_rle=request.mask_rle,
            polygon_norm=request.polygon,
            track_id=track_id,
        )

    store.update_track(track_id, status="active", lost_at_image_id=None)
    return annotation_to_response(ann)
