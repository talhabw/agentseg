"""Backend tracking orchestration (issue #3).

``TrackingJob`` owns the frame-to-frame tracking loop that used to live in
the frontend: frame iteration, per-track propagation, duplicate / ID-switch
handling, fallback re-anchoring on earlier anchors of the same track, the
per-track area trend, and persistence of results (source ``tracked``).

Jobs run in a background thread with their own ``ProjectStore`` (sqlite
connections are bound to their creating thread) and publish a sequence of
JSON events that any client (web UI, CLI, agent) can follow:

    started, frame_done, uncertain, possible_id_switch, object_lost,
    paused_waiting_for_label, reanchored, object_resumed, resumed,
    finished, error

On a lost object the job either pauses as a whole (``failure_mode='stop'``)
or only drops that object and keeps going (``'skip'`` / ``'continue'``).
A client supplies a new label with ``POST /api/tracks/{id}/reanchor`` (or by
drawing a new box of the same label on the paused frame) and then resumes.

The duplicate / anchor helpers below are shared with the single-step
``/api/ml/propagate*`` endpoints.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from core.masks import mask_iou, mask_to_rle, rle_to_mask
from core.polygons import mask_to_yolo_polygon
from core.store import ProjectStore
from ml.propagate import (
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_SIZE_MAX_RATIO,
    DEFAULT_SIZE_MIN_RATIO,
    PropagationError,
    PropagationNotFoundError,
    PropagationResult,
    STATUS_NEEDS_REVIEW,
    update_area_trend,
)

logger = logging.getLogger(__name__)


# ==================== Shared tracking rules ====================


@dataclass
class DuplicateCheck:
    """Result of comparing a propagated result against existing target annotations.

    skipped: same object already annotated (same track, or untracked same-label,
        or a different-label annotation at the same place) -> don't save.
    possible_id_switch: overlaps a *different track* of the same label. Not
        dropped silently: callers should save it flagged for review.
    """

    skipped: bool = False
    iou: float = 0.0
    conflicting_label_name: str | None = None
    possible_id_switch: bool = False
    conflicting_track_id: Optional[int] = None
    conflicting_annotation_id: Optional[int] = None


def check_duplicate(
    store,
    existing_annotations,
    new_bbox: list[float],
    new_mask: Optional[np.ndarray],
    source_label_id: int,
    threshold: float,
    source_track_id: Optional[int] = None,
) -> DuplicateCheck:
    """Compare a propagated result against existing annotations, track-aware.

    Same-track (or untracked same-label) overlap -> duplicate, skip.
    Different-track same-label overlap -> possible ID switch, reported.
    Different-label overlap (stricter threshold) -> conflict, skip.
    """
    if threshold <= 0:
        return DuplicateCheck()

    same_label_threshold = threshold
    different_label_threshold = max(threshold, 0.9)
    id_switch: Optional[DuplicateCheck] = None

    for existing_ann in existing_annotations:
        score = annotation_overlap_score(existing_ann, new_bbox, new_mask)
        if score <= 0:
            continue

        is_same_label = existing_ann.label_id == source_label_id
        required_score = same_label_threshold if is_same_label else different_label_threshold
        if score < required_score:
            continue

        if is_same_label:
            other_track = getattr(existing_ann, "track_id", None)
            if (
                source_track_id is not None
                and other_track is not None
                and other_track != source_track_id
            ):
                logger.info(
                    "Possible ID switch: track %s overlaps track %s (overlap=%.3f, ann %s)",
                    source_track_id,
                    other_track,
                    score,
                    existing_ann.id,
                )
                if id_switch is None or score > id_switch.iou:
                    id_switch = DuplicateCheck(
                        iou=score,
                        possible_id_switch=True,
                        conflicting_track_id=other_track,
                        conflicting_annotation_id=existing_ann.id,
                    )
                continue
            logger.info(
                "Skipping duplicate same-label annotation (overlap=%.3f with ann %s)",
                score,
                existing_ann.id,
            )
            return DuplicateCheck(
                skipped=True, iou=score, conflicting_annotation_id=existing_ann.id
            )

        existing_label = store.get_label_by_id(existing_ann.label_id)
        conflicting_label_name = (
            existing_label.name if existing_label else f"label_{existing_ann.label_id}"
        )
        logger.info(
            "Skipping - location already labeled as '%s' (overlap=%.3f with ann %s)",
            conflicting_label_name,
            score,
            existing_ann.id,
        )
        return DuplicateCheck(
            skipped=True,
            iou=score,
            conflicting_label_name=conflicting_label_name,
            conflicting_annotation_id=existing_ann.id,
        )

    return id_switch or DuplicateCheck()


def find_duplicate_annotation(
    store,
    existing_annotations,
    new_bbox: list[float],
    new_mask: Optional[np.ndarray],
    source_label_id: int,
    threshold: float,
) -> tuple[bool, float, str | None]:
    """Legacy label-only wrapper around check_duplicate (no track awareness)."""
    r = check_duplicate(
        store, existing_annotations, new_bbox, new_mask, source_label_id, threshold
    )
    return r.skipped, r.iou, r.conflicting_label_name


def annotation_overlap_score(
    annotation,
    new_bbox: list[float],
    new_mask: Optional[np.ndarray],
) -> float:
    """Return the best available overlap score for duplicate detection."""
    scores: list[float] = []

    if new_mask is not None and annotation.mask_rle:
        try:
            existing_mask = rle_to_mask(annotation.mask_rle)
            if existing_mask.shape == new_mask.shape:
                scores.append(mask_iou(new_mask, existing_mask))
        except Exception as e:
            logger.warning("Failed to compare masks for duplicate detection: %s", e)

    if annotation.bbox_xyxy:
        scores.append(bbox_duplicate_score(new_bbox, annotation.bbox_xyxy))

    return max(scores) if scores else 0.0


def bbox_duplicate_score(bbox1: list[float], bbox2: list[float]) -> float:
    """Score bbox overlap for duplicate detection.

    IoU catches similarly sized boxes. Intersection over the smaller box catches
    cases where SAM/YOLO makes one box slightly tighter than the other while the
    centers still refer to the same object.
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    w1 = max(0.0, bbox1[2] - bbox1[0])
    h1 = max(0.0, bbox1[3] - bbox1[1])
    w2 = max(0.0, bbox2[2] - bbox2[0])
    h2 = max(0.0, bbox2[3] - bbox2[1])
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    iou = intersection / union if union > 0 else 0.0

    smaller_area = min(area1, area2)
    containment = intersection / smaller_area if smaller_area > 0 else 0.0
    cx1 = (bbox1[0] + bbox1[2]) / 2
    cy1 = (bbox1[1] + bbox1[3]) / 2
    cx2 = (bbox2[0] + bbox2[2]) / 2
    cy2 = (bbox2[1] + bbox2[3]) / 2
    center_distance = float(np.hypot(cx1 - cx2, cy1 - cy2))
    smaller_diag = float(np.hypot(min(w1, w2), min(h1, h2)))

    if smaller_diag > 0 and center_distance <= smaller_diag * 0.5:
        return max(float(iou), float(containment))

    return float(iou)


ANCHOR_SEARCH_MAX_IMAGES = 50


def annotation_mask_area(annotation) -> Optional[float]:
    if annotation is None or not annotation.mask_rle:
        return None
    try:
        area = float(np.count_nonzero(rle_to_mask(annotation.mask_rle)))
        return area or None
    except Exception as e:
        logger.warning("Failed to decode anchor mask: %s", e)
        return None


def find_anchor_mask_area(
    store, source_ann, source_image_id: int, anchor_annotation_id: Optional[int]
) -> Optional[float]:
    """Mask area of the nearest trusted (approved) annotation of this object.

    Explicit ``anchor_annotation_id`` wins. Otherwise: the source itself if
    approved, else the track's most recent earlier anchor
    (``store.find_track_anchor``). Untracked sources fall back to the nearest
    earlier approved same-label annotation (closest centre to the source bbox)
    within ANCHOR_SEARCH_MAX_IMAGES images.
    """
    if anchor_annotation_id is not None:
        return annotation_mask_area(store.get_annotation_by_id(anchor_annotation_id))
    if source_ann.status == "approved":
        return annotation_mask_area(source_ann)

    src_image = store.get_image_by_id(source_image_id)
    track_id = getattr(source_ann, "track_id", None)
    if track_id is not None:
        found = store.find_track_anchor(
            track_id,
            before_image_index=src_image.order_index if src_image else None,
        )
        return annotation_mask_area(found[0]) if found else None

    if src_image is None or source_ann.bbox_xyxy is None:
        return None
    sb = source_ann.bbox_xyxy
    scx, scy = (sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2

    earlier = sorted(
        (
            img
            for img in store.list_images(src_image.project_id)
            if img.order_index < src_image.order_index
        ),
        key=lambda img: img.order_index,
        reverse=True,
    )[:ANCHOR_SEARCH_MAX_IMAGES]
    for img in earlier:
        best, best_d = None, float("inf")
        for a in store.list_annotations(img.id):
            if (
                a.label_id != source_ann.label_id
                or a.status != "approved"
                or a.bbox_xyxy is None
            ):
                continue
            b = a.bbox_xyxy
            d = float(np.hypot((b[0] + b[2]) / 2 - scx, (b[1] + b[3]) / 2 - scy))
            if d < best_d:
                best, best_d = a, d
        if best is not None:
            return annotation_mask_area(best)
    return None




# ==================== Tracking job ====================


FAILURE_MODES = {"stop", "skip", "continue"}
JOB_TERMINAL = {"finished", "stopped", "error"}


@dataclass
class TrackingSettings:
    mode: str = "auto"  # "peak" | "dense" | "auto"
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
    # "stop": a lost object pauses the whole job until resumed.
    # "skip"/"continue": only the lost object is paused; the job keeps going.
    failure_mode: str = "stop"
    # Performance mode only changes how clients display progress.
    performance: bool = False
    max_fallback_attempts: int = 2
    use_anchor: bool = True


# propagate_fn(store, source_ann, source_img, target_img, expected_area_ratio,
#              anchor_mask_area, settings) -> PropagationResult  (raises PropagationError)
PropagateFn = Callable[..., PropagationResult]


@dataclass
class JobTrack:
    track_id: int
    label_id: int
    state: str = "active"  # "active" | "lost"
    expected_area_ratio: float = 1.0
    lost_at_image_id: Optional[int] = None
    failed_image_ids: set[int] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "label_id": self.label_id,
            "state": self.state,
            "expected_area_ratio": round(self.expected_area_ratio, 4),
            "lost_at_image_id": self.lost_at_image_id,
        }


class TrackingJob:
    """One tracking run over a frame range of a project (see module doc)."""

    def __init__(
        self,
        db_path: str,
        project_id: int,
        propagate_fn: PropagateFn,
        start_index: int,
        end_index: Optional[int] = None,
        settings: Optional[TrackingSettings] = None,
        track_ids: Optional[list[int]] = None,
        label_ids: Optional[list[int]] = None,
    ):
        self.id = uuid.uuid4().hex[:12]
        self.db_path = db_path
        self.project_id = project_id
        self.propagate_fn = propagate_fn
        self.settings = settings or TrackingSettings()
        if self.settings.failure_mode not in FAILURE_MODES:
            raise ValueError(f"failure_mode must be one of {sorted(FAILURE_MODES)}")
        self.start_index = start_index
        self.end_index = end_index
        self.track_filter = set(track_ids) if track_ids else None
        self.label_filter = set(label_ids) if label_ids else None

        self.status = "pending"
        self.cursor = start_index  # index of the current source frame
        self.tracks: dict[int, JobTrack] = {}
        self.paused: Optional[dict] = None
        self.error: Optional[str] = None
        self.counts = {
            "frames_done": 0,
            "created": 0,
            "uncertain": 0,
            "lost": 0,
            "duplicates": 0,
            "id_switches": 0,
            "fallbacks": 0,
            "tracks_created": 0,
        }
        self.created_at = time.time()
        self.finished_at: Optional[float] = None

        self._cond = threading.Condition()
        self._events: list[dict] = []
        self._seq = 0
        self._stop = False
        self._resume = False
        self._thread: Optional[threading.Thread] = None
        self._images: list = []
        self._labels: dict[int, str] = {}

    # ---------- control ----------

    def start(self) -> "TrackingJob":
        self._thread = threading.Thread(
            target=self.run, name=f"tracking-job-{self.id}", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()

    def resume(self) -> bool:
        """Resume a paused job. Returns False if it was not paused."""
        with self._cond:
            if self.status != "paused":
                return False
            self._resume = True
            self._cond.notify_all()
            return True

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    @property
    def is_active(self) -> bool:
        return self.status not in JOB_TERMINAL

    # ---------- events ----------

    def _emit(self, type_: str, **data: Any) -> dict:
        with self._cond:
            self._seq += 1
            ev = {"seq": self._seq, "type": type_, "job_id": self.id, "ts": time.time()}
            ev.update(data)
            self._events.append(ev)
            self._cond.notify_all()
        return ev

    def events_after(self, after: int = 0) -> list[dict]:
        with self._cond:
            return [e for e in self._events if e["seq"] > after]

    def wait_events(self, after: int = 0, timeout: float = 15.0) -> list[dict]:
        """Block until events newer than ``after`` exist, the job ends, or timeout."""
        deadline = time.time() + timeout
        with self._cond:
            while True:
                new = [e for e in self._events if e["seq"] > after]
                remaining = deadline - time.time()
                if new or not self.is_active or remaining <= 0:
                    return new
                self._cond.wait(remaining)

    @property
    def last_seq(self) -> int:
        return self._seq

    def snapshot(self) -> dict:
        total = max(0, (self.end_index or self.start_index) - self.start_index)
        current = self._images[self.cursor] if 0 <= self.cursor < len(self._images) else None
        return {
            "id": self.id,
            "project_id": self.project_id,
            "status": self.status,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "current_index": self.cursor,
            "current_image_id": current.id if current else None,
            "total_frames": total,
            "counts": dict(self.counts),
            "tracks": [t.to_dict() for t in self.tracks.values()],
            "paused": self.paused,
            "settings": asdict(self.settings),
            "error": self.error,
            "last_seq": self._seq,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }

    # ---------- main loop ----------

    def run(self) -> None:
        store = ProjectStore(self.db_path)
        try:
            self._images = store.list_images(self.project_id)
            last = len(self._images) - 1
            if not (0 <= self.start_index <= last):
                raise ValueError(f"start_index {self.start_index} out of range 0..{last}")
            self.end_index = last if self.end_index is None else max(
                self.start_index, min(self.end_index, last)
            )
            self._labels = {l.id: l.name for l in store.list_labels(self.project_id)}
            self.status = "running"
            self._adopt_tracks(store, self._images[self.start_index])
            self._emit(
                "started",
                start_index=self.start_index,
                end_index=self.end_index,
                image_id=self._images[self.start_index].id,
                tracks=[t.to_dict() for t in self.tracks.values()],
                settings=asdict(self.settings),
            )

            while self.cursor < self.end_index and not self._stop:
                src = self._images[self.cursor]
                tgt = self._images[self.cursor + 1]
                lost_now = self._process_frame(store, src, tgt)
                self.cursor += 1
                self.counts["frames_done"] = self.cursor - self.start_index
                if lost_now and self.settings.failure_mode == "stop" and not self._stop:
                    self._pause(tgt, lost_now)
                    if self._stop:
                        break
                    self._on_resume(store, tgt)

            self.status = "stopped" if self._stop else "finished"
            self.finished_at = time.time()
            self._emit(
                "finished",
                reason="stopped" if self._stop else "completed",
                current_index=self.cursor,
                image_id=self._images[self.cursor].id if self._images else None,
                counts=dict(self.counts),
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Tracking job %s failed", self.id)
            self.error = str(e)
            self.status = "error"
            self.finished_at = time.time()
            self._emit("error", message=str(e), current_index=self.cursor)
        finally:
            store.close()

    # ---------- steps ----------

    def _label_name(self, label_id: int) -> str:
        return self._labels.get(label_id, f"label_{label_id}")

    def _adopt_tracks(self, store: ProjectStore, image) -> None:
        """Bring objects present on ``image`` into the job.

        Explicit ``track_ids``: only those. Otherwise every (label-filtered)
        annotation on the frame; untracked ones get a new track (policy for
        legacy / untracked start-frame annotations).
        """
        if self.track_filter is not None:
            for tid in self.track_filter:
                if tid in self.tracks:
                    continue
                tr = store.get_track(tid)
                if tr is None or tr.status == "ended":
                    continue
                self.tracks[tid] = JobTrack(tid, tr.label_id)
            return

        for ann in store.list_annotations(image.id):
            if self.label_filter is not None and ann.label_id not in self.label_filter:
                continue
            if ann.track_id is None:
                tr = store.create_track(self.project_id, ann.label_id)
                store.update_annotation(ann.id, track_id=tr.id)
                self.counts["tracks_created"] += 1
                self.tracks[tr.id] = JobTrack(tr.id, ann.label_id)
                continue
            if ann.track_id in self.tracks:
                continue
            tr = store.get_track(ann.track_id)
            if tr is None or tr.status == "ended":
                continue
            if tr.status == "lost" and ann.is_anchor:
                store.update_track(tr.id, status="active", lost_at_image_id=None)
            self.tracks[tr.id] = JobTrack(tr.id, tr.label_id)

    def _process_frame(self, store: ProjectStore, src, tgt) -> list[int]:
        """Propagate every job track from ``src`` to ``tgt``. Returns tracks lost here."""
        self._adopt_tracks(store, src)
        by_track: dict[int, Any] = {}
        for a in store.list_annotations(src.id):
            if a.track_id is None:
                continue
            cur = by_track.get(a.track_id)
            if cur is None or (a.is_anchor and not cur.is_anchor):
                by_track[a.track_id] = a
        on_target = {a.track_id for a in store.list_annotations(tgt.id) if a.track_id}

        created, existing, skipped, lost, uncertain = [], [], [], [], []
        for jt in list(self.tracks.values()):
            if self._stop:
                break
            if jt.state == "lost":
                tr = store.get_track(jt.track_id)
                if tr is None or tr.status != "active":
                    continue
                # Re-anchored by a client -> rejoin from its newest anchor.
                jt.state, jt.lost_at_image_id, jt.expected_area_ratio = "active", None, 1.0
                jt.failed_image_ids.clear()
                self._emit("object_resumed", track_id=jt.track_id, image_id=src.id,
                           image_index=src.order_index)
            if jt.track_id in on_target:
                existing.append(jt.track_id)
                continue

            result, err, attempts = self._propagate_track(store, jt, by_track.get(jt.track_id), src, tgt)
            if result is None:
                self._mark_lost(store, jt, src, tgt, err, attempts)
                lost.append(jt.track_id)
                continue

            dup = check_duplicate(
                store,
                store.list_annotations(tgt.id),
                result.bbox,
                result.mask,
                jt.label_id,
                self.settings.duplicate_threshold,
                source_track_id=jt.track_id,
            )
            if dup.skipped:
                self.counts["duplicates"] += 1
                skipped.append({
                    "track_id": jt.track_id,
                    "iou": round(float(dup.iou), 4),
                    "conflicting_label_name": dup.conflicting_label_name,
                    "conflicting_annotation_id": dup.conflicting_annotation_id,
                })
                continue

            status = (
                STATUS_NEEDS_REVIEW
                if result.uncertain or dup.possible_id_switch
                else "pending"
            )
            mask = result.mask
            ann = store.create_annotation(
                image_id=tgt.id,
                label_id=jt.label_id,
                bbox_xyxy=[float(v) for v in result.bbox],
                source="tracked",
                status=status,
                confidence=float(result.confidence),
                mask_rle=mask_to_rle(mask) if mask is not None else None,
                polygon_norm=mask_to_yolo_polygon(mask, tgt.width, tgt.height)
                if mask is not None
                else None,
                track_id=jt.track_id,
            )
            self.counts["created"] += 1
            created.append({"annotation_id": ann.id, "track_id": jt.track_id, "status": status})
            jt.expected_area_ratio = update_area_trend(
                jt.expected_area_ratio,
                result.signals.size_ratio if result.signals else None,
            )
            if result.uncertain:
                self.counts["uncertain"] += 1
                uncertain.append(jt.track_id)
                self._emit(
                    "uncertain",
                    track_id=jt.track_id,
                    label_id=jt.label_id,
                    image_id=tgt.id,
                    image_index=tgt.order_index,
                    annotation_id=ann.id,
                    **result.uncertainty(),
                )
            if dup.possible_id_switch:
                self.counts["id_switches"] += 1
                self._emit(
                    "possible_id_switch",
                    track_id=jt.track_id,
                    label_id=jt.label_id,
                    image_id=tgt.id,
                    image_index=tgt.order_index,
                    annotation_id=ann.id,
                    conflicting_track_id=dup.conflicting_track_id,
                    conflicting_annotation_id=dup.conflicting_annotation_id,
                    iou=round(float(dup.iou), 4),
                )

        self._emit(
            "frame_done",
            source_image_id=src.id,
            image_id=tgt.id,
            image_index=tgt.order_index,
            created=created,
            existing=existing,
            skipped=skipped,
            lost=lost,
            uncertain=uncertain,
            progress={
                "done": self.cursor + 1 - self.start_index,
                "total": (self.end_index or 0) - self.start_index,
            },
        )
        return lost

    def _propagate_track(self, store, jt: JobTrack, src_ann, src, tgt):
        """Propagate one track to ``tgt``; on failure retry from earlier anchors
        of the same track. Returns (result | None, last_error, fallback_attempts)."""
        tried = set(jt.failed_image_ids)
        last_err: Optional[PropagationError] = None

        def attempt(ann, img) -> PropagationResult:
            anchor_area = (
                find_anchor_mask_area(store, ann, img.id, None)
                if self.settings.use_anchor
                else None
            )
            try:
                res = self.propagate_fn(
                    store, ann, img, tgt, jt.expected_area_ratio, anchor_area, self.settings
                )
            except PropagationError:
                raise
            except Exception as e:  # noqa: BLE001 - one object must not kill the job
                logger.exception("Propagation error for track %s", jt.track_id)
                raise PropagationError(str(e), reason="error") from e
            if res is None:
                raise PropagationNotFoundError("No result", reason="not_found")
            return res

        if src_ann is not None and src_ann.bbox_xyxy is not None:
            try:
                return attempt(src_ann, src), None, 0
            except PropagationError as e:
                last_err = e
                tried.add(src.id)

        attempts = 0
        for _ in range(max(0, self.settings.max_fallback_attempts)):
            found = store.find_track_anchor(
                jt.track_id, before_image_index=tgt.order_index, exclude_image_ids=tried
            )
            if found is None:
                break
            ann, img = found
            tried.add(img.id)
            attempts += 1
            try:
                res = attempt(ann, img)
                self.counts["fallbacks"] += 1
                return res, None, attempts
            except PropagationError as e:
                last_err = e
        return None, last_err, attempts

    def _mark_lost(self, store, jt: JobTrack, src, tgt, err, attempts: int) -> None:
        jt.state = "lost"
        jt.lost_at_image_id = tgt.id
        jt.failed_image_ids.add(tgt.id)
        store.update_track(jt.track_id, status="lost", lost_at_image_id=tgt.id)
        self.counts["lost"] += 1
        payload = (
            err.to_dict()
            if err is not None
            else {
                "status": "lost",
                "reason": "no_source",
                "message": "No source annotation or anchor for this track",
                "signals": None,
            }
        )
        self._emit(
            "object_lost",
            track_id=jt.track_id,
            label_id=jt.label_id,
            label_name=self._label_name(jt.label_id),
            image_id=tgt.id,
            image_index=tgt.order_index,
            source_image_id=src.id,
            fallback_attempts=attempts,
            paused="job" if self.settings.failure_mode == "stop" else "object",
            **payload,
        )

    def _pause(self, image, lost_track_ids: list[int]) -> None:
        tracks = [
            {
                "track_id": tid,
                "label_id": self.tracks[tid].label_id,
                "label_name": self._label_name(self.tracks[tid].label_id),
            }
            for tid in lost_track_ids
        ]
        self.paused = {"image_id": image.id, "image_index": image.order_index, "tracks": tracks}
        self.status = "paused"
        self._emit("paused_waiting_for_label", scope="job", **self.paused)
        with self._cond:
            while not self._resume and not self._stop:
                self._cond.wait()
            self._resume = False
        self.paused = None
        if not self._stop:
            self.status = "running"
            self._emit("resumed", image_id=image.id, image_index=image.order_index)

    def _on_resume(self, store: ProjectStore, image) -> None:
        """Match new labels drawn on the paused frame to the tracks lost there.

        A track re-anchored explicitly (``/api/tracks/{id}/reanchor``) is already
        active. Otherwise the first new annotation of the same label that is not
        part of another job track re-anchors it.
        """
        anns = store.list_annotations(image.id)
        for jt in self.tracks.values():
            if jt.state != "lost" or jt.lost_at_image_id != image.id:
                continue
            tr = store.get_track(jt.track_id)
            if tr is not None and tr.status == "active":
                continue
            if any(a.track_id == jt.track_id for a in anns):
                store.update_track(jt.track_id, status="active", lost_at_image_id=None)
                continue
            cand = next(
                (
                    a
                    for a in anns
                    if a.label_id == jt.label_id
                    and (a.track_id is None or a.track_id not in self.tracks)
                ),
                None,
            )
            if cand is None:
                continue
            old_track = cand.track_id
            store.update_annotation(cand.id, track_id=jt.track_id, status="approved")
            store.update_track(jt.track_id, status="active", lost_at_image_id=None)
            if old_track is not None and not store.list_track_annotations(old_track):
                store.delete_track(old_track)
            anns = store.list_annotations(image.id)
            self._emit(
                "reanchored",
                track_id=jt.track_id,
                annotation_id=cand.id,
                image_id=image.id,
                image_index=image.order_index,
            )


class ServicePropagator:
    """``PropagateFn`` backed by the loaded ``PropagateService``."""

    def __init__(self, service, cache_size: int = 4):
        self.service = service
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    def _load(self, img) -> np.ndarray:
        arr = self._cache.get(img.id)
        if arr is None:
            from PIL import Image

            arr = np.array(Image.open(img.path).convert("RGB"))
            self._cache[img.id] = arr
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(img.id)
        return arr

    def __call__(self, store, source_ann, source_img, target_img, expected_area_ratio,
                 anchor_mask_area, s: TrackingSettings) -> PropagationResult:
        common = dict(
            source_image=self._load(source_img),
            source_bbox=source_ann.bbox_xyxy,
            source_mask=rle_to_mask(source_ann.mask_rle) if source_ann.mask_rle else None,
            target_image=self._load(target_img),
            source_image_id=str(source_img.id),
            target_image_id=str(target_img.id),
            annotation_id=source_ann.id,
            use_bbox_hint=s.use_bbox_hint,
            bbox_hint_scale=max(0.5, min(3.0, s.bbox_hint_scale)),
            prune_thin_artifacts=s.prune_thin_artifacts,
            size_min_ratio=s.size_min_ratio,
            size_max_ratio=s.size_max_ratio,
            top_k=s.top_k,
            min_score=s.min_similarity,
            expected_area_ratio=expected_area_ratio,
            anchor_mask_area=anchor_mask_area,
        )
        if s.mode != "peak" or s.iou_verify:
            return self.service.propagate_with_dense_fallback(
                mode=s.mode,
                iou_verify=s.iou_verify,
                iou_threshold=s.iou_threshold,
                use_cached_masks=s.use_cached_masks,
                **common,
            )
        return self.service.propagate_annotation(**common)


class TrackingJobManager:
    """In-process registry; at most one active job per project database."""

    MAX_KEPT = 20

    def __init__(self):
        self._jobs: OrderedDict[str, TrackingJob] = OrderedDict()
        self._lock = threading.Lock()

    def active(self, db_path: Optional[str] = None) -> Optional[TrackingJob]:
        with self._lock:
            for job in reversed(self._jobs.values()):
                if job.is_active and (db_path is None or job.db_path == db_path):
                    return job
        return None

    def submit(self, job: TrackingJob, start: bool = True) -> TrackingJob:
        with self._lock:
            for other in self._jobs.values():
                if other.is_active and other.db_path == job.db_path:
                    raise RuntimeError(f"Tracking job {other.id} is already active for this project")
            self._jobs[job.id] = job
            while len(self._jobs) > self.MAX_KEPT:
                oldest = next(iter(self._jobs.values()))
                if oldest.is_active:
                    break
                self._jobs.popitem(last=False)
        return job.start() if start else job

    def get(self, job_id: str) -> Optional[TrackingJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[TrackingJob]:
        with self._lock:
            return list(self._jobs.values())


_manager = TrackingJobManager()


def get_job_manager() -> TrackingJobManager:
    return _manager
