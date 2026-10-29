"""Backend tracking jobs (issue #3) with a stubbed propagator (no ML)."""

import sqlite3
import time

import numpy as np
import pytest
from PIL import Image

from core.store import ProjectStore
from ml.propagate import (
    STATUS_NEEDS_REVIEW,
    PropagationNotFoundError,
    PropagationResult,
    TrackingSignals,
)
from ml.tracking_job import TrackingJob, TrackingSettings, get_job_manager

W = H = 100


def _mask(b):
    m = np.zeros((H, W), np.uint8)
    m[int(b[1]):int(b[3]), int(b[0]):int(b[2])] = 1
    return m


class StubPropagator:
    """Moves each box by +1px per frame. ``script[(track_id, target_index)]``
    overrides: 'lost', 'uncertain', or a bbox list."""

    def __init__(self):
        self.script = {}
        self.calls = []

    def __call__(self, store, ann, src_img, tgt_img, expected, anchor_area, settings):
        self.calls.append((ann.track_id, src_img.order_index, tgt_img.order_index))
        action = self.script.get((ann.track_id, tgt_img.order_index))
        if action == "lost":
            raise PropagationNotFoundError("gone", reason="not_found")
        b = action if isinstance(action, list) else [v + 1 for v in ann.bbox_xyxy]
        uncertain = action == "uncertain"
        return PropagationResult(
            bbox=b, mask=_mask(b), confidence=0.9, fallback_used=False, area_ratio=1.0,
            method="peak", status=STATUS_NEEDS_REVIEW if uncertain else "ok",
            reasons=["size_change"] if uncertain else [],
            signals=TrackingSignals(0.9, 0.9, 1.0, 1.3 if uncertain else 1.0),
        )


@pytest.fixture
def project(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        Image.new("RGB", (W, H)).save(img_dir / f"img_{i:02d}.png")
    proj = ProjectStore.create_project(str(tmp_path / "proj"), str(img_dir), "t")
    store = ProjectStore(proj.db_path)
    yield proj, store
    store.close()


def _wait(job, type_, after=0, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        for ev in job.wait_events(after, timeout=0.2):
            after = ev["seq"]
            if ev["type"] == type_:
                return ev
    raise AssertionError(f"no {type_} event; got {[e['type'] for e in job.events_after(0)]}")


def test_uncertain_and_id_switch_events(project):
    proj, store = project
    box = store.upsert_label(proj.id, "box")
    imgs = store.list_images(proj.id)
    a = store.create_annotation(imgs[0].id, box.id, [10, 10, 30, 30], mask_rle=None)
    b = store.create_annotation(imgs[0].id, box.id, [60, 60, 80, 80])
    store.commit()

    stub = StubPropagator()
    job = TrackingJob(proj.db_path, proj.id, stub, start_index=0, end_index=3,
                      settings=TrackingSettings(failure_mode="skip", duplicate_threshold=0.75))
    job.run()  # synchronous

    assert job.status == "finished"
    # Untracked start-frame annotations got tracks at job start
    ta = store.get_annotation_by_id(a.id).track_id
    tb = store.get_annotation_by_id(b.id).track_id
    assert ta and tb and ta != tb and job.counts["tracks_created"] == 2
    assert [e["type"] for e in job.events_after(0)][-1] == "finished"
    assert len([e for e in job.events_after(0) if e["type"] == "frame_done"]) == 3
    assert all(ann.track_id in (ta, tb) and ann.source == "tracked"
               for img in imgs[1:4] for ann in store.list_annotations(img.id))

    # Now a second run over a fresh range with scripted events
    for img in imgs[1:]:
        for ann in store.list_annotations(img.id):
            store.delete_annotation(ann.id)
    stub.script[(ta, 2)] = "uncertain"
    stub.script[(tb, 1)] = [11, 11, 31, 31]  # lands on track a's frame-1 box
    job = TrackingJob(proj.db_path, proj.id, stub, start_index=0, end_index=2,
                      settings=TrackingSettings(failure_mode="skip"))
    job.run()
    evs = job.events_after(0)
    sw = next(e for e in evs if e["type"] == "possible_id_switch")
    assert sw["track_id"] == tb and sw["conflicting_track_id"] == ta and sw["image_index"] == 1
    assert store.get_annotation_by_id(sw["annotation_id"]).status == "needs_review"
    un = next(e for e in evs if e["type"] == "uncertain")
    assert un["track_id"] == ta and un["status"] == "needs_review" and "size_change" in un["reasons"]
    assert un["signals"]["mask_area_ratio"] == 1.3
    # area trend follows the observed ratio for that track
    jt = {t["track_id"]: t for t in job.snapshot()["tracks"]}
    assert jt[ta]["expected_area_ratio"] > 1.0 and jt[tb]["expected_area_ratio"] == 1.0


def test_skip_mode_pauses_only_object_and_fallback(project):
    proj, store = project
    box = store.upsert_label(proj.id, "box")
    imgs = store.list_images(proj.id)
    store.create_annotation(imgs[0].id, box.id, [10, 10, 30, 30])
    store.create_annotation(imgs[0].id, box.id, [60, 60, 80, 80])
    store.commit()
    stub = StubPropagator()
    job = TrackingJob(proj.db_path, proj.id, stub, start_index=0,
                      settings=TrackingSettings(failure_mode="skip"))
    # track a fails 1->2 but re-anchors from its frame-0 anchor; track b is lost at 3
    job._adopt_tracks(store, imgs[0])  # create tracks now so we can script them
    ta, tb = [a.track_id for a in store.list_annotations(imgs[0].id)]
    calls = {"n": 0}

    def flaky(st, ann, src, tgt, *rest):
        if ann.track_id == ta and tgt.order_index == 2 and src.order_index == 1:
            raise PropagationNotFoundError("gone", reason="not_found")
        if ann.track_id == tb and tgt.order_index == 3:
            raise PropagationNotFoundError("gone", reason="not_found")
        return stub(st, ann, src, tgt, *rest)

    job.propagate_fn = flaky
    job.run()
    assert job.status == "finished" and job.counts["fallbacks"] == 1
    lost = [e for e in job.events_after(0) if e["type"] == "object_lost"]
    assert len(lost) == 1 and lost[0]["track_id"] == tb and lost[0]["paused"] == "object"
    assert lost[0]["reason"] == "not_found" and lost[0]["status"] == "lost"
    assert store.get_track(tb).status == "lost"
    # track a kept going to the end, track b stopped at frame 2
    assert any(a.track_id == ta for a in store.list_annotations(imgs[4].id))
    assert not any(a.track_id == tb for a in store.list_annotations(imgs[4].id))
    assert not any(e["type"] == "paused_waiting_for_label" for e in job.events_after(0))


def test_lost_pause_new_label_resume(project):
    """Stop mode: a box of the same label drawn on the paused frame re-anchors the lost track."""
    proj, store = project
    box = store.upsert_label(proj.id, "box")
    imgs = store.list_images(proj.id)
    ta = store.create_track(proj.id, box.id).id
    store.create_annotation(imgs[0].id, box.id, [10, 10, 30, 30], track_id=ta)
    store.commit()
    stub = StubPropagator()
    stub.script[(ta, 2)] = "lost"  # lost from frame 1 and from the frame-0 anchor
    job = TrackingJob(proj.db_path, proj.id, stub, start_index=0, end_index=3,
                      settings=TrackingSettings(failure_mode="stop")).start()
    p = _wait(job, "paused_waiting_for_label")
    assert p["image_index"] == 2
    assert job.status == "paused" and p["tracks"][0]["track_id"] == ta

    # A client draws a new box (gets its own track, as POST /api/annotations does)
    s2 = ProjectStore(proj.db_path)
    other = s2.create_track(proj.id, box.id)
    s2.create_annotation(imgs[2].id, box.id, [40, 40, 60, 60], track_id=other.id)
    s2.close()
    assert job.resume()
    _wait(job, "finished")
    job.join(2)
    assert job.status == "finished"
    re = next(e for e in job.events_after(0) if e["type"] == "reanchored")
    assert re["track_id"] == ta
    assert store.get_track(other.id) is None  # orphan track removed
    assert store.get_track(ta).status == "active"
    [f3] = [a for a in store.list_annotations(imgs[3].id)]
    assert f3.track_id == ta and f3.bbox_xyxy == [41, 41, 61, 61]


@pytest.fixture
def client(project, monkeypatch):
    from fastapi.testclient import TestClient
    import backend.api.projects as projects_api
    import backend.api.track_jobs as track_jobs_api
    from backend.main import app

    proj, store = project
    store.close()
    store._conn = sqlite3.connect(proj.db_path, check_same_thread=False)
    store._conn.row_factory = sqlite3.Row
    store._conn.execute("PRAGMA foreign_keys = ON")
    projects_api._current_store = store
    projects_api._current_project = proj
    stub = StubPropagator()
    monkeypatch.setattr(track_jobs_api, "_propagate_fn", lambda: stub)
    yield TestClient(app), proj, store, stub
    for job in get_job_manager().list():
        job.stop()
        job.join(2)
    projects_api._current_store = None
    projects_api._current_project = None


def _poll(c, job_id, type_, after=0, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        r = c.get(f"/api/track/jobs/{job_id}/events", params={"after": after, "wait": 0.5}).json()
        for ev in r["events"]:
            after = ev["seq"]
            if ev["type"] == type_:
                return ev, after
    raise AssertionError(f"no {type_}")


def test_api_lost_pause_reanchor_resume_finished(client):
    c, proj, store, stub = client
    box = store.upsert_label(proj.id, "box")
    imgs = store.list_images(proj.id)
    a = c.post("/api/annotations", json={"image_id": imgs[0].id, "label_id": box.id,
                                         "bbox": [10, 10, 30, 30]}).json()
    ta = a["track_id"]
    stub.script[(ta, 2)] = "lost"

    job = c.post("/api/track/jobs", json={"start_index": 0, "failure_mode": "stop"}).json()
    jid = job["id"]
    assert c.post("/api/track/jobs", json={"start_index": 0}).status_code == 409

    lost, seq = _poll(c, jid, "object_lost")
    assert lost["track_id"] == ta and lost["paused"] == "job" and lost["image_index"] == 2
    paused, seq = _poll(c, jid, "paused_waiting_for_label", after=seq - 1)
    st = c.get(f"/api/track/jobs/{jid}").json()
    assert st["status"] == "paused" and st["paused"]["image_id"] == imgs[2].id
    assert c.get(f"/api/tracks/{ta}").json()["status"] == "lost"
    assert c.get("/api/track/jobs", params={"active": True}).json()[0]["id"] == jid

    # Any client supplies the new label via the tracks API, then resumes
    c.post(f"/api/tracks/{ta}/reanchor", json={"image_id": imgs[2].id, "bbox": [50, 50, 70, 70]})
    assert c.post(f"/api/track/jobs/{jid}/resume").status_code == 200
    fin, _ = _poll(c, jid, "finished", after=seq)
    assert fin["reason"] == "completed"
    assert c.post(f"/api/track/jobs/{jid}/resume").status_code == 409

    last = store.list_annotations(imgs[4].id)
    assert [x.track_id for x in last] == [ta] and last[0].bbox_xyxy == [52, 52, 72, 72]
    assert last[0].source == "tracked"

    # SSE stream replays all events of a finished job and closes
    with c.stream("GET", f"/api/track/jobs/{jid}/stream") as r:
        body = "".join(r.iter_text())
    assert "event: started" in body and "event: paused_waiting_for_label" in body
    assert body.rstrip().endswith("}") and "event: finished" in body

    # Single step (num_frames=1) and stop
    job2 = c.post("/api/track/jobs", json={"start_index": 0, "num_frames": 1}).json()
    _poll(c, job2["id"], "finished")
    assert c.get(f"/api/track/jobs/{job2['id']}").json()["end_index"] == 1
