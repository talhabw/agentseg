"""Tests for object identity (tracks, issue #4): migration, store, API, duplicate checks."""

import os
import sqlite3
import sys
import types

import numpy as np
import pytest
from PIL import Image


def _stub_heavy_modules():
    """backend.api.ml imports torch and ML services; stub them so tests stay light."""
    try:
        import torch  # noqa: F401
    except ImportError:
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        sys.modules["torch"] = torch
    for name, attrs in {
        "ml.segment": ["get_segment_service", "clear_segment_service"],
        "ml.embed": [
            "get_embed_service",
            "set_default_model",
            "clear_embed_service",
            "get_available_models",
        ],
        "ml.propagate": [
            "PropagateService",
            "PropagationSizeMismatchError",
            "PropagationNotFoundError",
        ],
        "backend.yolo_autoannotate": [
            "YoloAnnotateOptions",
            "run_yolo_on_image",
            "run_yolo_on_project",
        ],
    }.items():
        try:
            __import__(name)
        except Exception:
            mod = types.ModuleType(name)
            for a in attrs:
                setattr(mod, a, type(a, (Exception,), {}) if a.endswith("Error") else object)
            sys.modules[name] = mod


_stub_heavy_modules()

from core.db import SCHEMA_VERSION, get_schema_version, migrate_db  # noqa: E402
from core.store import ProjectStore  # noqa: E402


@pytest.fixture
def project(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        Image.new("RGB", (100, 100)).save(img_dir / f"img_{i:02d}.png")
    proj = ProjectStore.create_project(str(tmp_path / "proj"), str(img_dir), "t")
    store = ProjectStore(proj.db_path)
    yield proj, store
    store.close()


V1_SCHEMA = """
CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE projects (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, root_dir TEXT NOT NULL,
  db_path TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, settings_json TEXT);
CREATE TABLE labels (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL, name TEXT NOT NULL,
  color_hex TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, UNIQUE(project_id, name));
CREATE TABLE images (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL, path TEXT NOT NULL,
  width INTEGER NOT NULL, height INTEGER NOT NULL, order_index INTEGER NOT NULL, UNIQUE(project_id, path));
CREATE TABLE annotations (id INTEGER PRIMARY KEY AUTOINCREMENT, image_id INTEGER NOT NULL, label_id INTEGER NOT NULL,
  bbox_xyxy_json TEXT, polygon_norm_json TEXT, mask_rle_json TEXT, source TEXT DEFAULT 'manual', confidence REAL,
  status TEXT DEFAULT 'approved', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT);
INSERT INTO schema_version (version) VALUES (1);
INSERT INTO projects (name, root_dir, db_path) VALUES ('old', '/x', '/x/agentseg.db');
INSERT INTO labels (project_id, name, color_hex) VALUES (1, 'box', '#fff');
INSERT INTO images (project_id, path, width, height, order_index) VALUES (1, '/x/a.png', 10, 10, 0);
INSERT INTO annotations (image_id, label_id, bbox_xyxy_json, status) VALUES (1, 1, '[0,0,5,5]', 'needs_review');
"""


def test_migrate_v1_to_v2(tmp_path):
    db = str(tmp_path / "agentseg.db")
    conn = sqlite3.connect(db)
    conn.executescript(V1_SCHEMA)
    conn.close()

    migrate_db(db)
    assert get_schema_version(db) == SCHEMA_VERSION == 2
    migrate_db(db)  # idempotent

    store = ProjectStore(db)
    ann = store.list_annotations(1)[0]
    assert ann.track_id is None and ann.bbox_xyxy == [0, 0, 5, 5]
    t = store.create_track(1, 1)
    store.update_annotation(ann.id, track_id=t.id)
    assert store.list_annotations(1)[0].track_id == t.id
    # list_annotations_by_status no longer depends on column positions
    [(a, img)] = store.list_annotations_by_status(1, "needs_review")
    assert img.path == "/x/a.png" and a.track_id == t.id
    store.close()


def test_store_tracks(project):
    proj, store = project
    label = store.upsert_label(proj.id, "box")
    imgs = store.list_images(proj.id)
    t = store.create_track(proj.id, label.id, name="box#1")
    store.create_annotation(imgs[0].id, label.id, [0, 0, 10, 10], status="approved", track_id=t.id)
    store.create_annotation(imgs[1].id, label.id, [1, 1, 11, 11], status="pending", track_id=t.id)
    store.create_annotation(imgs[2].id, label.id, [2, 2, 12, 12], status="approved", track_id=t.id)

    summ = store.track_summaries(proj.id)[t.id]
    assert summ["annotation_count"] == 3 and summ["anchor_count"] == 2
    assert (summ["first_image_index"], summ["last_image_index"]) == (0, 2)

    ann, img = store.find_track_anchor(t.id, before_image_index=4)
    assert img.order_index == 2
    ann, img = store.find_track_anchor(t.id, before_image_index=4, exclude_image_ids={imgs[2].id})
    assert img.order_index == 0

    store.delete_track(t.id)
    assert all(a.track_id is None for a in store.list_annotations(imgs[0].id))


@pytest.fixture
def client(project):
    from fastapi.testclient import TestClient
    import backend.api.projects as projects_api
    from backend.main import app

    proj, store = project
    # TestClient runs handlers in another thread; allow cross-thread sqlite use in tests.
    store.close()
    store._conn = sqlite3.connect(proj.db_path, check_same_thread=False)
    store._conn.row_factory = sqlite3.Row
    store._conn.execute("PRAGMA foreign_keys = ON")
    projects_api._current_store = store
    projects_api._current_project = proj
    yield TestClient(app), proj, store
    projects_api._current_store = None
    projects_api._current_project = None


def test_tracks_api(client):
    c, proj, store = client
    label = store.upsert_label(proj.id, "box")
    imgs = store.list_images(proj.id)

    # Initial manual labels get their own new tracks
    a1 = c.post("/api/annotations", json={"image_id": imgs[0].id, "label_id": label.id, "bbox": [0, 0, 10, 10]}).json()
    a2 = c.post("/api/annotations", json={"image_id": imgs[0].id, "label_id": label.id, "bbox": [50, 50, 60, 60]}).json()
    assert a1["track_id"] and a2["track_id"] and a1["track_id"] != a2["track_id"]
    assert a1["is_anchor"] is True
    t1 = a1["track_id"]

    # Tracked results inherit the given track; without track_id they stay untracked
    p = c.post("/api/annotations", json={"image_id": imgs[1].id, "label_id": label.id, "bbox": [1, 1, 11, 11],
                                          "source": "tracked", "status": "pending", "track_id": t1}).json()
    assert p["track_id"] == t1 and p["is_anchor"] is False
    u = c.post("/api/annotations", json={"image_id": imgs[1].id, "label_id": label.id, "bbox": [1, 1, 11, 11],
                                          "source": "tracked", "status": "pending"}).json()
    assert u["track_id"] is None
    assert c.post("/api/annotations", json={"image_id": imgs[1].id, "label_id": label.id, "bbox": [0, 0, 1, 1],
                                             "track_id": 9999}).status_code == 404

    tracks = c.get("/api/tracks").json()
    assert len(tracks) == 2
    tr = next(t for t in tracks if t["id"] == t1)
    assert tr["annotation_count"] == 2 and tr["anchor_count"] == 1 and tr["status"] == "active"

    frames = c.get(f"/api/tracks/{t1}/annotations").json()
    assert [f["image_index"] for f in frames] == [0, 1]

    lost = c.post(f"/api/tracks/{t1}/lost", json={"image_id": imgs[2].id}).json()
    assert lost["status"] == "lost" and lost["lost_at_image_id"] == imgs[2].id
    assert c.get("/api/tracks", params={"status": "lost"}).json()[0]["id"] == t1

    # Anchor lookup by track (and legacy label endpoint filtered by track)
    anc = c.get(f"/api/tracks/{t1}/anchor", params={"before_image_index": 3}).json()
    assert anc["found"] and anc["annotation"]["id"] == a1["id"]
    legacy = c.get(f"/api/annotations/fallback/{label.id}",
                   params={"before_image_index": 3, "project_id": proj.id, "track_id": a2["track_id"]}).json()
    assert legacy["annotation"]["id"] == a2["id"]
    legacy_label = c.get(f"/api/annotations/fallback/{label.id}",
                         params={"before_image_index": 3, "project_id": proj.id}).json()
    assert legacy_label["found"]

    # Re-anchor on frame 2 with new geometry -> approved anchor, track active again
    ra = c.post(f"/api/tracks/{t1}/reanchor", json={"image_id": imgs[2].id, "bbox": [3, 3, 13, 13]}).json()
    assert ra["track_id"] == t1 and ra["status"] == "approved"
    tr = c.get(f"/api/tracks/{t1}").json()
    assert tr["status"] == "active" and tr["lost_at_image_id"] is None and tr["anchor_count"] == 2

    # Re-anchor by adopting an existing (untracked) annotation; replaces the drifted one on that frame
    ra2 = c.post(f"/api/tracks/{t1}/reanchor", json={"image_id": imgs[1].id, "annotation_id": u["id"]}).json()
    assert ra2["id"] == u["id"] and ra2["track_id"] == t1 and ra2["status"] == "approved"
    assert c.get(f"/api/annotations/{p['id']}").status_code == 404

    # Reassign / clear track via update
    assert c.put(f"/api/annotations/{ra2['id']}", json={"clear_track": True}).json()["track_id"] is None
    assert c.put(f"/api/annotations/{ra2['id']}", json={"track_id": t1}).json()["track_id"] == t1

    assert c.post(f"/api/tracks/{t1}/end").json()["status"] == "ended"
    assert c.patch(f"/api/tracks/{t1}", json={"status": "bogus"}).status_code == 400


def test_duplicate_check_reports_id_switch(project):
    from backend.api.ml import _check_duplicate, _find_duplicate_annotation

    proj, store = project
    box = store.upsert_label(proj.id, "box")
    cup = store.upsert_label(proj.id, "cup")
    img = store.list_images(proj.id)[1]
    t1 = store.create_track(proj.id, box.id)
    t2 = store.create_track(proj.id, box.id)
    store.create_annotation(img.id, box.id, [10, 10, 50, 50], track_id=t2.id)
    existing = store.list_annotations(img.id)
    bbox = [10, 10, 50, 50]

    # Track 1 lands on track 2's object -> reported, not dropped
    r = _check_duplicate(store, existing, bbox, None, box.id, 0.9, source_track_id=t1.id)
    assert not r.skipped and r.possible_id_switch and r.conflicting_track_id == t2.id

    # Same track already present -> plain duplicate
    r = _check_duplicate(store, existing, bbox, None, box.id, 0.9, source_track_id=t2.id)
    assert r.skipped and not r.possible_id_switch

    # Untracked source -> legacy behaviour (skip)
    assert _find_duplicate_annotation(store, existing, bbox, None, box.id, 0.9)[0] is True

    # Different label at same place -> conflict skip
    r = _check_duplicate(store, existing, bbox, None, cup.id, 0.9, source_track_id=t1.id)
    assert r.skipped and r.conflicting_label_name == "box"

    # Disabled threshold -> nothing
    assert not _check_duplicate(store, existing, bbox, np.zeros((1, 1), bool), box.id, 0, t1.id).skipped


def test_anchor_mask_area_uses_track(project):
    from backend.api.ml import _find_anchor_mask_area
    from core.masks import mask_to_rle

    proj, store = project
    box = store.upsert_label(proj.id, "box")
    imgs = store.list_images(proj.id)
    t1 = store.create_track(proj.id, box.id)
    t2 = store.create_track(proj.id, box.id)

    def m(n):
        a = np.zeros((100, 100), np.uint8)
        a[:n, :n] = 1
        return mask_to_rle(a)

    # t1 anchor is far away; t2 anchor sits right on top of the source -> label+proximity would pick t2
    store.create_annotation(imgs[0].id, box.id, [80, 80, 90, 90], mask_rle=m(10), track_id=t1.id)
    store.create_annotation(imgs[0].id, box.id, [0, 0, 20, 20], mask_rle=m(20), track_id=t2.id)
    src = store.create_annotation(imgs[1].id, box.id, [0, 0, 20, 20], status="pending",
                                  mask_rle=m(12), track_id=t1.id)
    assert _find_anchor_mask_area(store, src, imgs[1].id, None) == 100.0
