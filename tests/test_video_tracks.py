"""HTTP contract tests for the SAM3 video backend using the GPU-free fake client."""
import sqlite3
import time

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from core.store import ProjectStore
from tests.fake_sam3 import FakeSam3Client


def test_video_track_lifecycle(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(5):
        Image.new("RGB", (32, 24), (10, 20, 30)).save(image_dir / f"{i}.png")
    project = ProjectStore.create_project(str(tmp_path / "project"), str(image_dir), "video")
    store = ProjectStore(project.db_path)
    store.close()
    store._conn = sqlite3.connect(project.db_path, check_same_thread=False)
    store._conn.row_factory = sqlite3.Row
    store._conn.execute("PRAGMA foreign_keys = ON")

    import backend.api.projects as projects_api
    import backend.api.video_tracks as video_api
    from backend.main import app

    projects_api._current_project = project
    projects_api._current_store = store
    fake = FakeSam3Client(5, 24, 32, dx=0)
    monkeypatch.setattr(video_api, "_client", lambda: fake)
    # A fresh registry avoids leaking a session between tests.
    video_api._ENGINES.clear()
    video_api._JOBS.clear()
    try:
        with TestClient(app) as client:
            preview = client.post("/api/video/segment", json={
                "frame": 0, "box": [2, 3, 10, 12], "pos": [[5, 6]], "neg": []
            })
            assert preview.status_code == 200
            preview_json = preview.json()
            assert preview_json["area"] == 72 and preview_json["preview_id"]

            label = store.upsert_label(project.id, "object")
            created = client.post("/api/video/tracks", json={
                "preview_id": preview_json["preview_id"], "label_id": label.id
            })
            assert created.status_code == 200
            track_id = created.json()["track_id"]

            started = client.post(f"/api/video/tracks/{track_id}/propagate", json={})
            assert started.status_code == 200
            job_id = started.json()["id"]
            for _ in range(100):
                status = client.get(f"/api/video/jobs/{job_id}").json()
                if status["status"] in {"finished", "error"}:
                    break
                time.sleep(0.01)
            assert status["status"] == "finished"
            assert status["counts"]["frames"] == 4
            anns = [a for a, _ in store.list_track_annotations(track_id)]
            assert sum(a.source == "propagated" for a in anns) == 4
            assert client.get(f"/api/video/tracks/{track_id}/issues").status_code == 200
            assert client.get("/api/video/render/frame", params={"frame": 1, "grid": True}).headers["content-type"] == "image/png"
            assert client.get("/api/video/render/sheet", params={"every": 2}).headers["content-type"] == "image/png"
    finally:
        projects_api._current_project = None
        projects_api._current_store = None
        for engine in video_api._ENGINES.values():
            engine.close()
        video_api._ENGINES.clear()
