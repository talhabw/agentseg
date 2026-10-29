"""Tracking uncertainty (issue #5): size change flags needs_review, not lost."""

import numpy as np
import pytest
import torch

from ml.propagate import (
    STATUS_NEEDS_REVIEW,
    STATUS_OK,
    PropagateService,
    PropagationNotFoundError,
    PropagationSizeMismatchError,
    update_area_trend,
)

H = W = 200


def rect_mask(x1, y1, x2, y2):
    m = np.zeros((H, W), dtype=np.uint8)
    m[y1:y2, x1:x2] = 1
    return m


class StubEmbed:
    _img_size = 16
    _patch_size = 4

    def __init__(self, sim):
        self.sim = sim

    def get_object_descriptor(self, *a, **k):
        v = np.ones(8, dtype=np.float32)
        return v / np.linalg.norm(v)

    def get_image_features(self, *a, **k):
        return torch.rand(1, 16, 8)

    def compute_similarity(self, a, b):
        return self.sim


class StubSegment:
    """Always returns the same target mask, whatever the prompt."""

    def __init__(self, mask, score=0.9):
        self.mask = mask
        self.score = score

    def set_image(self, *a, **k):
        pass

    def _out(self):
        ys, xs = np.nonzero(self.mask)
        bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        return self.mask.copy(), self.score, bbox

    def segment_with_point(self, *a, **k):
        return self._out()

    def segment_with_bbox(self, *a, **k):
        return self._out()


def run(target_mask, sim=0.9, source_mask=None, **kw):
    if source_mask is None:
        source_mask = rect_mask(50, 50, 90, 90)  # 40x40
    ys, xs = np.nonzero(source_mask)
    source_bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    svc = PropagateService(StubEmbed(sim), StubSegment(target_mask))
    img = np.zeros((H, W, 3), dtype=np.uint8)
    return svc.propagate_annotation(
        source_image=img,
        source_bbox=source_bbox,
        source_mask=source_mask,
        target_image=img,
        prune_thin_artifacts=False,
        **kw,
    )


def test_consistent_result_is_ok():
    r = run(rect_mask(52, 50, 92, 90))
    assert r.status == STATUS_OK
    assert r.reasons == []
    assert r.signals.mask_area_ratio == pytest.approx(1.0)


def test_size_change_flags_needs_review_instead_of_stopping():
    r = run(rect_mask(40, 40, 120, 120))  # 4x area
    assert r.status == STATUS_NEEDS_REVIEW
    assert "size_change" in r.reasons
    d = r.uncertainty()
    assert d["status"] == "needs_review"
    for key in (
        "bbox_area_ratio",
        "mask_area_ratio",
        "embed_similarity",
        "sam_score",
        "center_jump",
    ):
        assert key in d["signals"]
    assert d["signals"]["mask_area_ratio"] == pytest.approx(4.0)


def test_legacy_stop_on_size_mismatch_still_raises():
    with pytest.raises(PropagationSizeMismatchError) as ei:
        run(rect_mask(40, 40, 120, 120), stop_on_size_mismatch=True)
    assert ei.value.signals is not None


def test_low_similarity_everywhere_is_lost():
    with pytest.raises(PropagationNotFoundError) as ei:
        run(rect_mask(50, 50, 90, 90), sim=0.2)
    assert ei.value.reason == "low_similarity"
    assert ei.value.to_dict()["status"] == "lost"


def test_vanished_object_is_lost():
    with pytest.raises(PropagationNotFoundError) as ei:
        run(rect_mask(0, 0, 3, 3))  # 9 px vs 1600 px
    assert ei.value.reason == "object_vanished"


def test_rotation_uses_mask_area_not_bbox_area():
    # Thin diagonal-ish object: big bbox, small mask. Target: same pixel count, compact.
    src = np.zeros((H, W), dtype=np.uint8)
    for i in range(100):
        src[50 + i, 50 + i : 54 + i] = 1  # 400 px, bbox 103x100
    tgt = rect_mask(80, 80, 100, 100)  # 400 px, bbox 20x20
    r = run(tgt, source_mask=src)
    assert r.signals.bbox_area_ratio < 0.1
    assert r.signals.mask_area_ratio == pytest.approx(1.0)
    assert "size_change" not in r.reasons


def test_expected_area_trend_accepts_steady_growth():
    tgt = rect_mask(40, 40, 120, 120)  # 4x
    assert "size_change" not in run(tgt, expected_area_ratio=3.0).reasons


def test_anchor_drift_flagged():
    r = run(rect_mask(52, 50, 92, 90), anchor_mask_area=100.0)  # 16x anchor
    assert r.reasons == ["anchor_size_drift"]
    assert r.signals.anchor_area_ratio == pytest.approx(16.0)


def test_update_area_trend():
    assert update_area_trend(1.0, None) == 1.0
    t = update_area_trend(1.0, 1.5)
    assert 1.0 < t < 1.5
    assert update_area_trend(1.0, 100.0, alpha=1.0) == 2.0  # clamped


# ---- backend helpers ----


def test_find_anchor_prefers_nearest_approved_same_label():
    from types import SimpleNamespace as NS

    from backend.api.ml import _find_anchor_mask_area
    from core.masks import mask_to_rle

    imgs = [NS(id=i, project_id=1, order_index=i) for i in range(4)]
    near = NS(label_id=7, status="approved", bbox_xyxy=[50, 50, 90, 90],
              mask_rle=mask_to_rle(rect_mask(50, 50, 90, 90)))
    far = NS(label_id=7, status="approved", bbox_xyxy=[0, 0, 10, 10],
             mask_rle=mask_to_rle(rect_mask(0, 0, 10, 10)))
    other = NS(label_id=8, status="approved", bbox_xyxy=[50, 50, 60, 60],
               mask_rle=mask_to_rle(rect_mask(50, 50, 60, 60)))
    anns = {0: [], 1: [far, near, other], 2: []}
    store = NS(
        get_image_by_id=lambda i: imgs[i],
        list_images=lambda pid: imgs,
        list_annotations=lambda i: anns.get(i, []),
        get_annotation_by_id=lambda i: None,
    )
    src = NS(label_id=7, status="pending", bbox_xyxy=[52, 50, 92, 90], mask_rle=None)
    assert _find_anchor_mask_area(store, src, 3, None) == 1600.0

    src.status = "approved"
    src.mask_rle = mask_to_rle(rect_mask(0, 0, 20, 20))
    assert _find_anchor_mask_area(store, src, 3, None) == 400.0


def test_tracking_fields_shape():
    from backend.api.ml import _tracking_fields

    r = run(rect_mask(40, 40, 120, 120))
    f = _tracking_fields(r, 1.0)
    assert f["status"] == "needs_review"
    assert "size_change" in f["uncertain_reasons"]
    assert f["signals"]["mask_area_ratio"] == pytest.approx(4.0)
    assert 1.0 < f["next_expected_area_ratio"] <= 2.0
