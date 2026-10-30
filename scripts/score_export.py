"""Score an `autoseg export` (YOLO-seg or YOLO-detect labels) against YOLO box GT.

Same definitions as scripts/oracle_agent.py: box IoU between the predicted
mask's bounding box and the GT box, averaged over frames with GT; a frame with
empty GT counts as correct when the prediction is empty too.

    python scripts/score_export.py <export_dir> <gt_labels_dir> [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _boxes(path: Path) -> list[tuple[float, float, float, float]]:
    """Normalized xyxy boxes from a YOLO detect (cls cx cy w h) or seg (cls x y x y ...) file."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().split("\n"):
        v = [float(x) for x in line.split()[1:]]
        if len(v) == 4:
            cx, cy, w, h = v
            out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        elif len(v) >= 6:
            xs, ys = v[0::2], v[1::2]
            out.append((min(xs), min(ys), max(xs), max(ys)))
    return out


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def score(export_dir: Path, gt_dir: Path) -> dict:
    pred_files = {p.stem: p for p in export_dir.rglob("*.txt") if p.parent.name != "classes"}
    ious, bad, absent_ok, absent_n, missing = [], [], 0, 0, 0
    gt_files = sorted(gt_dir.glob("*.txt"))
    for g in gt_files:
        gt, pred = _boxes(g), _boxes(pred_files.get(g.stem, Path("/nonexistent")))
        if g.stem not in pred_files:
            missing += 1
        if not gt:
            absent_n += 1
            absent_ok += not pred
            if pred:
                bad.append(g.stem)
            continue
        iou = _iou(_union(gt), _union(pred)) if pred else 0.0
        ious.append(iou)
        if iou < 0.8:
            bad.append(g.stem)
    n = len(ious)
    return {"frames": len(gt_files), "gt_frames": n, "gt_empty_frames": absent_n,
            "mean_iou": round(sum(ious) / n, 4) if n else None,
            "hit_rate": round(sum(i >= 0.5 for i in ious) / n, 4) if n else None,
            "good_rate": round(sum(i >= 0.8 for i in ious) / n, 4) if n else None,
            "absent_acc": round(absent_ok / absent_n, 4) if absent_n else None,
            "bad_frames": len(bad), "missing_label_files": missing, "bad_examples": bad[:20]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir", type=Path); ap.add_argument("gt_labels_dir", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    res = score(a.export_dir, a.gt_labels_dir)
    if a.json:
        a.json.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
