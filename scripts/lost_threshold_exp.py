"""How long can SAM 3 lose an object and still pick it up again? Finds the best `--max-lost`.

For each sequence (`<i>.jpg` + YOLO box GT, see lvos_to_seq.py): one keyframe from the GT
box on the frame where the object is largest, tracked both ways with no stop (max_lost=0).
Then every run of empty masks is classified by what SAM does right after it:
  recovered  - the next non-empty masks are on the object (box IoU >= 0.5)
  wrong      - SAM comes back on something else
  never      - SAM stays empty to the end of the sequence
and whether the object really was gone (GT empty) during the run.

    python scripts/lost_threshold_exp.py track <seq_dir>... --out runs.json   # needs sam_server
    python scripts/lost_threshold_exp.py analyze runs.json                     # thresholds table
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.masks import mask_to_bbox  # noqa: E402

PROBE = 5  # non-empty frames after a run that decide its outcome


def gt_boxes(seq: Path) -> list:
    from PIL import Image
    w, h = Image.open(seq / "images" / "0.jpg").size
    out = []
    for i in range(len(list((seq / "images").glob("*.jpg")))):
        g = (seq / "labels" / f"{i}.txt").read_text().split()
        if not g:
            out.append(None); continue
        cx, cy, bw, bh = map(float, g[1:5])
        out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def track(seqs: list[str], out: str, url: str):
    from ml.sam3_video import FrameSet, Sam3Client, TrackEngine
    results = json.loads(Path(out).read_text()) if Path(out).exists() else {}
    for s in seqs:
        seq = Path(s)
        if seq.name in results:
            continue
        gt = gt_boxes(seq)
        n = len(gt)
        kf = max((i for i in range(n) if gt[i]), key=lambda i: (gt[i][2] - gt[i][0]) * (gt[i][3] - gt[i][1]))
        client = Sam3Client(url)
        frames = FrameSet([str(seq / "images" / f"{i}.jpg") for i in range(n)],
                          root=tempfile.mkdtemp(prefix="lostexp_"))
        eng = TrackEngine(client, frames, offload_video_to_cpu=True)
        t0 = time.time()
        try:
            mask, _ = client.segment(frames.path(kf), box=gt[kf], points=[], point_labels=[])
            eng.add_track(1)
            eng.add_keyframe(1, kf, np.asarray(mask, bool))
            eng.run(max_lost=0)
            t = eng.track(1)
            boxes = [mask_to_bbox(t.masks[i]) if i in t.masks and t.masks[i].any() else None for i in range(n)]
        finally:
            eng.close()
        results[seq.name] = {"keyframe": kf, "gt": gt, "pred": boxes, "seconds": round(time.time() - t0, 1)}
        Path(out).write_text(json.dumps(results))
        print(f"{seq.name}: {n} frames, keyframe {kf}, {results[seq.name]['seconds']}s", file=sys.stderr)


def iou(a, b) -> float:
    if not a or not b:
        return 0.0
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def empty_runs(r: dict) -> list[dict]:
    """Runs of empty masks in tracking order (away from the keyframe), with their outcome."""
    gt, pred, kf = r["gt"], r["pred"], r["keyframe"]
    out = []
    for direction, order in (("forward", range(kf + 1, len(gt))), ("backward", range(kf - 1, -1, -1))):
        order = list(order)
        i = 0
        while i < len(order):
            if pred[order[i]] is not None:
                i += 1; continue
            j = i
            while j < len(order) and pred[order[j]] is None:
                j += 1
            frames = order[i:j]
            after = [f for f in order[j:] if pred[f] is not None][:PROBE]
            if not after:
                outcome = "never"
            else:
                hits = sum(iou(pred[f], gt[f]) >= 0.5 for f in after)
                outcome = "recovered" if hits * 2 > len(after) else "wrong"
            gone = sum(gt[f] is None for f in frames) / len(frames)
            out.append({"direction": direction, "start": frames[0], "len": len(frames),
                        "outcome": outcome, "gt_gone": round(gone, 2),
                        "frames_left": len(order) - j})
            i = j
    return out


def analyze(path: str, thresholds: list[int]):
    results = json.loads(Path(path).read_text())
    runs = []
    for name, r in results.items():
        pred, gt = r["pred"], r["gt"]
        good = sum(iou(p, g) >= 0.5 for p, g in zip(pred, gt) if g)
        print(f"{name}: {len(gt)} frames, keyframe {r['keyframe']}, {r['seconds']}s, "
              f"hit rate {good / max(1, sum(g is not None for g in gt)):.2f}")
        for e in empty_runs(r):
            e["seq"] = name
            runs.append(e)
            if e["len"] >= 3:
                print(f"   {e['direction']:8s} empty from {e['start']:4d} for {e['len']:3d} frames -> "
                      f"{e['outcome']:9s} (gt gone {e['gt_gone']:.0%}, {e['frames_left']} frames after)")
    print("\nN = max-lost. false stops: runs SAM would have recovered from but N cuts them "
          "(frames after the cut need an agent keyframe);\ncaught: runs it never/wrongly came "
          "back from, cut at N; wasted: empty frames tracked before the cut in those runs.")
    print(f"{'N':>4} {'false stops':>12} {'frames lost to them':>20} {'caught':>7} {'wasted frames':>14}")
    for n in thresholds:
        false = [e for e in runs if e["outcome"] == "recovered" and e["len"] >= n]
        bad = [e for e in runs if e["outcome"] != "recovered"]
        caught = [e for e in bad if e["len"] >= n]
        wasted = sum(min(e["len"], n) for e in bad)
        print(f"{n:>4} {len(false):>12} {sum(e['frames_left'] + e['len'] - n for e in false):>20} "
              f"{len(caught):>4}/{len(bad):<2} {wasted:>14}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("track"); t.add_argument("seqs", nargs="+"); t.add_argument("--out", required=True)
    t.add_argument("--url", default="http://localhost:8770")
    a = sub.add_parser("analyze"); a.add_argument("runs")
    a.add_argument("--thresholds", default="12,24,36,48,72,96,144,200")
    args = ap.parse_args()
    if args.cmd == "track":
        track(args.seqs, args.out, args.url)
    else:
        analyze(args.runs, [int(x) for x in args.thresholds.split(",")])


if __name__ == "__main__":
    main()
