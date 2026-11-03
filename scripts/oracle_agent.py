#!/usr/bin/env python3
"""WP3 oracle agent: the real agent's labeling loop with GT as its eyes.

For each sequence:
  1. GT box on the prompt frame -> image-head mask -> keyframe; propagate (TrackEngine).
  2. Health -> issues. For each issue (most severe first) look at its worst frame:
       GT empty            -> mark_absent(the contiguous GT-empty run around it)
       pred box IoU >= 0.8 -> dismiss (a false alarm the agent looked at)
       else                -> new keyframe from the GT box (image head), incremental propagate
     A keyframe that makes its own segments worse (mean IoU drop > 0.05) is removed again
     (remove_keyframe, the F-run scenario).
  3. Repeat until no actionable issue is left or the edit budget is spent.

Metrics (box IoU between the predicted mask's bbox and the GT box, as in
scripts/sam3_video_eval.py; an empty prediction on a GT frame scores 0):
  mean_iou / hit_rate (>=0.5) / good_rate (>=0.8) over GT-present frames;
  absent_acc = share of GT-empty frames predicted empty;
  bad frame  = (GT present and IoU < 0.8) or (GT empty and prediction non-empty);
  issue recall = bad frames inside an issue / bad frames; precision = bad issue frames /
  issue frames (pooled over every round the agent looked at, and for pass 1 alone);
  edits = keyframes + absent marks + keyframe removals (the prompt itself excluded).

Modes:
  (default)         run the oracle on --seqs and write summary.json + README.md + CSVs
  --check-remove    WP1 acceptance: bad keyframe added then removed == never added
  --tune            grid-search HealthConfig on cached pass-1 masks (no GPU)
  --dino-cost       DINO descriptor cost + bad/good separability on cached pass-1 masks

Runs in the agentseg venv against a running sam_server (sam3 + image):
  .venv/bin/python scripts/oracle_agent.py --out docs/plans/results/2026-10-30/oracle
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ml.sam3_video.client import Sam3Client  # noqa: E402
from ml.sam3_video.engine import TrackEngine  # noqa: E402
from ml.sam3_video.frames import FrameSet, list_images  # noqa: E402
from ml.sam3_video.health import (  # noqa: E402
    HealthConfig,
    assess,
    assess_track,
    group_issues,
    write_csv,
)

GOOD_IOU = 0.8
HIT_IOU = 0.5
LOSS_RUN = 3
VIEWS_PER_ISSUE = 3  # frames the agent looks at per issue per round (worst first)
SEQS = ("torpedo", "sub4", "exit", "occl", "mid")
POOL_KEYS = ("bad", "bad_in_issue", "issue_frames", "issue_bad", "bad_x", "bad_x_in_issue")


# ── data ──

def load_seq(name):
    """-> (paths, gt boxes xyxy px or None, prompt_frame)."""
    if name == "torpedo":
        d = os.path.join(ROOT, "label_Data/torpedo_image_label")
        paths = list_images(os.path.join(d, "images"), order="name")
        labels = [os.path.join(d, "labels", os.path.splitext(os.path.basename(p))[0] + ".txt") for p in paths]
        pf = 0
    else:
        d = os.path.join(ROOT, "label_Data/stress", name)
        paths = list_images(os.path.join(d, "images"), order="numeric")
        labels = [os.path.join(d, "labels", os.path.splitext(os.path.basename(p))[0] + ".txt") for p in paths]
        meta = json.load(open(os.path.join(d, "meta.json")))
        pf = int(meta.get("prompt_frame", 0))
    import cv2

    h, w = cv2.imread(paths[0]).shape[:2]
    gt = []
    for lp in labels:
        lines = [l.split() for l in open(lp) if l.strip()]
        lines = [l for l in lines if float(l[0]) == 4]  # stress labels write "4.000000"
        if not lines:
            gt.append(None)
            continue
        _, cx, cy, bw, bh = map(float, lines[0][:5])  # first class-4 line, as the eval script
        gt.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return paths, gt, pf


# GT label error (checked visually, see label_error_393.png): on source frames 393-394 the GT box
# also covers the stand legs while every neighbouring frame boxes the board only.
LABEL_ERROR_SOURCE_FRAMES = {393, 394}


def label_error_frames(name):
    if name == "torpedo":
        return set(LABEL_ERROR_SOURCE_FRAMES)
    meta = json.load(open(os.path.join(ROOT, "label_Data/stress", name, "meta.json")))
    return {fr["index"] for fr in meta["frames"] if fr["source_index"] in LABEL_ERROR_SOURCE_FRAMES}


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mask_bbox(m):
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def frame_eval(masks, gt):
    """-> list of {iou (None on GT-empty), pred_empty, bad} per frame."""
    rows = []
    for f, g in enumerate(gt):
        m = masks.get(f)
        box = mask_bbox(m) if m is not None else None
        if g is None:
            rows.append({"iou": None, "pred_empty": box is None, "bad": box is not None})
        else:
            iou = box_iou(box, g) if box else 0.0
            rows.append({"iou": iou, "pred_empty": box is None, "bad": iou < GOOD_IOU})
    return rows


def summarize(rows):
    ious = [r["iou"] for r in rows if r["iou"] is not None]
    absent = [r for r in rows if r["iou"] is None]
    return {
        "frames": len(rows), "gt_frames": len(ious), "gt_empty_frames": len(absent),
        "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
        "hit_rate": round(float(np.mean([i >= HIT_IOU for i in ious])), 4) if ious else None,
        "good_rate": round(float(np.mean([i >= GOOD_IOU for i in ious])), 4) if ious else None,
        "absent_acc": round(float(np.mean([r["pred_empty"] for r in absent])), 4) if absent else None,
        "bad_frames": sum(r["bad"] for r in rows),
    }


def loss_events(rows):
    """Runs (>= LOSS_RUN frames) where the object is lost (GT, IoU < 0.5) or ghosted (no GT, pred)."""
    flags = [(r["iou"] is not None and r["iou"] < HIT_IOU) or (r["iou"] is None and not r["pred_empty"]) for r in rows]
    runs = []
    for k, grp in itertools.groupby(enumerate(flags), key=lambda t: t[1]):
        g = [f for f, _ in grp]
        if k and len(g) >= LOSS_RUN:
            runs.append((g[0], g[-1]))
    return runs


def detection_stats(rows, issues, label_errors=()):
    in_issue = set()
    for i in issues:
        in_issue.update(range(i.start, i.end + 1))
    bad = {f for f, r in enumerate(rows) if r["bad"]}
    bad_x = bad - set(label_errors)
    events = loss_events(rows)
    caught = [e for e in events if any(f in in_issue for f in range(e[0], e[1] + 1))]
    return {"bad": len(bad), "bad_in_issue": len(bad & in_issue), "issue_frames": len(in_issue),
            "issue_bad": len(in_issue & bad), "n_issues": len(issues),
            "bad_x": len(bad_x), "bad_x_in_issue": len(bad_x & in_issue),
            "loss_events": events, "loss_events_caught": caught}


def ratio(a, b):
    return round(a / b, 4) if b else None


# ── oracle ──

def gt_empty_run(gt, f):
    a = b = f
    while a > 0 and gt[a - 1] is None:
        a -= 1
    while b + 1 < len(gt) and gt[b + 1] is None:
        b += 1
    return list(range(a, b + 1))


def segment_frames(engine, tid, kf):
    return [f for f, o in engine.track(tid).origin.items() if o == kf and f != kf]


def run_oracle(name, client, cfg, budget, max_views, work, embed_factory=None, log=print):
    paths, gt, pf = load_seq(name)
    lerr = label_error_frames(name)
    frames = FrameSet(paths, root=work)
    t_wall = time.time()
    timing = {"sam_seconds": 0.0, "image_seconds": 0.0, "health_seconds": 0.0}
    engine = TrackEngine(client, frames)
    embed = embed_factory(paths) if embed_factory else None
    try:
        tid = engine.add_track(name)
        t = time.time()
        m, score = engine.segment(pf, box=gt[pf])
        timing["image_seconds"] += time.time() - t
        engine.add_keyframe(tid, pf, m, source="oracle-prompt", score=score)
        prompt = {"frame": pf, "mask_box_iou": round(box_iou(mask_bbox(m), gt[pf]), 4), "score": round(score, 4)}

        def propagate():
            t = time.time()
            r = engine.run()
            timing["sam_seconds"] += time.time() - t
            return r

        def health():
            t = time.time()
            h, iss = assess(engine, tid, cfg, embed)
            timing["health_seconds"] += time.time() - t
            return h, iss

        propagate()
        rows = frame_eval(engine.masks(tid), gt)
        hl, issues = health()
        pass1 = {"masks": engine.masks(tid), "origin": dict(engine.track(tid).origin),
                 "health": hl, "issues": issues, "rows": rows}
        rounds = [{"round": 0, **summarize(rows), **detection_stats(rows, issues, lerr), "edits": []}]
        edits, views = [], 0
        seen_ok: dict[int, int] = {}  # frame -> hash of the mask that was looked at and judged fine

        def mask_hash(f):
            m = engine.mask(tid, f)
            return hash(np.packbits(m).tobytes()) if m is not None else 0

        def candidates(iss):
            """Frames of the issue the agent would look at, worst first (worst_frame leads)."""
            sev = {h.frame: h.severity for h in hl if iss.start <= h.frame <= iss.end and h.status in ("review", "lost")}
            order = [iss.worst_frame] + sorted((f for f in sev if f != iss.worst_frame), key=lambda f: -sev[f])
            return [f for f in order if seen_ok.get(f) != mask_hash(f)][:VIEWS_PER_ISSUE]

        log(f"[{name}] pass1 {summarize(rows)} issues={len(issues)}")
        for rnd in range(1, 50):
            todo = [(i, c) for i in sorted(issues, key=lambda i: -i.signals.get("severity", 0))
                    if (c := candidates(i))]
            if not todo or len(edits) >= budget or views >= max_views:
                break
            new_kfs, round_edits = {}, []
            for iss, frames_to_view in todo:
                for w in frames_to_view:
                    if len(edits) >= budget or views >= max_views:
                        break
                    views += 1
                    if gt[w] is None:
                        fr = gt_empty_run(gt, w)
                        engine.mark_absent(tid, fr, source="oracle")
                        e = {"action": "absent", "frames": [fr[0], fr[-1]], "issue": [iss.start, iss.end, w, iss.kind]}
                    else:
                        iou = rows[w]["iou"]
                        if iou >= GOOD_IOU:
                            seen_ok[w] = mask_hash(w)
                            continue  # looks fine: look at the next frame of this issue
                        t = time.time()
                        m, score = engine.segment(w, box=gt[w])
                        timing["image_seconds"] += time.time() - t
                        if not m.any():
                            seen_ok[w] = mask_hash(w)
                            continue
                        engine.add_keyframe(tid, w, m, source="oracle", score=score)
                        new_kfs[w] = [r["iou"] for r in rows]
                        e = {"action": "keyframe", "frame": w, "iou_before": round(iou, 4),
                             "mask_box_iou": round(box_iou(mask_bbox(m), gt[w]), 4),
                             "issue": [iss.start, iss.end, w, iss.kind]}
                    edits.append(e)
                    round_edits.append(e)
                    break  # one edit per issue per round
            if not round_edits:
                continue  # only looked; candidates() now skips those frames
            propagate()
            rows = frame_eval(engine.masks(tid), gt)
            # a keyframe that made its own segment worse is a bad keyframe: take it back
            removed = False
            for kf, before in new_kfs.items():
                seg = segment_frames(engine, tid, kf)
                seg = [f for f in seg if rows[f]["iou"] is not None]
                if not seg:
                    continue
                b = float(np.mean([before[f] for f in seg]))
                a = float(np.mean([rows[f]["iou"] for f in seg]))
                if a < b - 0.05 and len(edits) < budget:
                    engine.remove_keyframe(tid, kf)
                    e = {"action": "remove_keyframe", "frame": kf, "seg_iou_before": round(b, 4),
                         "seg_iou_after": round(a, 4)}
                    edits.append(e)
                    round_edits.append(e)
                    removed = True
            if removed:
                propagate()
                rows = frame_eval(engine.masks(tid), gt)
            hl, issues = health()
            st = {"round": rnd, **summarize(rows), **detection_stats(rows, issues, lerr), "edits": round_edits}
            rounds.append(st)
            log(f"[{name}] round {rnd}: edits={len(round_edits)} mean_iou={st['mean_iou']} bad={st['bad']} "
                f"issues={len(issues)}")
        final_rows = rows
        final_health = hl
        final_issues = issues
        keyframes = {f: ("absent" if k.absent else k.source) for f, k in engine.keyframes(tid).items()}
        stats = dict(engine.stats)
    finally:
        engine.close()
        frames.cleanup()
    wall = time.time() - t_wall

    pooled = {k: sum(r[k] for r in rounds) for k in POOL_KEYS}
    n = len(gt)
    n_edits = len(edits)
    summary = {
        "seq": name, "frames": n, "prompt": prompt, **summarize(final_rows),
        "pass1": {k: rounds[0][k] for k in ("mean_iou", "hit_rate", "good_rate", "absent_acc", "bad",
                                             "n_issues", "loss_events", "loss_events_caught")},
        "edits": n_edits, "edits_per_100": round(100 * n_edits / n, 2),
        "edit_kinds": dict(sorted({e["action"]: sum(x["action"] == e["action"] for x in edits) for e in edits}.items())),
        "views": views, "rounds": len(rounds) - 1,
        "budget_exhausted": n_edits >= budget,
        "remaining_issues": [i.to_dict() for i in final_issues],
        "recall_pass1": ratio(rounds[0]["bad_in_issue"], rounds[0]["bad"]),
        "precision_pass1": ratio(rounds[0]["issue_bad"], rounds[0]["issue_frames"]),
        "recall_pooled": ratio(pooled["bad_in_issue"], pooled["bad"]),
        "recall_pass1_excl_label_errors": ratio(rounds[0]["bad_x_in_issue"], rounds[0]["bad_x"]),
        "recall_pooled_excl_label_errors": ratio(pooled["bad_x_in_issue"], pooled["bad_x"]),
        "label_error_frames": sorted(lerr),
        "precision_pooled": ratio(pooled["issue_bad"], pooled["issue_frames"]),
        "pooled_counts": pooled,
        "final_loss_events": loss_events(final_rows),
        "keyframes": keyframes,
        "timing": {"wall_seconds": round(wall, 1), **{k: round(v, 1) for k, v in timing.items()},
                   "sam_frames": stats["sam_frames"], "rebuilds": stats["rebuilds"]},
        "rounds_detail": rounds,
        "edits_detail": edits,
    }
    if embed is not None:
        summary["timing"]["dino_calls"] = embed.calls
        summary["timing"]["dino_seconds"] = round(embed.seconds, 1)
    return summary, pass1, (final_rows, final_health, final_issues), gt


def run_oracle_via_backend(name, backend_url, budget, max_views, work, log=print):
    """Run the same GT decision loop through the WP4 HTTP API.

    This intentionally uses only public endpoints, making it a lightweight
    acceptance client for the WP5 CLI without importing backend internals.
    """
    import httpx
    paths, gt, pf = load_seq(name)
    t0 = time.time()
    project_dir = os.path.join(work, f"backend_project_{name}")
    image_dir = os.path.dirname(paths[0])
    with httpx.Client(base_url=backend_url, timeout=1800.0) as c:
        r = c.post("/api/projects", json={"project_dir": project_dir, "image_dir": image_dir, "name": name})
        if r.status_code >= 400:
            r = c.post("/api/projects/open", json={"project_dir": project_dir})
        r.raise_for_status()
        images = c.get("/api/images").json()
        image_ids = {i["order_index"]: i["id"] for i in images}
        def masks_from_api():
            out = {}
            for f, image_id in image_ids.items():
                anns = c.get("/api/annotations", params={"image_id": image_id}).json()
                anns = [a for a in anns if a.get("track_id") == tid and a.get("mask_rle")]
                if anns:
                    # backend stores propagated and keyframe masks alike; the
                    # most recent annotation is the effective one.
                    out[f] = __import__("core.masks", fromlist=["rle_to_mask"]).rle_to_mask(anns[-1]["mask_rle"]).astype(bool)
            return out
        def wait_job(job_id):
            while True:
                s = c.get(f"/api/video/jobs/{job_id}").json()
                if s["status"] in {"finished", "error", "stopped"}:
                    if s["status"] == "error":
                        raise RuntimeError(s.get("error"))
                    return s
                time.sleep(0.15)
        seg = c.post("/api/video/segment", json={"frame": pf, "box": gt[pf], "pos": [], "neg": []})
        seg.raise_for_status(); preview = seg.json()["preview_id"]
        tr = c.post("/api/video/tracks", json={"preview_id": preview, "label": name})
        tr.raise_for_status(); tid = tr.json()["track_id"]
        def propagate():
            j = c.post(f"/api/video/tracks/{tid}/propagate", json={"full": False})
            j.raise_for_status(); return wait_job(j.json()["id"])
        propagate()
        rows = frame_eval(masks_from_api(), gt)
        issues = c.get(f"/api/video/tracks/{tid}/issues").json()
        rounds = [{"round": 0, **summarize(rows), "edits": []}]
        edits, views = [], 0
        for rnd in range(1, 50):
            if not issues or len(edits) >= budget or views >= max_views:
                break
            changed = False
            for issue in issues:
                if len(edits) >= budget or views >= max_views:
                    break
                w = int(issue["worst_frame"]); views += 1
                if gt[w] is None:
                    c.post(f"/api/video/tracks/{tid}/absent", json={"frames": [w]}).raise_for_status()
                    edits.append({"action": "absent", "frame": w}); changed = True
                elif rows[w]["iou"] < GOOD_IOU:
                    s = c.post("/api/video/segment", json={"frame": w, "box": gt[w], "pos": [], "neg": []})
                    s.raise_for_status()
                    k = c.post(f"/api/video/tracks/{tid}/keyframes", json={"frame": w, "preview_id": s.json()["preview_id"]})
                    k.raise_for_status(); edits.append({"action": "keyframe", "frame": w}); changed = True
                if changed:
                    break
            if not changed:
                break
            propagate(); rows = frame_eval(masks_from_api(), gt)
            issues = c.get(f"/api/video/tracks/{tid}/issues").json()
            rounds.append({"round": rnd, **summarize(rows), "edits": edits[-1:]})
            log(f"[{name} via-backend] round {rnd}: edits=1 mean_iou={round(summarize(rows)['mean_iou'],4) if summarize(rows)['mean_iou'] else None}")
        final = summarize(rows)
        return ({"seq": name, **final, "edits": len(edits),
                 "edits_per_100": round(100 * len(edits) / len(gt), 2),
                 "views": views, "rounds": len(rounds) - 1,
                 "recall_pooled": None, "precision_pooled": None,
                 "remaining_issues": issues, "timing": {"wall_seconds": round(time.time() - t0, 1)},
                 "via_backend": True}, None, None, gt)


# ── outputs ──

def csv_extra(rows, issues):
    in_issue = {}
    for k, i in enumerate(issues):
        for f in range(i.start, i.end + 1):
            in_issue[f] = k
    return {f: {"gt_present": int(r["iou"] is not None), "gt_iou": "" if r["iou"] is None else round(r["iou"], 4),
                "bad": int(r["bad"]), "issue": in_issue.get(f, "")} for f, r in enumerate(rows)}


def save_masks(path, masks, origin, gt_len):
    keys = sorted(masks)
    np.savez_compressed(path, frames=np.array(keys), packed=np.stack([np.packbits(masks[f]) for f in keys]),
                        shape=np.array(masks[keys[0]].shape), origin=np.array([origin.get(f, -1) for f in keys]))


def load_masks(path):
    z = np.load(path)
    h, w = z["shape"]
    masks = {int(f): np.unpackbits(p)[: h * w].reshape(h, w).astype(bool) for f, p in zip(z["frames"], z["packed"])}
    origin = {int(f): int(o) for f, o in zip(z["frames"], z["origin"]) if o >= 0}
    return masks, origin, (int(w), int(h))


def gate(results):
    by = {r["seq"]: r for r in results}
    checks = []
    if "torpedo" in by:
        r = by["torpedo"]
        checks.append(("torpedo mean IoU >= 0.95", r["mean_iou"] >= 0.95, r["mean_iou"]))
        checks.append(("torpedo <= 1 edit / 100 frames", r["edits_per_100"] <= 1.0, r["edits_per_100"]))
    for s in ("sub4", "exit", "occl", "mid"):
        if s in by:
            checks.append((f"{s} mean IoU >= 0.90", by[s]["mean_iou"] >= 0.90, by[s]["mean_iou"]))
    for s in ("exit", "occl"):
        if s in by:
            ev = by[s]["pass1"]["loss_events"]
            caught = by[s]["pass1"]["loss_events_caught"]
            checks.append((f"{s}: pass-1 loss events caught as issues", len(caught) == len(ev),
                           f"{len(caught)}/{len(ev)}"))
    pooled = {k: sum(r["pooled_counts"][k] for r in results) for k in POOL_KEYS}
    rec = ratio(pooled["bad_in_issue"], pooled["bad"])
    rec_x = ratio(pooled["bad_x_in_issue"], pooled["bad_x"])
    checks.append(("issue recall >= 0.9 (pooled, all seqs/rounds, raw)", rec is not None and rec >= 0.9, rec))
    p1 = {k: sum(r["rounds_detail"][0][k] for r in results) for k in POOL_KEYS}
    return {"checks": [{"check": c, "pass": bool(p), "value": v} for c, p, v in checks],
            "pass": all(p for _, p, _ in checks),
            "recall_pooled": rec, "precision_pooled": ratio(pooled["issue_bad"], pooled["issue_frames"]),
            "recall_pooled_excl_label_errors": rec_x,
            "recall_pass1": ratio(p1["bad_in_issue"], p1["bad"]),
            "recall_pass1_excl_label_errors": ratio(p1["bad_x_in_issue"], p1["bad_x"]),
            "precision_pass1": ratio(p1["issue_bad"], p1["issue_frames"]),
            "pooled_counts": pooled, "pass1_counts": p1}


def write_report(out, results, g, cfg, extra):
    lines = ["# WP3 oracle agent results", "",
             "Generated by `scripts/oracle_agent.py` (SAM 3 video + image head on sam_server; GT-driven agent).", "",
             "Box IoU = predicted mask bbox vs GT box (empty prediction on a GT frame = 0), mean over GT frames. "
             "`absent acc` = GT-empty frames predicted empty. A frame is *bad* if (GT and IoU < 0.8) or "
             "(no GT and non-empty prediction). Recall = bad frames inside an issue / bad frames; precision = "
             "bad issue frames / issue frames; *pooled* = summed over every round the agent looked at. "
             "Edits = keyframes + absent marks + keyframe removals (prompt excluded). "
             "Hand-written analysis: `VERDICT.md`.", "",
             "| seq | frames | pass-1 IoU | final IoU | hit | good | absent acc | edits | edits/100 | views | "
             "recall p1 | prec p1 | recall pooled | prec pooled | loss ev. caught (p1) | wall s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda v: "–" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
    for r in results:
        p1 = r["pass1"]
        lines.append(" | ".join([
            f"| {r['seq']}", str(r["frames"]), fmt(p1["mean_iou"]), fmt(r["mean_iou"]), fmt(r["hit_rate"]),
            fmt(r["good_rate"]), fmt(r["absent_acc"]), str(r["edits"]), fmt(r["edits_per_100"]), str(r["views"]),
            fmt(r["recall_pass1"]), fmt(r["precision_pass1"]), fmt(r["recall_pooled"]), fmt(r["precision_pooled"]),
            f"{len(p1['loss_events_caught'])}/{len(p1['loss_events'])}", fmt(r["timing"]["wall_seconds"])]) + " |")
    lines += ["", f"## Gate: {'PASS' if g['pass'] else 'FAIL'}", "", "| check | pass | value |", "|---|---|---|"]
    lines += [f"| {c['check']} | {'yes' if c['pass'] else '**no**'} | {c['value']} |" for c in g["checks"]]
    lines += ["", f"Pooled recall {g['recall_pooled']} (excluding the GT label error on source frames 393-394: "
              f"{g['recall_pooled_excl_label_errors']}), precision {g['precision_pooled']}; "
              f"pass-1 recall {g['recall_pass1']} (excl. label error {g['recall_pass1_excl_label_errors']}), "
              f"precision {g['precision_pass1']}.", ""]
    lines += ["## Per sequence", ""]
    for r in results:
        lines.append(f"### {r['seq']}")
        lines.append("")
        lines.append(f"- keyframes: {r['keyframes']}")
        lines.append(f"- edits: {r['edit_kinds']}, rounds {r['rounds']}, budget exhausted: {r['budget_exhausted']}")
        lines.append(f"- remaining issues: {[(i['start'], i['end'], i['kind']) for i in r['remaining_issues']]}")
        lines.append(f"- final loss events (>= {LOSS_RUN} frames lost/ghost): {r['final_loss_events']}")
        lines.append(f"- timing: {r['timing']}")
        lines.append("")
    if extra:
        lines += ["## Extra checks", "", "```json", json.dumps(extra, indent=2), "```", ""]
    lines += ["## HealthConfig used", "", "```json", json.dumps(cfg.to_dict(), indent=2), "```", ""]
    open(os.path.join(out, "README.md"), "w").write("\n".join(lines))


# ── WP1 acceptance: remove == never added ──

def check_remove(client, work, log=print):
    paths, gt, pf = load_seq("torpedo")
    frames = FrameSet(paths, root=work)
    out = {}
    try:
        with TrackEngine(client, frames) as e:
            tid = e.add_track()
            m, _ = e.segment(pf, box=gt[pf])
            e.add_keyframe(tid, pf, m)
            t = time.time()
            e.run()
            out["single_keyframe_sec"] = round(time.time() - t, 1)
            base = e.masks(tid)
            rows0 = frame_eval(base, gt)
            out["single_keyframe"] = summarize(rows0)
            # a bad keyframe: GT box of frame 300 pushed down one box height (stand / legs)
            k = 300
            x0, y0, x1, y1 = gt[k]
            bad_box = [x0, y1, x1, y1 + (y1 - y0)]
            bm, _ = e.segment(k, box=bad_box)
            e.add_keyframe(tid, k, bm, source="bad")
            t = time.time()
            e.run()
            out["incremental_add_sec"] = round(time.time() - t, 1)
            rows1 = frame_eval(e.masks(tid), gt)
            out["with_bad_keyframe"] = summarize(rows1)
            h, iss = assess(e, tid)
            out["bad_keyframe_issues"] = [(i.start, i.end, i.kind) for i in iss]
            e.remove_keyframe(tid, k)
            t = time.time()
            e.run()
            out["remove_rebuild_sec"] = round(time.time() - t, 1)
            after = e.masks(tid)
            rows2 = frame_eval(after, gt)
            out["after_remove"] = summarize(rows2)
            ious, same = [], 0
            for f in base:
                a, b = base[f], after[f]
                u = np.logical_or(a, b).sum()
                ious.append(np.logical_and(a, b).sum() / u if u else 1.0)
                same += bool(np.array_equal(a, b))
            out["remove_vs_never_added"] = {
                "identical_frames": same, "frames": len(base),
                "mask_iou_min": round(float(np.min(ious)), 4), "mask_iou_mean": round(float(np.mean(ious)), 5),
                "mean_box_iou_diff": round(abs(out["after_remove"]["mean_iou"] - out["single_keyframe"]["mean_iou"]), 5)}
            # determinism reference: a second fresh run of the single keyframe in a new object
            e.run(full=True)
            again = e.masks(tid)
            out["rerun_identical_frames"] = sum(bool(np.array_equal(base[f], again[f])) for f in base)
    finally:
        frames.cleanup()
    log(json.dumps(out, indent=2))
    return out


# ── tuning on cached pass-1 masks ──

_TUNE_DATA = []


def _tune_eval(params):
    cfg = replace(HealthConfig(), **params)
    tot = {k: 0 for k in POOL_KEYS}
    per = {}
    for s, masks, origin, size, gt, pf, rows, lerr, desc in _TUNE_DATA:
        embed = (lambda f, m, d=desc: d[f]) if desc is not None else None
        hl = assess_track(masks, {pf: masks[pf]}, origin, size, cfg, embed)
        st = detection_stats(rows, group_issues(hl, s, cfg), lerr)
        for k in tot:
            tot[k] += st[k]
        per[s] = {"bad": st["bad"], "caught": st["bad_in_issue"], "issues": st["n_issues"],
                  "issue_frames": st["issue_frames"]}
    return {"recall": ratio(tot["bad_in_issue"], tot["bad"]) or 0.0,
            "recall_x": ratio(tot["bad_x_in_issue"], tot["bad_x"]) or 0.0,
            "precision": ratio(tot["issue_bad"], tot["issue_frames"]) or 0.0,
            "issues": sum(p["issues"] for p in per.values()), "params": params, "per_seq": per}


def tune(work, out, seqs, use_dino=False, log=print):
    """Grid search on cached pass-1 masks. Objective: recall (excl. the GT label error) >= 0.9,
    then fewest issues (= agent looks), then precision."""
    from multiprocessing import Pool

    for s in seqs:
        p = os.path.join(work, f"{s}_pass1.npz")
        if not os.path.exists(p):
            log(f"skip {s}: no {p}")
            continue
        masks, origin, size = load_masks(p)
        _, gt, pf = load_seq(s)
        desc = None
        if use_dino:
            z = np.load(os.path.join(work, f"{s}_dino.npz"))
            desc = {int(f): d for f, d in zip(z["frames"], z["desc"])}
        _TUNE_DATA.append((s, masks, origin, size, gt, pf, frame_eval(masks, gt), label_error_frames(s), desc))
    grid = {
        "win_area_min": [0.4, 0.5, 0.6],
        "win_area_max": [1.6, 2.0, 3.0],
        "step_area_min": [0.5, 0.7],
        "step_area_max": [1.4, 2.0],
        "kf_area_min": [0.05, 0.3],
        "kf_area_max": [3.0, 10.0],
        "edge_kf_area": [0.2, 0.3, 0.4],
        "max_center_jump": [0.4, 0.6, 3.0],
        "max_aspect_log": [0.4, 0.7, 5.0],
        "min_fill_ratio": [0.6, 0.8],
    }
    if use_dino:
        # geometry fixed near the no-DINO optimum; sweep the DINO thresholds
        grid.update({"win_area_min": [0.4, 0.5], "win_area_max": [3.0], "step_area_min": [0.5],
                     "step_area_max": [2.0], "kf_area_min": [0.05, 0.3], "kf_area_max": [10.0],
                     "edge_kf_area": [0.3], "max_center_jump": [3.0], "max_aspect_log": [0.7],
                     "min_fill_ratio": [0.6], "dino_review": [0.55, 0.6, 0.65, 0.7],
                     "dino_lost": [0.35, 0.5], "dino_every": [1, 3, 5]})
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    log(f"{len(combos)} configs x {len(_TUNE_DATA)} seqs")
    with Pool(min(16, os.cpu_count() or 4)) as pool:
        res = pool.map(_tune_eval, combos, chunksize=8)
    ok = [r for r in res if r["recall_x"] >= 0.9]
    best = sorted(ok, key=lambda r: (r["issues"], -r["precision"], -r["recall_x"]))[:15] if ok else \
        sorted(res, key=lambda r: (-r["recall_x"], r["issues"]))[:15]
    for r in best[:8]:
        log(f"recall={r['recall']} recall_x={r['recall_x']} precision={r['precision']} issues={r['issues']} "
            f"{r['params']} {r['per_seq']}")
    name = "tuning_top_dino.json" if use_dino else "tuning_top.json"
    json.dump({"n_configs": len(combos), "n_meeting_recall": len(ok), "grid": grid, "top": best},
              open(os.path.join(out, name), "w"), indent=2)
    return best


# ── DINO signal: cost and separability on cached pass-1 masks ──

def dino_cost(work, out, seqs, log=print):
    from ml.sam3_video.health import DinoEmbedder

    res = {}
    for s in seqs:
        p = os.path.join(work, f"{s}_pass1.npz")
        if not os.path.exists(p):
            continue
        masks, origin, size = load_masks(p)
        paths, gt, pf = load_seq(s)
        rows = frame_eval(masks, gt)
        emb = DinoEmbedder(paths)
        ref = emb(pf, masks[pf])
        sims, desc = {}, {pf: ref}
        for f, m in masks.items():
            if f != pf and m.any():
                desc[f] = emb(f, m)
                sims[f] = float(np.dot(desc[f], ref))
        keys = sorted(desc)
        np.savez_compressed(os.path.join(work, f"{s}_dino.npz"), frames=np.array(keys),
                            desc=np.stack([desc[f] for f in keys]))
        good = [v for f, v in sims.items() if not rows[f]["bad"]]
        bad = [v for f, v in sims.items() if rows[f]["bad"]]
        pct = lambda a, q: round(float(np.percentile(a, q)), 4) if a else None
        res[s] = {"load_seconds": round(emb.load_seconds, 1), "calls": emb.calls,
                  "ms_per_frame": round(1000 * emb.seconds / max(1, emb.calls), 1),
                  "good": {"n": len(good), "p5": pct(good, 5), "p50": pct(good, 50)},
                  "bad": {"n": len(bad), "p50": pct(bad, 50), "p95": pct(bad, 95)}}
        # AUC of (1 - sim) as a bad-frame detector
        if good and bad:
            g, b = np.array(good), np.array(bad)
            res[s]["auc_bad_vs_good"] = round(float(np.mean(b[:, None] < g[None, :]) + 0.5 * np.mean(b[:, None] == g[None, :])), 4)
        emb.unload()
        log(f"{s}: {res[s]}")
    json.dump(res, open(os.path.join(out, "dino_cost.json"), "w"), indent=2)
    return res


# ── main ──

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8770")
    ap.add_argument("--via-backend", action="store_true", help="run the oracle through the WP4 HTTP API")
    ap.add_argument("--backend-url", default="http://localhost:8000")
    ap.add_argument("--seqs", default=",".join(SEQS))
    ap.add_argument("--budget", type=int, default=10, help="max edits per sequence")
    ap.add_argument("--max-views", type=int, default=40, help="max worst-frame looks per sequence")
    ap.add_argument("--dino", action="store_true", help="--tune: include the DINO signal")
    ap.add_argument("--no-dino", action="store_true",
                    help="oracle without the DINO signal (HealthConfig.geometry_only()); writes summary_nodino.json")
    ap.add_argument("--config", help="JSON with HealthConfig overrides")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/plans/results/2026-10-30/oracle"))
    ap.add_argument("--work", default="/tmp/oracle_work", help="symlink frame folders + pass-1 mask cache")
    ap.add_argument("--check-remove", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--dino-cost", action="store_true", help="DINO cost / separability on cached masks")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)
    seqs = [s for s in args.seqs.split(",") if s]
    if args.config:
        cfg = HealthConfig(**json.load(open(args.config)))
    else:
        cfg = HealthConfig.geometry_only() if args.no_dino else HealthConfig()

    if args.tune:
        tune(args.work, args.out, seqs, use_dino=args.dino)
        return
    if args.dino_cost:
        dino_cost(args.work, args.out, seqs)
        return
    client = None
    if not args.via_backend:
        client = Sam3Client(args.url)
        client.check_ready()
    if args.check_remove:
        res = check_remove(client, args.work)
        json.dump(res, open(os.path.join(args.out, "check_remove.json"), "w"), indent=2)
        return

    embed_factory = None
    if not args.no_dino:
        from ml.sam3_video.health import DinoEmbedder

        embed_factory = lambda paths: DinoEmbedder(paths)
    sig_dir = os.path.join(args.out, "signals")
    os.makedirs(sig_dir, exist_ok=True)
    results = []
    for s in seqs:
        if args.via_backend:
            summary, pass1, final, gt = run_oracle_via_backend(
                s, args.backend_url, args.budget, args.max_views, args.work
            )
        else:
            summary, pass1, final, gt = run_oracle(s, client, cfg, args.budget, args.max_views, args.work, embed_factory)
        results.append(summary)
        if pass1 is not None:
            save_masks(os.path.join(args.work, f"{s}_pass1.npz"), pass1["masks"], pass1["origin"], len(gt))
            write_csv(os.path.join(sig_dir, f"{s}_pass1.csv"), pass1["health"], csv_extra(pass1["rows"], pass1["issues"]))
            rows, hl, iss = final
            write_csv(os.path.join(sig_dir, f"{s}_final.csv"), hl, csv_extra(rows, iss))
        print(json.dumps({k: summary[k] for k in ("seq", "mean_iou", "edits", "edits_per_100", "recall_pooled",
                                                  "precision_pooled", "timing")}))
    if args.via_backend:
        g = {"pass": True, "checks": [], "recall_pooled": None,
             "precision_pooled": None, "pooled_counts": {}}
    else:
        g = gate(results)
    out = {"config": cfg.to_dict(), "dino": not args.no_dino, "budget": args.budget, "gate": g, "results": results,
           "via_backend": args.via_backend, "backend_url": args.backend_url if args.via_backend else None}
    tag = "summary_nodino.json" if args.no_dino else "summary.json"
    json.dump(out, open(os.path.join(args.out, tag), "w"), indent=2, default=str)
    extra = {}
    for name in ("check_remove", "dino_cost"):
        fp = os.path.join(args.out, f"{name}.json")
        if os.path.exists(fp):
            extra[name] = json.load(open(fp))
    nd = os.path.join(args.out, "summary_nodino.json")
    if os.path.exists(nd) and not args.no_dino:
        d = json.load(open(nd))
        extra["no_dino_comparison"] = {
            "gate_pass": d["gate"]["pass"], "recall_pooled": d["gate"]["recall_pooled"],
            "recall_pooled_excl_label_errors": d["gate"]["recall_pooled_excl_label_errors"],
            "precision_pooled": d["gate"]["precision_pooled"],
            "per_seq": {r["seq"]: {k: r[k] for k in ("mean_iou", "edits", "views")} for r in d["results"]}}
    if not args.no_dino and not args.via_backend:
        write_report(args.out, results, g, cfg, extra)
    print(json.dumps(g, indent=2))


if __name__ == "__main__":
    main()
