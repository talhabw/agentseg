#!/usr/bin/env python3
"""Evaluate SAM 3 / 3.1 native video tracking (detect + track) on a YOLO-labeled sequence.

Client only: talks HTTP to a running sam_server. Needs the sam_server venv (httpx, numpy,
opencv, pycocotools, sam_server.codec):

  cd sam_server && uv run --group dev python ../scripts/sam3_video_eval.py \
      --frames /path/frames --labels ../label_Data/torpedo_image_label/labels \
      --text "white board" --exemplar-gt --name B_white_board --out /tmp/exp

Prompt modes (on --prompt-frame, GT box of that frame):
  --text T             text prompt (detector finds all instances, tracker tracks them)
  --exemplar-gt        GT box as positive box exemplar (with or without --text)
  --points-gt          positive click at GT box centre with obj_id 1 (tracker only, no text/box)
  --mask-gt            (model sam3) mask from image predict_inst(box=GT), then video add_mask
--correct-at auto|IDX: after the first pass, add one corrective prompt (same kind as the
initial prompt, or --correct-with) with the GT of that frame and re-propagate forward from
it; 'auto' = first loss frame, repeated up to --max-corrections times.

--frames must hold <int>.jpg files whose i-th entry (numeric order) matches the i-th label
file (sorted by name) in --labels. Box IoU is computed in pixels against the FIRST label line.
"""

import argparse
import csv
import json
import os
import re
import sys
import time

import cv2
import httpx
import numpy as np

from sam_server.codec import decode, encode

HIT_IOU = 0.5
FP_IOU = 0.1
LOSS_RUN = 5


# ── data ──

def list_frames(frames_dir):
    names = [f for f in os.listdir(frames_dir) if re.fullmatch(r"\d+\.(jpg|jpeg|png)", f, re.I)]
    names.sort(key=lambda f: int(f.split(".")[0]))
    return [os.path.join(frames_dir, f) for f in names]


def load_gt(labels_dir, n, w, h):
    files = sorted(f for f in os.listdir(labels_dir) if f.endswith(".txt"))
    if len(files) != n:
        sys.exit(f"{len(files)} label files vs {n} frames")
    gt = []
    for f in files:
        lines = [l.split() for l in open(os.path.join(labels_dir, f)) if l.strip()]
        if not lines:
            gt.append(None)
            continue
        _, cx, cy, bw, bh = map(float, lines[0][:5])  # first line only (spurious 2nd boxes)
        gt.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return gt


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


# ── server ──

class Sam:
    def __init__(self, url, model):
        self.c = httpx.Client(base_url=url, timeout=1800)
        self.model = model
        self.session = None

    def req(self, body):
        r = self.c.post(f"/video/{self.model}/request", json=encode(body))
        if r.status_code != 200:
            raise RuntimeError(f"{body['type']} -> {r.status_code}: {r.text[:500]}")
        return r.json()

    def start(self, resource_path):
        self.session = self.req({"type": "start_session", "resource_path": resource_path})["session_id"]

    def close(self):
        if self.session is None:
            return
        sess, self.session = self.session, None
        try:
            self.c.post(f"/video/{self.model}/request", json={"type": "cancel_propagation", "session_id": sess})
        except Exception:
            pass
        r = self.c.post(f"/video/{self.model}/request",
                        json={"type": "close_session", "session_id": sess, "clear_cache_threshold": 0})
        print(f"close_session {sess}: {r.status_code}")

    def propagate(self, start_frame, direction, thresh, max_frames=None, evict=True):
        """Yield (frame_index, raw_outputs_json) — masks stay RLE-encoded until needed."""
        body = {"type": "propagate_in_video", "session_id": self.session,
                "propagation_direction": direction, "start_frame_index": start_frame,
                "evict_cached_frame_outputs": evict}
        if thresh is not None:
            body["output_prob_thresh"] = thresh
        # max_frames is enforced client-side (cancel_propagation): upstream max_frame_num_to_track
        # crashes sam3.1 batched grounding near the end of the range (tensor size mismatch).
        n = 0
        with self.c.stream("POST", f"/video/{self.model}/stream_request", json=body) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"propagate -> {resp.status_code}: {resp.read()[:500]}")
            for line in resp.iter_lines():
                if not line:
                    continue
                item = json.loads(line)
                if "error" in item:
                    raise RuntimeError(f"propagate error: {item}")
                yield item["frame_index"], item["outputs"]
                n += 1
                if max_frames and n >= max_frames:
                    self.c.post(f"/video/{self.model}/request",
                                json={"type": "cancel_propagation", "session_id": self.session})
                    break

    def info(self):
        return self.c.get("/info").json()

    def image_mask_from_box(self, image_path, box):
        sid = self.c.post("/image/set_image", json={"image_path": image_path}).json()["state_id"]
        try:
            r = self.c.post("/image/call/predict_inst",
                            json={"state_id": sid, "kwargs": {"box": [box], "multimask_output": False}})
            if r.status_code != 200:
                raise RuntimeError(f"predict_inst -> {r.status_code}: {r.text[:500]}")
            masks, scores, _ = decode(r.json()["result"])
        finally:
            self.c.delete(f"/image/states/{sid}")
        m = np.asarray(masks).reshape(-1, *np.asarray(masks).shape[-2:])[0]
        return m > 0 if m.dtype != bool else m, float(np.asarray(scores).ravel()[0])


# ── prompting ──

def add_prompt(sam, kind, args, frame_idx, gt_box, w, h, frame_path, obj_id=1):
    """kind: 'semantic' (text and/or exemplar box), 'points', 'mask'. Returns (outputs, details)."""
    x0, y0, x1, y1 = gt_box
    d = {"kind": kind, "frame_index": frame_idx}
    if kind == "semantic":
        body = {"type": "add_prompt", "session_id": sam.session, "frame_index": frame_idx}
        if args.text:
            body["text"] = d["text"] = args.text
        if args.exemplar_gt:
            xywh = [x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h]
            xywh = [min(max(v, 0.0), 1.0) for v in xywh]
            body["bounding_boxes"] = d["bounding_boxes"] = [xywh]
            body["bounding_box_labels"] = d["bounding_box_labels"] = [1]
        if args.output_prob_thresh is not None:
            body["output_prob_thresh"] = args.output_prob_thresh
    elif kind == "points":
        cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        npts = getattr(args, "n_points", 1)
        offs = [(0, 0, 1)] + ([(-.25, 0, 1), (.25, 0, 1), (0, -.25, 1), (0, .25, 1)] if npts >= 5 else [])
        if npts == 7:  # + 2 negatives just below the box (stand legs), so the face is not grown into the stand
            offs += [(-.4, .8, 0), (.4, .8, 0)]
        pt = [[min(max((cx + dx * bw) / w, 0.0), 1.0), min(max((cy + dy * bh) / h, 0.0), 1.0)] for dx, dy, _ in offs]
        lab = [l for _, _, l in offs]
        body = {"type": "add_prompt", "session_id": sam.session, "frame_index": frame_idx,
                "points": pt, "point_labels": lab, "obj_id": obj_id, "rel_coordinates": True}
        d.update(points=pt, point_labels=lab, obj_id=obj_id)
    elif kind == "mask":
        mask, score = sam.image_mask_from_box(frame_path, gt_box)
        body = {"type": "add_mask", "session_id": sam.session, "frame_index": frame_idx,
                "obj_id": obj_id, "mask": mask}
        d.update(obj_id=obj_id, mask_area=int(mask.sum()), mask_score=score,
                 mask_box_iou_vs_gt=box_iou(mask_bbox(mask) or [0, 0, 0, 0], gt_box))
    else:
        raise ValueError(kind)
    t = time.time()
    r = sam.req(body)
    out = r.get("outputs", {})
    if kind == "points" and np.asarray(decode(out.get("out_obj_ids", []))).size == 0:
        # sam3.1 quirk: the first point prompt in a fresh session returns nothing and creates no
        # object (a second identical call does); retry once.
        r = sam.req(body)
        out = r.get("outputs", {})
        d["point_prompt_retried"] = True
    d["latency_sec"] = round(time.time() - t, 3)
    d["n_objects_on_prompt_frame"] = int(np.asarray(decode(out.get("out_obj_ids", []))).size)
    return out, d


# ── per-frame evaluation ──

def eval_frame(idx, outputs, gt, w, h, box_check):
    ids = np.asarray(decode(outputs.get("out_obj_ids", [])), dtype=int).ravel()
    probs = np.asarray(decode(outputs.get("out_probs", [])), dtype=float).ravel()
    boxes = np.asarray(decode(outputs.get("out_boxes_xywh", [])), dtype=float).reshape(-1, 4)
    objs = []
    for k, oid in enumerate(ids):
        bx, by, bw, bh = boxes[k] if k < len(boxes) else (0, 0, 0, 0)
        box = [bx * w, by * h, (bx + bw) * w, (by + bh) * h]  # normalized top-left xywh -> px xyxy
        objs.append({"obj_id": int(oid), "box": [round(v, 1) for v in box],
                     "prob": round(float(probs[k]), 4) if k < len(probs) else None,
                     "iou": round(box_iou(box, gt), 4) if gt else None})
    # verify box convention against the mask bbox on a sample of frames
    if objs and len(box_check) < 40 and idx % 13 == 0:
        masks = decode(outputs["out_binary_masks"])
        for k, o in enumerate(objs):
            mb = mask_bbox(masks[k])
            if mb:
                box_check.append(max(abs(a - b) for a, b in zip(mb, o["box"])))
    prim = max(objs, key=lambda o: o["iou"] or 0.0) if (objs and gt) else None
    piou = prim["iou"] if prim else 0.0
    return {"frame": idx, "gt": [round(v, 1) for v in gt] if gt else None, "n_pred": len(objs),
            "primary_obj_id": prim["obj_id"] if prim and piou > 0 else None,
            "primary_iou": piou, "primary_prob": prim["prob"] if prim else None,
            "hit": int(piou >= HIT_IOU),
            "fp": sum(1 for o in objs if gt and (o["iou"] or 0) < FP_IOU),
            "objects": objs}


def first_loss(rows, start=0, iou_thresh=HIT_IOU):
    run = 0
    for r in rows:
        if r["frame"] < start or r["gt"] is None:
            continue
        run = run + 1 if r["primary_iou"] < iou_thresh else 0
        if run >= LOSS_RUN:
            return r["frame"] - LOSS_RUN + 1
    return None


def summarize(rows):
    ev = [r for r in rows if r["gt"] is not None]
    ious = [r["primary_iou"] for r in ev]
    hits = [r for r in ev if r["hit"]]
    switches = sum(1 for a, b in zip(hits, hits[1:]) if a["primary_obj_id"] != b["primary_obj_id"])
    return {"frames": len(rows), "frames_with_gt": len(ev),
            "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
            "median_iou": round(float(np.median(ious)), 4) if ious else 0.0,
            "hit_rate": round(len(hits) / len(ev), 4) if ev else 0.0,
            "zero_pred_frames": sum(1 for r in rows if r["n_pred"] == 0),
            "fp_total": sum(r["fp"] for r in rows),
            "frames_with_fp": sum(1 for r in rows if r["fp"]),
            "max_objects": max((r["n_pred"] for r in rows), default=0),
            "id_switches": switches,
            "first_loss": first_loss(rows)}


# ── overlays ──

def draw(frame_path, row, masks_enc, out_path, width=640):
    img = cv2.imread(frame_path)
    masks = decode(masks_enc) if masks_enc is not None else None
    for k, o in enumerate(row["objects"]):
        prim = o["obj_id"] == row["primary_obj_id"]
        col = (60, 60, 255) if prim else (0, 220, 255)
        if masks is not None and k < len(masks):
            m = masks[k]
            img[m] = (0.55 * img[m] + 0.45 * np.array(col)).astype(np.uint8)
        x0, y0, x1, y1 = map(int, o["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
        cv2.putText(img, f"id{o['obj_id']} {o['prob']:.2f}", (x0, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    if row["gt"]:
        x0, y0, x1, y1 = map(int, row["gt"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
    txt = f"#{row['frame']} IoU {row['primary_iou']:.2f} n={row['n_pred']}"
    cv2.putText(img, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    s = width / img.shape[1]
    img = cv2.resize(img, (width, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return img


def contact_sheet(imgs, out_path, cols=4, width=320):
    tiles = [cv2.resize(im, (width, int(im.shape[0] * width / im.shape[1]))) for im in imgs]
    th = max(t.shape[0] for t in tiles)
    tiles += [np.zeros((th, width, 3), np.uint8)] * (-len(tiles) % cols)
    tiles = [np.pad(t, ((0, th - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 80])


# ── main ──

def run_pass(sam, start, args, gt, w, h, rows, masks, box_check, direction=None):
    n, t = 0, time.time()
    for idx, out in sam.propagate(start, direction or args.direction, args.output_prob_thresh, args.max_frames,
                                  evict=not args.keep_cache):
        rows[idx] = eval_frame(idx, out, gt[idx], w, h, box_check)
        masks[idx] = out.get("out_binary_masks")
        n += 1
        if n % 50 == 0:
            print(f"  frame {idx}: iou={rows[idx]['primary_iou']:.2f} n={rows[idx]['n_pred']} "
                  f"({n / (time.time() - t):.1f} FPS)")
    return n, time.time() - t


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8770")
    ap.add_argument("--model", default="sam3.1", choices=["sam3.1", "sam3"])
    ap.add_argument("--frames", required=True, help="dir of <int>.jpg (session resource_path)")
    ap.add_argument("--labels", default=os.path.join(root, "label_Data/torpedo_image_label/labels"))
    ap.add_argument("--text")
    ap.add_argument("--exemplar-gt", action="store_true")
    ap.add_argument("--points-gt", action="store_true")
    ap.add_argument("--mask-gt", action="store_true")
    ap.add_argument("--n-points", type=int, default=1, choices=[1, 5, 7],
                    help="--points-gt: 1 = GT box centre; 5 = centre + 4 points at +-25%% of box w/h; "
                    "7 = those 5 + 2 negatives below the box (legs)")
    ap.add_argument("--prompt-frame", type=int, default=0)
    ap.add_argument("--direction", default="forward", choices=["forward", "backward", "both"])
    ap.add_argument("--output-prob-thresh", type=float, help="add_prompt + propagate (server default 0.5)")
    ap.add_argument("--correct-at", help="'auto' (first loss) or a frame index")
    ap.add_argument("--correct-with", default="same", choices=["same", "semantic", "points", "mask"])
    ap.add_argument("--max-corrections", type=int, default=1, help="only for --correct-at auto")
    ap.add_argument("--loss-iou", type=float, default=HIT_IOU,
                    help="--correct-at auto: correct at the first run of %d frames with IoU below this" % LOSS_RUN)
    ap.add_argument("--max-frames", type=int, help="stop each pass after N frames (quick tests)")
    ap.add_argument("--prime-text", help="sam3.1 points: first run a full detector pass with this (null-concept) "
                    "text so every frame gets a cached_frame_outputs entry; upstream _build_sam2_output returns {} "
                    "for uncached frames, which drops point-only objects after the prompt frame. Implies --keep-cache")
    ap.add_argument("--keep-cache", action="store_true",
                    help="propagate with evict_cached_frame_outputs=False (needed for sam3.1 point corrections)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.points_gt or args.mask_gt:
        if args.text or args.exemplar_gt or (args.points_gt and args.mask_gt):
            sys.exit("--points-gt / --mask-gt cannot be combined with other prompts")
        kind = "points" if args.points_gt else "mask"
    elif args.text or args.exemplar_gt:
        kind = "semantic"
    else:
        sys.exit("give --text and/or --exemplar-gt, or --points-gt, or --mask-gt")
    if kind == "mask" and args.model != "sam3":
        sys.exit("--mask-gt needs --model sam3 (sam3.1 has no add_mask)")
    corr_kind = kind if args.correct_with == "same" else args.correct_with
    if args.prime_text:
        args.keep_cache = True

    frames = list_frames(args.frames)
    h, w = cv2.imread(frames[0]).shape[:2]
    gt = load_gt(args.labels, len(frames), w, h)
    out_dir = os.path.join(args.out, args.name)
    os.makedirs(out_dir, exist_ok=True)
    pf = args.prompt_frame
    if gt[pf] is None:
        sys.exit(f"no GT on prompt frame {pf}")

    sam = Sam(args.url, args.model)
    info0 = sam.info()
    if args.model not in info0["models"]:
        sys.exit(f"model {args.model} not loaded on server: {info0['models']}")
    rows, masks, box_check = [None] * len(frames), [None] * len(frames), []
    t_wall = time.time()
    corrections = []
    try:
        sam.start(args.frames)
        print(f"session {sam.session} ({len(frames)} frames {w}x{h})")
        prime = None
        if args.prime_text:
            t = time.time()
            r = sam.req({"type": "add_prompt", "session_id": sam.session, "frame_index": 0, "text": args.prime_text})
            n_prime_objs = set()
            for _, out in sam.propagate(0, "forward", None, None, evict=False):
                n_prime_objs.update(np.asarray(decode(out.get("out_obj_ids", [])), dtype=int).ravel().tolist())
            prime = {"text": args.prime_text, "sec": round(time.time() - t, 1), "obj_ids_seen": sorted(n_prime_objs)}
            print("prime:", prime)
        _, prompt = add_prompt(sam, kind, args, pf, gt[pf], w, h, frames[pf])
        print("prompt:", prompt)
        n, dt = run_pass(sam, pf, args, gt, w, h, rows, masks, box_check)
        n_total, dt_total = n, dt
        print(f"pass 1: {n} frames in {dt:.1f} s")

        if args.correct_at:
            before = summarize([r for r in rows if r])
            start = 0
            for _ in range(args.max_corrections if args.correct_at == "auto" else 1):
                k = first_loss([r for r in rows if r], start, args.loss_iou) if args.correct_at == "auto" else int(args.correct_at)
                if k is None or gt[k] is None:
                    break
                obj_id = rows[k]["primary_obj_id"] if rows[k] and rows[k]["primary_obj_id"] is not None else 1
                if corr_kind == "semantic":  # resets the session state; frames < k keep pass-1 results
                    _, cp = add_prompt(sam, "semantic", argparse.Namespace(**{**vars(args), "exemplar_gt": True}),
                                       k, gt[k], w, h, frames[k])
                else:
                    _, cp = add_prompt(sam, corr_kind, args, k, gt[k], w, h, frames[k], obj_id=obj_id)
                seg_before = summarize([r for r in rows if r and r["frame"] >= k])
                n, dt = run_pass(sam, k, args, gt, w, h, rows, masks, box_check, direction="forward")
                n_total, dt_total = n_total + n, dt_total + dt
                cp.update(segment_before={x: seg_before[x] for x in ("mean_iou", "hit_rate", "first_loss")},
                          segment_after={x: summarize([r for r in rows if r and r["frame"] >= k])[x]
                                         for x in ("mean_iou", "hit_rate", "first_loss")},
                          frames_repropagated=n)
                corrections.append(cp)
                print(f"correction at {k}: {cp}")
                start = k + 1
    finally:
        info1 = sam.info()
        sam.close()
    wall = time.time() - t_wall

    rows_ok = [r for r in rows if r]
    summ = summarize(rows_ok)
    summ.update(name=args.name, model=args.model, prompt=prompt, prompt_frame=pf,
                direction=args.direction, output_prob_thresh=args.output_prob_thresh,
                fps_http=round(n_total / dt_total, 2) if dt_total else None,
                propagate_sec=round(dt_total, 1), wall_sec=round(wall, 1),
                torch_peak_vram_gib=round(info1["gpu"]["max_allocated_bytes"] / 2**30, 2),
                torch_peak_vram_note="torch.cuda.max_memory_allocated since server start (not reset per run)",
                torch_peak_before_run_gib=round(info0["gpu"]["max_allocated_bytes"] / 2**30, 2),
                box_vs_mask_bbox_max_px=round(max(box_check), 2) if box_check else None,
                frames_missing=[i for i, r in enumerate(rows) if r is None][:20],
                prime=prime, keep_cache=args.keep_cache)
    if args.correct_at:
        summ.update(corrections=corrections, n_corrections=len(corrections), before_correction=before)

    with open(os.path.join(out_dir, "frames.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "gt_x0", "gt_y0", "gt_x1", "gt_y1", "n_pred", "primary_obj_id", "primary_iou",
                     "primary_prob", "hit", "fp", "objects"])
        for r in rows_ok:
            wr.writerow([r["frame"], *(r["gt"] or [""] * 4), r["n_pred"], r["primary_obj_id"], r["primary_iou"],
                         r["primary_prob"], r["hit"], r["fp"], json.dumps(r["objects"])])
    json.dump([r["primary_iou"] if r else None for r in rows], open(os.path.join(out_dir, "ious.json"), "w"))
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)

    # overlays
    fl = summ["first_loss"]
    worst = sorted((r for r in rows_ok if r["gt"]), key=lambda r: r["primary_iou"])[:5]
    pick = {pf: "prompt"}
    for i in range(0, len(frames), 25):
        pick.setdefault(i, "every25")
    if fl is not None:
        for j in range(fl, min(fl + 4, len(frames))):
            pick[j] = "firstloss" if j == fl else f"firstloss+{j - fl}"
    for r in worst:
        pick.setdefault(r["frame"], "worst")
    for c in corrections:
        pick[c["frame_index"]] = "correction"
    imgs = {}
    for i, tag in sorted(pick.items()):
        if rows[i] is None:
            continue
        imgs[i] = draw(frames[i], rows[i], masks[i], os.path.join(out_dir, f"{i:05d}_{tag}.jpg"))
    sheet = [pf] + ([fl, fl + 1] if fl is not None else []) + [r["frame"] for r in worst[:2]]
    regular = [i for i in sorted(imgs) if pick[i] == "every25" and i not in sheet]
    need = 12 - len(sheet)
    sheet += [regular[int(k * len(regular) / need)] for k in range(need)] if len(regular) > need else regular
    sheet = sorted(dict.fromkeys(i for i in sheet if i in imgs))
    contact_sheet([imgs[i] for i in sheet], os.path.join(out_dir, "contact.jpg"))

    print(json.dumps({k: v for k, v in summ.items() if k not in ("prompt", "corrections", "before_correction")}))
    print("->", out_dir)


if __name__ == "__main__":
    main()
