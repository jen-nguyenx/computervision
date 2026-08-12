"""Shared evaluation core for the COCO5K detection models (assignment §10).

Every evaluation script in src/evaluate imports from here, so there is exactly one
implementation of: which images are in a split, how ground truth is built, how
predictions are produced and cached, how COCOeval is called, and how a detection is
matched to a ground-truth box.

Design rule — predict once, analyse many times. Inference costs 2-4 minutes per model
per split on MPS and every downstream analysis (threshold sweep, per-class table, size
slices, error gallery) needs the SAME detections. So predictions are computed once at
conf=0.001 and cached to reports/metrics/preds_{label}_{split}.json. That is a speed
win, but more importantly it is a correctness property: all analyses provably describe
one identical set of boxes.

Protocol notes that apply to everything in this module:
  - Ground truth KEEPS iscrowd=1 annotations so COCOeval treats them as IGNORE regions
    (the official COCO protocol). Training excluded them (D6); evaluation must not
    invent false positives out of them.
  - Detections are produced at conf=0.001, iou=0.7, max_det=300 — the mAP protocol,
    not an operating point. Threshold selection is a separate step (§10.4).
  - Category ids: model outputs are contiguous 0..79, COCO ground truth uses the
    original gapped ids 1..90. The explicit map from configs/coco_category_map.json is
    always used; never `id - 1` (D2).

See docs/COCO_PIPELINE_DECISIONS.md for the D-numbers cited above.
"""

import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.data.common import load_instances, read_manifest, read_splits, sha256_file

# The mAP protocol. These are constants, not tunables: changing them changes what mAP
# means, and every cached prediction file records them so a stale cache is detectable.
PRED_CONF = 0.001
PRED_IOU = 0.7
PRED_MAX_DET = 300

# Class-frequency bands for the common-vs-minority slice (§10.5), counted on train.
BAND_COMMON_MIN = 500
BAND_MID_MIN = 100


def rows_for_split(cfg, split):
    """Manifest rows belonging to one split, as a DataFrame.

    The frozen manifest + splits files are the single source of truth for which
    images are in which partition (D1, D4), so every script filters the same way.
    Columns are the manifest columns: image_id, file_name, width, height, source_split.
    """
    manifest = read_manifest(cfg)
    splits = read_splits(cfg)
    split_ids = set(splits.loc[splits["split"] == split, "image_id"])
    return manifest[manifest["image_id"].isin(split_ids)]


def category_maps(cfg):
    """Return (contiguous_to_original, original_to_name) from the frozen category map.

    Policy D2: original COCO ids run 1-90 with 10 gaps, so the mapping to contiguous
    0..79 model ids is an explicit sorted table stored in configs/coco_category_map.json.
      contiguous_to_original: {0: 1, 1: 2, ..., 11: 13, ...}  (model id  -> COCO id)
      original_to_name:       {1: "person", ..., 13: "stop sign", ...}  (COCO id -> name)
    """
    with open(cfg["paths"]["category_map"]) as f:
        cat_map = json.load(f)
    contiguous_to_original = {int(v): int(k) for k, v in cat_map["map"].items()}
    original_to_name = {
        contiguous_to_original[int(contiguous)]: name
        for contiguous, name in cat_map["names"].items()
    }
    return contiguous_to_original, original_to_name


def build_gt_json(cfg, split, out_path=None, refresh=False):
    """Write COCO-format ground truth for one split; return (path, n_images, n_annotations).

    Built straight from the source annotations, restricted to the frozen split image
    ids (D1, D4), so the evaluated image set is exactly the partition on disk.

    Crowd annotations are KEPT with iscrowd=1. That is the whole point: COCOeval reads
    the flag and treats those boxes as ignore regions instead of as missed objects.

    Reading the source instances json costs ~7 seconds, so an existing file whose image
    count already matches the split size is reused unless refresh=True.
    """
    if out_path is None:
        out_path = cfg["paths"]["metrics_dir"] / f"gt_{split}.json"
    out_path = Path(out_path)

    rows = rows_for_split(cfg, split)
    split_ids = set(rows["image_id"])

    if out_path.exists() and not refresh:
        with open(out_path) as f:
            existing = json.load(f)
        if len(existing["images"]) == len(rows):
            return out_path, len(existing["images"]), len(existing["annotations"])
        print(f"ground truth {out_path.name} has {len(existing['images'])} images but the "
              f"{split} split has {len(rows)}; rebuilding")

    instances = load_instances(cfg)
    images = [
        {"id": int(r.image_id), "file_name": r.file_name, "width": int(r.width), "height": int(r.height)}
        for r in rows.itertuples()
    ]
    annotations = [
        {
            "id": int(a["id"]),
            "image_id": int(a["image_id"]),
            "category_id": int(a["category_id"]),
            "bbox": [float(v) for v in a["bbox"]],
            "area": float(a["area"]),
            "iscrowd": int(a.get("iscrowd", 0)),
        }
        for a in instances["annotations"]
        if a["image_id"] in split_ids
    ]
    categories = [{"id": int(c["id"]), "name": c["name"]} for c in instances["categories"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
    return out_path, len(images), len(annotations)


def load_gt_annotations(gt_path):
    """Read back the annotation list written by build_gt_json (crowd rows included)."""
    with open(gt_path) as f:
        return json.load(f)["annotations"]


def _cache_path(cfg, label, split):
    return cfg["paths"]["metrics_dir"] / f"preds_{label}_{split}.json"


def predict_cached(cfg, model_path, label, split, imgsz, refresh=False, device="mps"):
    """Return (detections, meta) for one model on one split, using the prediction cache.

    Detections are COCO-format dicts {image_id, category_id, bbox (xywh), score} at the
    mAP protocol settings (conf=0.001, iou=0.7, max_det=300), cached to
    reports/metrics/preds_{label}_{split}.json.

    A cache is only reused if its recorded checkpoint_sha256, split, imgsz and conf all
    match what is being asked for. That check matters more than the speed: a cache
    silently answering for a different checkpoint would attribute one model's numbers to
    another, and nothing downstream could detect it. Any mismatch re-predicts.
    """
    cache = _cache_path(cfg, label, split)
    checkpoint_sha256 = sha256_file(model_path)

    if cache.exists() and not refresh:
        with open(cache) as f:
            cached = json.load(f)
        meta = cached["meta"]
        mismatches = []
        if meta.get("checkpoint_sha256") != checkpoint_sha256:
            mismatches.append("checkpoint_sha256")
        if meta.get("split") != split:
            mismatches.append("split")
        if meta.get("imgsz") != imgsz:
            mismatches.append("imgsz")
        if meta.get("conf") != PRED_CONF:
            mismatches.append("conf")
        if not mismatches:
            print(f"[{label}] using cached predictions: {cache.name} "
                  f"({meta['n_detections']} detections over {meta['n_images']} images)")
            return cached["detections"], meta
        print(f"[{label}] cache {cache.name} is stale ({', '.join(mismatches)}); re-predicting")

    detections, meta = _predict(cfg, model_path, label, split, imgsz, device, checkpoint_sha256)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump({"meta": meta, "detections": detections}, f)
    print(f"[{label}] wrote {cache}")
    return detections, meta


def _predict(cfg, model_path, label, split, imgsz, device, checkpoint_sha256):
    """Run the model over every image of the split. The slow path behind predict_cached."""
    import ultralytics                 # imported here so --help works without torch loaded
    from ultralytics import YOLO

    contiguous_to_original, _ = category_maps(cfg)
    rows = rows_for_split(cfg, split)
    images_dir = cfg["paths"]["images_dir"]

    model = YOLO(model_path)
    detections = []
    latencies = []

    for r in rows.itertuples():
        started = time.perf_counter()
        result = model.predict(
            images_dir / r.file_name, imgsz=imgsz, conf=PRED_CONF, iou=PRED_IOU,
            max_det=PRED_MAX_DET, device=device, verbose=False,
        )[0]
        latencies.append((time.perf_counter() - started) * 1000)

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        for (x1, y1, x2, y2), cls, score in zip(xyxy, boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
            detections.append({
                "image_id": int(r.image_id),
                "category_id": contiguous_to_original[int(cls)],
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],  # COCO wants xywh
                "score": float(score),
            })

    latencies = np.array(latencies)
    meta = {
        "checkpoint": str(model_path),
        "checkpoint_sha256": checkpoint_sha256,
        "label": label,
        "split": split,
        "imgsz": imgsz,
        "conf": PRED_CONF,
        "iou": PRED_IOU,
        "max_det": PRED_MAX_DET,
        "device": device,
        "ultralytics_version": ultralytics.__version__,
        "n_images": len(rows),
        "n_detections": len(detections),
        # Latency is recorded so a cache hit can still report it. It is NOT a
        # steady-state benchmark: no warm-up, so the first call carries model-load and
        # graph-compile cost. §10.7 (benchmark.py) is the number to quote.
        "latency_median_ms": float(np.median(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
    }
    return detections, meta


def run_cocoeval(gt_path, detections, cat_ids=None):
    """Run pycocotools COCOeval and return the 12 standard statistics.

    Returns None when there are no detections (COCOeval cannot score an empty result).
    Passing cat_ids restricts evaluation to those categories; a single-id list is how
    per-class AP is obtained (§10.1). cat_ids are ORIGINAL COCO ids, not contiguous.
    """
    with contextlib.redirect_stdout(io.StringIO()):   # pycocotools is very chatty
        gt = COCO(str(gt_path))
        if not detections:
            return None
        dt = gt.loadRes(detections)
        ev = COCOeval(gt, dt, iouType="bbox")
        if cat_ids is not None:
            ev.params.catIds = list(cat_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    return {
        "mAP50-95": float(s[0]), "mAP50": float(s[1]), "mAP75": float(s[2]),
        "AP_small": float(s[3]), "AP_medium": float(s[4]), "AP_large": float(s[5]),
        "AR_1": float(s[6]), "AR_10": float(s[7]), "AR_100": float(s[8]),
        "AR_small": float(s[9]), "AR_medium": float(s[10]), "AR_large": float(s[11]),
    }


def iou_xywh(box_a, box_b):
    """Intersection-over-union of two COCO boxes [x, y, w, h]. Pure function."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def intersection_over_detection(det_box, crowd_box):
    """Overlap of a detection with a crowd region, as a fraction of the DETECTION area.

    The COCO convention for iscrowd regions: a crowd box covers many objects, so a
    detection sitting inside it is correct-ish and must not be punished. Scoring that
    overlap with plain IoU would fail — a small box inside a big crowd region has tiny
    IoU — so the denominator is the detection area alone.
    """
    dx1, dy1, dw, dh = det_box
    cx1, cy1, cw, ch = crowd_box
    inter_w = min(dx1 + dw, cx1 + cw) - max(dx1, cx1)
    inter_h = min(dy1 + dh, cy1 + ch) - max(dy1, cy1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    det_area = dw * dh
    if det_area <= 0:
        return 0.0
    return (inter_w * inter_h) / det_area


def match_detections(gt_annotations, detections, conf, iou_thr=0.5, n_images=None):
    """Greedy detection-to-ground-truth matching at one operating point (§10.4, §10.6).

    COCOeval gives AP, which is threshold-free by construction. It does not expose the
    counts a practitioner has to report at a chosen confidence: how many true positives,
    false positives, misses and duplicate boxes. This is that matcher, written out
    explicitly so every number in the error analysis can be traced to a rule.

    Algorithm, one detection at a time in descending score order (greedy = highest
    scoring detection gets first claim on a ground-truth box):
      1. keep only detections with score >= conf;
      2. find the highest-IoU UNMATCHED ground-truth box of the SAME class in the SAME
         image with IoU >= iou_thr -> true positive, that box is now used;
      3. else, if a same-class ground-truth box in that image is already matched at
         IoU >= iou_thr -> duplicate. Duplicates are counted separately AND as false
         positives, because a spurious extra box is exactly that;
      4. else, if the detection overlaps an iscrowd=1 region of the same class by
         >= iou_thr of its own area -> IGNORED: neither true nor false positive. This
         is the rule that matters most. Crowd regions mark "many objects here, not
         individually labelled"; counting detections there as errors would penalise the
         model for being right;
      5. otherwise -> false positive.
    Non-crowd ground truth left unmatched is a false negative. Crowd ground truth never
    counts as a false negative — it was never an individual object to find.

    Note on crowds: COCOeval intersects against the crowd RLE mask; here only boxes are
    available, so the crowd's bounding box is used. That is the more generous shape, so
    this matcher ignores slightly more than COCOeval would — conservative in the right
    direction (it never invents a false positive on a crowd).

    Note on rule order: rules 3 and 4 can both apply to one detection (a double box on a
    labelled object that also happens to sit inside a crowd region). Duplicate wins,
    because a crowd bounding box frequently swallows individually-labelled objects and
    would otherwise excuse genuine double-detections. Ignoring is reserved for
    detections with no other explanation. Measured effect on val at conf 0.25: 10 of 876
    false positives; every false positive touching a crowd is one of these duplicates.

    Args:
        gt_annotations: COCO annotation dicts with id, image_id, category_id, bbox, iscrowd.
        detections: COCO detection dicts with image_id, category_id, bbox, score.
        conf: confidence threshold; detections scoring below it are discarded.
        iou_thr: IoU required for a match (0.5 is the usual reporting point).
        n_images: images in the split, for detections_per_image. Defaults to the number
            of distinct image ids seen in ground truth and detections.

    Returns a dict of counts, the derived P/R/F1, and the per-detection bookkeeping
    (matches, fp_indices, fn_gt_ids) that the error gallery needs to draw examples.
    Indices in matches/fp_indices refer to positions in the INPUT detections list.
    """
    # Ground truth split by role, bucketed per image so matching stays local.
    gt_by_image = {}
    crowd_by_image = {}
    n_gt_real = 0
    for ann in gt_annotations:
        bucket = crowd_by_image if int(ann.get("iscrowd", 0)) == 1 else gt_by_image
        bucket.setdefault(ann["image_id"], []).append(ann)
        if int(ann.get("iscrowd", 0)) != 1:
            n_gt_real += 1

    kept = [(i, det) for i, det in enumerate(detections) if det["score"] >= conf]
    # Descending score. Python's sort is stable, so equal scores keep input order and
    # the result is deterministic across runs.
    kept.sort(key=lambda pair: -pair[1]["score"])

    matched_gt_ids = set()
    matches = []
    fp_indices = []
    duplicates = 0
    ignored = 0

    for det_idx, det in kept:
        candidates = gt_by_image.get(det["image_id"], [])
        same_class = [g for g in candidates if g["category_id"] == det["category_id"]]

        best_gt = None
        best_iou = iou_thr
        for gt in same_class:
            if gt["id"] in matched_gt_ids:
                continue
            overlap = iou_xywh(det["bbox"], gt["bbox"])
            if overlap >= best_iou:
                best_iou = overlap
                best_gt = gt

        if best_gt is not None:
            matched_gt_ids.add(best_gt["id"])
            matches.append((det_idx, best_gt["id"], best_iou))
            continue

        hits_used_gt = any(
            gt["id"] in matched_gt_ids and iou_xywh(det["bbox"], gt["bbox"]) >= iou_thr
            for gt in same_class
        )
        if hits_used_gt:
            duplicates += 1
            fp_indices.append(det_idx)
            continue

        crowds = crowd_by_image.get(det["image_id"], [])
        on_crowd = any(
            crowd["category_id"] == det["category_id"]
            and intersection_over_detection(det["bbox"], crowd["bbox"]) >= iou_thr
            for crowd in crowds
        )
        if on_crowd:
            ignored += 1
            continue

        fp_indices.append(det_idx)

    fn_gt_ids = sorted(
        ann["id"] for anns in gt_by_image.values() for ann in anns
        if ann["id"] not in matched_gt_ids
    )

    tp = len(matches)
    fp = len(fp_indices)
    fn = len(fn_gt_ids)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_gt_real if n_gt_real else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    if n_images is None:
        n_images = len(set(gt_by_image) | set(crowd_by_image) | {d["image_id"] for d in detections})

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "duplicates": duplicates,
        "ignored": ignored,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_detections": len(kept),
        "detections_per_image": len(kept) / n_images if n_images else 0.0,
        "matches": matches,
        "fp_indices": fp_indices,
        "fn_gt_ids": fn_gt_ids,
    }


def class_bands(cfg, verbose=True):
    """Group the 80 classes into common / mid / rare by TRAIN instance count (§10.5).

    Bands are counted on the TRAIN split only. Using val counts would define the bands
    with the same data used to measure the effect, which is circular: the question is
    "does a class the model rarely saw score worse", and "rarely saw" is a property of
    training data.

    Crowd annotations are excluded from the counts because training excluded them (D6),
    so these are the instances the model actually learned from. Counts come from the
    source annotations restricted to the frozen train image ids — never from a cached
    summary file, which could describe an older split.

    Thresholds: common >= 500, mid 100-499, rare < 100 instances.
    """
    _, original_to_name = category_maps(cfg)
    train_ids = set(rows_for_split(cfg, "train")["image_id"])

    counts = {name: 0 for name in original_to_name.values()}
    instances = load_instances(cfg)
    for ann in instances["annotations"]:
        if ann["image_id"] not in train_ids:
            continue
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        counts[original_to_name[int(ann["category_id"])]] += 1

    bands = {"common": [], "mid": [], "rare": []}
    for name, n in counts.items():
        if n >= BAND_COMMON_MIN:
            bands["common"].append(name)
        elif n >= BAND_MID_MIN:
            bands["mid"].append(name)
        else:
            bands["rare"].append(name)
    # Sort each band by descending train count so the tables read most-frequent first.
    for band in bands:
        bands[band].sort(key=lambda name: (-counts[name], name))

    if verbose:
        print(f"class bands by train instance count "
              f"(common >= {BAND_COMMON_MIN}, mid {BAND_MID_MIN}-{BAND_COMMON_MIN - 1}, "
              f"rare < {BAND_MID_MIN}):")
        for band in ("common", "mid", "rare"):
            print(f"  {band:<7} {len(bands[band]):>3} classes")

    return {
        "thresholds": {"common_min": BAND_COMMON_MIN, "mid_min": BAND_MID_MIN, "counted_on": "train"},
        "bands": bands,
        "train_counts": counts,
    }
