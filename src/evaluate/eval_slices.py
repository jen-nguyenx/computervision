"""Per-object-size evaluation with the official COCO evaluator (§10.5 slice analysis).

Ultralytics' own val() reports one overall mAP; the pre-registered imgsz hypothesis
(reports/metrics/experiment_imgsz_hypothesis.md) is about SMALL objects specifically,
so it needs APsmall / APmedium / APlarge. pycocotools.COCOeval computes exactly those
using the same 32^2 / 96^2 thresholds the dataset audit used.

Protocol notes:
  - Evaluation runs on the VALIDATION split only. The frozen test split stays untouched
    until model and threshold decisions are final (assignment section 10).
  - Ground truth keeps iscrowd=1 annotations so COCOeval treats them as IGNORE regions
    (the official protocol): detections landing on a crowd are neither rewarded nor
    punished. Training excluded them (D6); evaluation must not invent false positives
    out of them.
  - Detections are produced at conf=0.001 (the mAP protocol), not at any operating
    threshold. Threshold selection is a separate step (section 10.4).
  - Each model is evaluated at ITS OWN training resolution, since that is the setting
    the run is a baseline for.

Usage:
    python -m src.evaluate.eval_slices \
        --model 640=artifacts/.../detect-300ep-640/weights/best.pt \
        --model 960=artifacts/.../detect-300ep-960/weights/best.pt
"""

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.data.common import DEFAULT_CONFIG, load_config, load_instances, read_manifest, read_splits

IMGSZ_FROM_LABEL = {"640": 640, "960": 960}  # label -> inference size; falls back to 640


def build_gt_json(cfg, split, out_path):
    """COCO-format ground truth for one split, straight from the source annotations.

    Uses the frozen manifest + splits so the evaluated image set is exactly the
    partition on disk (D1, D4). Crowd annotations are kept, flagged iscrowd=1.
    """
    manifest = read_manifest(cfg)
    splits = read_splits(cfg)
    split_ids = set(splits.loc[splits["split"] == split, "image_id"])
    rows = manifest[manifest["image_id"].isin(split_ids)]

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
    return len(images), len(annotations)


def predict_to_coco(model_path, cfg, split, imgsz, contiguous_to_original):
    """Run the model over the split and return COCO-format detections + latency stats."""
    from ultralytics import YOLO  # imported here so --help works without torch loaded

    manifest = read_manifest(cfg)
    splits = read_splits(cfg)
    split_ids = set(splits.loc[splits["split"] == split, "image_id"])
    rows = manifest[manifest["image_id"].isin(split_ids)]

    model = YOLO(model_path)
    detections = []
    latencies = []
    images_dir = cfg["paths"]["images_dir"]

    for r in rows.itertuples():
        started = time.perf_counter()
        # conf=0.001 is the mAP protocol: keep almost everything, let the PR curve decide.
        result = model.predict(
            images_dir / r.file_name, imgsz=imgsz, conf=0.001, iou=0.7,
            max_det=300, device="mps", verbose=False,
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
    stats = {
        "images": len(rows),
        "detections": len(detections),
        "latency_median_ms": float(np.median(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
    }
    return detections, stats


def cocoeval(gt_path, detections):
    """Run COCOeval and pull out the 12 standard numbers."""
    with contextlib.redirect_stdout(io.StringIO()):   # pycocotools is very chatty
        gt = COCO(str(gt_path))
        if not detections:
            return None
        dt = gt.loadRes(detections)
        ev = COCOeval(gt, dt, iouType="bbox")
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


def main():
    parser = argparse.ArgumentParser(description="Per-size COCO evaluation of one or more models.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                        help="val by default; test is frozen until thresholds are final")
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=PATH",
                        help="repeatable, e.g. --model 640=path/to/best.pt")
    parser.add_argument("--out", default=None, help="where to write the JSON report")
    args = parser.parse_args()

    cfg = load_config(args.config)
    with open(cfg["paths"]["category_map"]) as f:
        cat_map = json.load(f)
    # labels are contiguous 0..79; COCO ground truth uses the original gapped ids (D2)
    contiguous_to_original = {int(v): int(k) for k, v in cat_map["map"].items()}

    gt_path = cfg["paths"]["metrics_dir"] / f"gt_{args.split}.json"
    n_img, n_ann = build_gt_json(cfg, args.split, gt_path)
    print(f"ground truth: {n_img} images, {n_ann} annotations ({args.split} split) -> {gt_path}")

    report = {"split": args.split, "ground_truth": {"images": n_img, "annotations": n_ann}, "models": {}}
    for spec in args.model:
        label, path = spec.split("=", 1)
        imgsz = IMGSZ_FROM_LABEL.get(label, 640)
        print(f"\n[{label}] predicting at imgsz={imgsz} ...")
        detections, stats = predict_to_coco(path, cfg, args.split, imgsz, contiguous_to_original)
        metrics = cocoeval(gt_path, detections)
        report["models"][label] = {"checkpoint": path, "imgsz": imgsz, **stats, "metrics": metrics}
        print(f"[{label}] {stats['detections']} detections, "
              f"median latency {stats['latency_median_ms']:.1f} ms")

    out = Path(args.out) if args.out else cfg["paths"]["metrics_dir"] / f"slices_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    labels = list(report["models"])
    print(f"\n{'metric':<12}" + "".join(f"{l:>12}" for l in labels) +
          ("       change" if len(labels) == 2 else ""))
    for key in ["mAP50-95", "mAP50", "AP_small", "AP_medium", "AP_large",
                "AR_small", "AR_medium", "AR_large", "latency_median_ms"]:
        values = [report["models"][l]["metrics"][key] if key in report["models"][l]["metrics"]
                  else report["models"][l][key] for l in labels]
        line = f"{key:<12}" + "".join(f"{v:>12.4f}" for v in values)
        if len(labels) == 2 and values[0] > 0:
            line += f"   {100 * (values[1] - values[0]) / values[0]:+7.1f}%"
        print(line)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
