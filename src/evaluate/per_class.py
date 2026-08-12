"""Per-class detection metrics on the validation split (assignment §10.1).

§10.1 asks for per-class performance *together with* the inference settings the numbers
were produced under, because a per-class table only means something next to the
confidence / NMS configuration that generated the boxes. Both are written here: the
settings sit at the top level of the JSON and in the header of the markdown table.

For each model and each of the 80 COCO classes this reports:

  - **AP50-95 and AP50** from the official evaluator, obtained by running COCOeval
    restricted to that single category (cat_ids=[id]). COCO's AP is already a
    macro-average over categories, so a one-category run is that class's AP.
  - **precision / recall / F1 and the raw TP / FP / FN counts** at the operating
    threshold chosen on validation in §10.4, from the explicit greedy matcher in
    coco_eval.py restricted to that class. COCOeval never reports these — AP is
    threshold-free — but they are what a deployed model's behaviour actually looks like.
  - **train and val instance counts.** D13 recorded a severe imbalance (person alone is
    31% of all annotations) and deliberately deferred it to evaluation rather than
    resampling. Without the counts, an AP of 0.0 on a class with 3 val objects reads
    like the same finding as an AP of 0.0 on a class with 300; it is not.
  - **the class-frequency band** (common / mid / rare) used by the §10.5 slice, so this
    table and that slice can be read against each other.

The threshold comes from reports/metrics/thresholds.json (§10.4, max F1 on validation)
via eval_slices.load_operating_threshold. If that file has not been written yet the
script falls back to conf 0.25 and says so loudly — in the console, in the JSON, and in
the markdown header. A precision/recall table whose threshold is unknown is not
interpretable, so the fallback is never allowed to be quiet.

VALIDATION ONLY in normal use. The frozen test split stays untouched until the
threshold decision is final (assignment section 10 preamble).

Usage:
    python -m src.evaluate.per_class \
        --model 640=artifacts/.../detect-300ep-640/weights/best.pt \
        --model 960=artifacts/.../detect-300ep-960/weights/best.pt
"""

import argparse
import json
from pathlib import Path

from src.data.common import DEFAULT_CONFIG, load_config
from src.evaluate.coco_eval import (
    PRED_CONF,
    PRED_IOU,
    PRED_MAX_DET,
    build_gt_json,
    category_maps,
    class_bands,
    load_gt_annotations,
    match_detections,
    predict_cached,
    run_cocoeval,
)
from src.evaluate.eval_slices import IMGSZ_FROM_LABEL, MATCH_IOU, threshold_for_label


def count_val_instances(gt_annotations):
    """Non-crowd and crowd validation instance counts per original category id.

    Non-crowd is the number the recall column is measured against, so the two are kept
    apart: crowd regions are ignore regions (D6), never objects the model had to find.
    """
    non_crowd = {}
    crowd = {}
    for ann in gt_annotations:
        bucket = crowd if int(ann.get("iscrowd", 0)) == 1 else non_crowd
        bucket[ann["category_id"]] = bucket.get(ann["category_id"], 0) + 1
    return non_crowd, crowd


def per_class_rows(gt_path, gt_annotations, detections, conf, bands_info,
                   original_to_name, n_images):
    """One metrics row per COCO class, for a single model. Returns a list of dicts.

    Two independent measurements per class, on purpose:
      COCOeval with cat_ids=[id]      -> AP50-95 / AP50, threshold-free ranking quality
      match_detections on that class  -> P / R / F1 / TP / FP / FN at the operating point
    A class can look fine on one and bad on the other (good ranking, badly calibrated
    scores), and §10.1 wants both.
    """
    band_of = {}
    for band, names in bands_info["bands"].items():
        for name in names:
            band_of[name] = band
    train_counts = bands_info["train_counts"]
    val_counts, val_crowd_counts = count_val_instances(gt_annotations)

    rows = []
    for category_id in sorted(original_to_name):
        name = original_to_name[category_id]

        class_gt = [a for a in gt_annotations if a["category_id"] == category_id]
        class_dets = [d for d in detections if d["category_id"] == category_id]

        # run_cocoeval returns None when the model produced nothing for this class;
        # with ground truth present that is an AP of exactly zero, not a missing value.
        ap = run_cocoeval(gt_path, class_dets, cat_ids=[category_id])
        counts = match_detections(class_gt, class_dets, conf=conf, iou_thr=MATCH_IOU,
                                  n_images=n_images)

        rows.append({
            "category_id": category_id,
            "name": name,
            "band": band_of[name],
            "train_instances": train_counts[name],
            "val_instances": val_counts.get(category_id, 0),
            "val_crowd_instances": val_crowd_counts.get(category_id, 0),
            "ap50_95": ap["mAP50-95"] if ap else 0.0,
            "ap50": ap["mAP50"] if ap else 0.0,
            "precision": counts["precision"],
            "recall": counts["recall"],
            "f1": counts["f1"],
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "duplicates": counts["duplicates"],
            "ignored_on_crowd": counts["ignored"],
            "detections_above_threshold": counts["n_detections"],
            "detections_total": len(class_dets),
        })
    return rows


def summarise(rows):
    """Headline numbers over the 80 per-class rows.

    The macro means are the unweighted average over classes, which is exactly what COCO
    reports as overall mAP — printing it here is a cheap self-check that the 80
    single-category runs agree with the whole-dataset run.
    """
    n = len(rows)
    return {
        "n_classes": n,
        "macro_mean_ap50_95": sum(r["ap50_95"] for r in rows) / n,
        "macro_mean_ap50": sum(r["ap50"] for r in rows) / n,
        "classes_with_zero_ap50_95": sum(1 for r in rows if r["ap50_95"] == 0.0),
        "classes_with_zero_ap50": sum(1 for r in rows if r["ap50"] == 0.0),
        "classes_with_zero_recall": sum(1 for r in rows if r["recall"] == 0.0),
    }


def write_markdown(path, report):
    """Human-readable per-class table, sorted by val instance count descending.

    Sorted by val count rather than by AP because the first question a reader asks of a
    per-class table on an imbalanced dataset is "how much of this is the tail?" — the
    ordering puts the classes carrying the evaluation weight at the top and the
    statistically fragile ones at the bottom.
    """
    settings = report["inference_settings"]
    threshold = report["operating_threshold"]
    summary = report["summary"]

    lines = []
    lines.append(f"# Per-class metrics — model {report['label']} — {report['split']} split")
    lines.append("")
    lines.append("Assignment §10.1. Generated by `src/evaluate/per_class.py`.")
    lines.append("")
    lines.append(f"- checkpoint: `{report['checkpoint']}`")
    lines.append(f"- ground truth: {report['ground_truth']['images']} images, "
                 f"{report['ground_truth']['annotations']} annotations "
                 f"({report['ground_truth']['non_crowd_annotations']} non-crowd + "
                 f"{report['ground_truth']['crowd_annotations']} iscrowd kept as ignore regions)")
    lines.append(f"- inference: imgsz {settings['imgsz']}, device {settings['device']}, "
                 f"conf {settings['conf_predict']} (mAP protocol), NMS IoU {settings['nms_iou']}, "
                 f"max_det {settings['max_det']}, ultralytics {settings['ultralytics_version']}")
    lines.append("- AP columns: pycocotools COCOeval restricted to one category at a time")
    lines.append(f"- P/R columns: greedy matcher at IoU {settings['match_iou']}, "
                 f"confidence {threshold['conf']:g}")
    if threshold["is_fallback"]:
        lines.append("")
        lines.append(f"> **WARNING — FALLBACK THRESHOLD.** {threshold['source']}")
    else:
        lines.append(f"- threshold source: {threshold['source']}")
    lines.append("")
    lines.append(f"Macro mean AP50-95 {summary['macro_mean_ap50_95']:.4f} · "
                 f"macro mean AP50 {summary['macro_mean_ap50']:.4f} · "
                 f"{summary['classes_with_zero_ap50_95']} of {summary['n_classes']} classes "
                 f"score AP50-95 = 0.")
    lines.append("")
    lines.append("| class | band | train | val | AP50-95 | AP50 | P | R | F1 | TP | FP | FN |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    ordered = sorted(report["classes"], key=lambda r: (-r["val_instances"], r["name"]))
    for r in ordered:
        lines.append(
            f"| {r['name']} | {r['band']} | {r['train_instances']} | {r['val_instances']} | "
            f"{r['ap50_95']:.4f} | {r['ap50']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['f1']:.4f} | {r['tp']} | {r['fp']} | {r['fn']} |"
        )
    lines.append("")

    Path(path).write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Per-class COCO metrics (§10.1).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                        help="val by default; test is frozen until thresholds are final")
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=PATH",
                        help="repeatable, e.g. --model 640=path/to/best.pt")
    parser.add_argument("--refresh", action="store_true",
                        help="re-run inference instead of reading the prediction cache")
    parser.add_argument("--conf", type=float, default=None,
                        help="override the operating threshold; by default it is read "
                             "from reports/metrics/thresholds.json (§10.4)")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics_dir = Path(cfg["paths"]["metrics_dir"])

    gt_path, n_img, n_ann = build_gt_json(cfg, args.split, refresh=args.refresh)
    gt_annotations = load_gt_annotations(gt_path)
    n_crowd = sum(1 for a in gt_annotations if int(a.get("iscrowd", 0)) == 1)
    print(f"ground truth: {n_img} images, {n_ann} annotations "
          f"({n_ann - n_crowd} non-crowd + {n_crowd} iscrowd) -> {gt_path}")

    # Bands and train counts are a property of the dataset, not of the model, so they
    # are computed once and reused for every model.
    bands_info = class_bands(cfg)
    _, original_to_name = category_maps(cfg)

    for spec in args.model:
        label, path = spec.split("=", 1)
        imgsz = IMGSZ_FROM_LABEL.get(label, 640)
        print(f"\n[{label}] predicting at imgsz={imgsz} ...")
        detections, meta = predict_cached(cfg, path, label, args.split, imgsz,
                                          refresh=args.refresh, device=args.device)

        conf, source, is_fallback = threshold_for_label(cfg, label, args.conf)
        if is_fallback:
            print(f"!! [{label}] {source}")
        else:
            print(f"[{label}] operating threshold conf={conf:g} ({source})")

        print(f"[{label}] scoring 80 classes one category at a time ...")
        rows = per_class_rows(gt_path, gt_annotations, detections, conf, bands_info,
                              original_to_name, n_images=n_img)

        report = {
            "assignment_section": "10.1",
            "split": args.split,
            "label": label,
            "checkpoint": meta["checkpoint"],
            "checkpoint_sha256": meta["checkpoint_sha256"],
            # §10.1 asks for the inference settings alongside the per-class numbers.
            "inference_settings": {
                "imgsz": meta["imgsz"],
                "device": meta["device"],
                "conf_predict": PRED_CONF,
                "conf_operating_point": conf,
                "nms_iou": PRED_IOU,
                "max_det": PRED_MAX_DET,
                "match_iou": MATCH_IOU,
                "ultralytics_version": meta["ultralytics_version"],
                "evaluator": "pycocotools COCOeval (bbox) + greedy matcher, coco_eval.py",
            },
            "operating_threshold": {"conf": conf, "source": source, "is_fallback": is_fallback},
            "ground_truth": {
                "images": n_img,
                "annotations": n_ann,
                "non_crowd_annotations": n_ann - n_crowd,
                "crowd_annotations": n_crowd,
            },
            "band_definition": bands_info["thresholds"],
            "summary": summarise(rows),
            "classes": rows,
        }

        json_path = metrics_dir / f"per_class_{label}_{args.split}.json"
        md_path = metrics_dir / f"per_class_{label}_{args.split}.md"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        write_markdown(md_path, report)

        s = report["summary"]
        print(f"[{label}] macro mean AP50-95 {s['macro_mean_ap50_95']:.4f}, "
              f"AP50 {s['macro_mean_ap50']:.4f}, "
              f"{s['classes_with_zero_ap50_95']}/{s['n_classes']} classes at AP50-95 = 0")
        best = sorted(rows, key=lambda r: -r["ap50_95"])[:5]
        print(f"[{label}] best: " + ", ".join(f"{r['name']} {r['ap50_95']:.3f}" for r in best))
        print(f"[{label}] written: {json_path}")
        print(f"[{label}] written: {md_path}")


if __name__ == "__main__":
    main()
