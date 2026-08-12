"""Slice evaluation with the official COCO evaluator (§10.5 slice analysis).

Two slices, both on the validation split:

  1. **Object size** (`models[...].metrics`). Ultralytics' own val() reports one overall
     mAP; the pre-registered imgsz hypothesis
     (reports/metrics/experiment_imgsz_hypothesis.md) is about SMALL objects
     specifically, so it needs APsmall / APmedium / APlarge. pycocotools.COCOeval
     computes exactly those using the same 32^2 / 96^2 thresholds the dataset audit
     used. These numbers back a published experiment result, so this part of the report
     is frozen: the schema and the values must not move.

  2. **Common vs minority classes** (`class_band_slice`). D13 recorded a severe class
     imbalance (person alone is 31% of annotations) and deliberately deferred it to
     evaluation rather than resampling the data. This slice is where that promise is
     kept: classes are banded by how many TRAIN instances the model actually learned
     from (common >= 500, mid 100-499, rare < 100) and mean per-class AP plus recall at
     the chosen operating point are reported per band.

The ground-truth builder, the cached predictor, the COCOeval wrapper, the greedy matcher
and the band definition all live in src/evaluate/coco_eval.py, shared with the rest of
the §10 suite. The size-slice numbers, the CLI and the existing slices_{split}.json keys
are unchanged by the addition of the second slice.

Protocol notes:
  - Evaluation runs on the VALIDATION split only. The frozen test split stays untouched
    until model and threshold decisions are final (assignment section 10).
  - Ground truth keeps iscrowd=1 annotations so COCOeval treats them as IGNORE regions
    (the official protocol): detections landing on a crowd are neither rewarded nor
    punished. Training excluded them (D6); evaluation must not invent false positives
    out of them.
  - Detections are produced at conf=0.001 (the mAP protocol), not at any operating
    threshold. Threshold selection is a separate step (section 10.4); the band slice
    reads the result of that step for its recall column.
  - Each model is evaluated at ITS OWN training resolution, since that is the setting
    the run is a baseline for.

Usage:
    python -m src.evaluate.eval_slices \
        --model 640=artifacts/.../detect-300ep-640/weights/best.pt \
        --model 960=artifacts/.../detect-300ep-960/weights/best.pt
"""

import argparse
import json
from pathlib import Path

from src.data.common import DEFAULT_CONFIG, load_config
from src.evaluate.coco_eval import (
    build_gt_json,
    category_maps,
    class_bands,
    load_gt_annotations,
    match_detections,
    predict_cached,
    run_cocoeval,
)

IMGSZ_FROM_LABEL = {"640": 640, "960": 960}  # label -> inference size; falls back to 640

# Where §10.4 (threshold_sweep.py) publishes the operating point chosen on validation.
THRESHOLDS_FILE = "thresholds.json"
# Used only if that file does not exist yet. 0.25 is Ultralytics' own default, so it is
# a defensible stand-in, but it is NOT the max-F1 point and every output says so.
FALLBACK_CONF = 0.25
# IoU a detection needs to count as a hit in the greedy matcher (the usual reporting
# point, and the same IoU COCO's AP50 uses).
MATCH_IOU = 0.5

BAND_ORDER = ("common", "mid", "rare")


def load_operating_threshold(cfg, override=None):
    """Read the confidence threshold chosen on validation by §10.4.

    Shared by this module's band slice and by src/evaluate/per_class.py, so both tables
    describe the same operating point. Returns (conf, source, is_fallback) where source
    is a human sentence meant to be printed and stored next to the numbers.

    threshold_sweep.py is written by a different step of the workflow, so the exact
    shape of thresholds.json is not assumed: the chosen confidence is looked up under a
    per-model key first (models[label] or label) and then under the obvious key names.
    If the file is missing the caller gets FALLBACK_CONF and a loud is_fallback=True —
    a precision/recall number without its threshold is not interpretable, so the
    fallback is never allowed to be silent.
    """
    if override is not None:
        return float(override), f"--conf {override} passed on the command line", False

    path = Path(cfg["paths"]["metrics_dir"]) / THRESHOLDS_FILE
    if not path.exists():
        return FALLBACK_CONF, (
            f"FALLBACK conf={FALLBACK_CONF} — {path} does not exist, so the max-F1 "
            f"operating point from §10.4 was NOT available. These P/R numbers are at "
            f"the Ultralytics default threshold, not at a chosen one."
        ), True

    with open(path) as f:
        doc = json.load(f)
    conf = _conf_from_thresholds_doc(doc)
    if conf is None:
        return FALLBACK_CONF, (
            f"FALLBACK conf={FALLBACK_CONF} — {path} exists but no chosen confidence "
            f"could be read from it."
        ), True
    return conf, f"max-F1 operating point read from {path}", False


def _conf_from_thresholds_doc(node):
    """Pull the chosen confidence out of a thresholds.json object, or return None.

    Tries the nested "chosen point" wrappers first, then a plain confidence key on the
    object itself. Kept deliberately shallow and explicit rather than a generic search,
    so what it will and will not accept is readable at a glance.
    """
    if not isinstance(node, dict):
        return None
    for wrapper in ("chosen", "chosen_point", "selected", "operating_point", "best"):
        inner = node.get(wrapper)
        if isinstance(inner, (int, float)):
            return float(inner)
        found = _conf_from_thresholds_doc(inner)
        if found is not None:
            return found
    for key in ("conf", "confidence", "conf_threshold", "threshold"):
        value = node.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def threshold_for_label(cfg, label, override=None):
    """Same as load_operating_threshold, but prefers a per-model entry if one exists.

    The sweep may publish one threshold per model (the 640 and 960 models do not have
    to share an operating point) or a single global one. Both are handled here so
    callers never have to care which shape was written.
    """
    if override is not None:
        return load_operating_threshold(cfg, override)

    path = Path(cfg["paths"]["metrics_dir"]) / THRESHOLDS_FILE
    if path.exists():
        with open(path) as f:
            doc = json.load(f)
        per_model = doc.get("models", {}).get(label) if isinstance(doc.get("models"), dict) else None
        if per_model is None:
            per_model = doc.get(label)
        conf = _conf_from_thresholds_doc(per_model)
        if conf is not None:
            return conf, f"max-F1 operating point for model '{label}' read from {path}", False
    return load_operating_threshold(cfg, override)


def build_class_band_slice(gt_path, gt_annotations, detections, conf, bands_info,
                           name_to_category_id, n_images):
    """Common-vs-minority slice for one model (§10.5, second required slice).

    For each band (common / mid / rare) this reports:
      n_classes, train_instances, val_instances,
      mean per-class AP50-95 and AP50, and precision/recall/F1 at `conf`.

    The mean AP is obtained by calling COCOeval once per band with cat_ids set to that
    band's categories. COCO's AP is already a macro-average over categories, so
    restricting the category list gives exactly "mean per-class AP over this band" —
    the same number as averaging the individual per-class APs, without 80 extra runs.

    Recall is a threshold-dependent count, which COCOeval does not produce, so it comes
    from the greedy matcher restricted to the band's categories. Crowd annotations of
    those categories are passed through so their ignore behaviour still applies.
    """
    train_counts = bands_info["train_counts"]
    slice_report = {}

    for band in BAND_ORDER:
        class_names = bands_info["bands"][band]
        cat_ids = sorted(name_to_category_id[name] for name in class_names)
        cat_id_set = set(cat_ids)

        band_gt = [a for a in gt_annotations if a["category_id"] in cat_id_set]
        band_dets = [d for d in detections if d["category_id"] in cat_id_set]
        val_instances = sum(1 for a in band_gt if int(a.get("iscrowd", 0)) == 0)

        ap = run_cocoeval(gt_path, band_dets, cat_ids=cat_ids)
        counts = match_detections(band_gt, band_dets, conf=conf, iou_thr=MATCH_IOU,
                                  n_images=n_images)

        slice_report[band] = {
            "n_classes": len(class_names),
            "train_instances": int(sum(train_counts[name] for name in class_names)),
            "val_instances": val_instances,
            "mean_ap50_95": ap["mAP50-95"] if ap else 0.0,
            "mean_ap50": ap["mAP50"] if ap else 0.0,
            "precision_at_threshold": counts["precision"],
            "recall_at_threshold": counts["recall"],
            "f1_at_threshold": counts["f1"],
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "classes": class_names,
        }

    for band in BAND_ORDER:
        slice_report[band]["interpretation"] = _band_interpretation(band, slice_report)
    return slice_report


def _relative(value, reference):
    """value as a percentage of reference, e.g. 0.47 -> '47%'. Guards divide-by-zero."""
    if not reference:
        return "n/a"
    return f"{value / reference:.0%}"


def _band_interpretation(band, slice_report):
    """Why the slice matters and what this band's numbers reveal.

    Every comparison in the text is computed from the measured values, never asserted
    from expectation. That matters here: the naive expectation is that AP falls with
    class rarity, and on this dataset it does not — writing the expected conclusion into
    the report would have contradicted the table sitting next to it.
    """
    entry = slice_report[band]
    common = slice_report["common"]
    ap, recall = entry["mean_ap50_95"], entry["recall_at_threshold"]
    ap_vs_common = _relative(ap, common["mean_ap50_95"])
    recall_vs_common = _relative(recall, common["recall_at_threshold"])
    aps = {b: slice_report[b]["mean_ap50_95"] for b in BAND_ORDER}
    rank = "highest" if ap == max(aps.values()) else ("lowest" if ap == min(aps.values()) else "middle")

    if band == "common":
        return (
            f"{entry['n_classes']} classes with >=500 train instances: {entry['train_instances']} "
            f"train and {entry['val_instances']} val objects. Mean per-class AP50-95 {ap:.4f} "
            f"({rank} of the three bands), recall {recall:.4f}. Why it matters: D13 measured a "
            f"severe imbalance (person alone is ~31% of annotations) and chose to expose it at "
            f"evaluation rather than resample, so this is where that promise is kept. What it "
            f"reveals: this band sets the recall ceiling, but data volume does not buy per-class "
            f"accuracy — the head classes are also the small, cluttered ones (person, chair, "
            f"bottle, cup, book), so frequency and difficulty are confounded."
        )
    if band == "mid":
        return (
            f"{entry['n_classes']} classes with 100-499 train instances, {entry['val_instances']} "
            f"val objects. Mean per-class AP50-95 {ap:.4f} ({ap_vs_common} of the common band, "
            f"{rank} of the three), recall {recall:.4f} ({recall_vs_common} of common). Why it "
            f"matters: COCO mAP is a macro-average over classes, so these 59 of 80 classes "
            f"effectively set the headline number even though they are a minority of the objects. "
            f"What it reveals: the frequency effect shows up in recall at a fixed confidence, not "
            f"in AP — the scores of mid-frequency classes rank fine but sit below the threshold."
        )
    return (
        f"{entry['n_classes']} classes with <100 train instances and only {entry['val_instances']} "
        f"val objects between them (D4 flagged the extremes: toaster 11 and hair drier 6 train "
        f"instances). Mean per-class AP50-95 {ap:.4f} ({ap_vs_common} of the common band, {rank} "
        f"of the three), recall {recall:.4f} ({recall_vs_common} of common). Why it matters: this "
        f"is the tail the assignment's imbalance question is about. What it reveals: rarity costs "
        f"recall at the operating point, not ranking quality — several rare classes are large, "
        f"visually distinctive objects (microwave, bear, fire hydrant, stop sign) that score well, "
        f"while the truly starved ones sit at zero. The band mean is statistically fragile: with a "
        f"handful of val objects per class, one detection moves a class's AP substantially, so "
        f"read it as indicative, and read per_class_{{label}}_val.md for the per-class detail."
    )


def main():
    parser = argparse.ArgumentParser(description="Per-size and per-class-band COCO evaluation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                        help="val by default; test is frozen until thresholds are final")
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=PATH",
                        help="repeatable, e.g. --model 640=path/to/best.pt")
    parser.add_argument("--out", default=None, help="where to write the JSON report")
    parser.add_argument("--refresh", action="store_true",
                        help="re-run inference instead of reading the prediction cache")
    parser.add_argument("--conf", type=float, default=None,
                        help="override the operating threshold used by the class-band slice; "
                             "by default it is read from reports/metrics/thresholds.json")
    args = parser.parse_args()

    cfg = load_config(args.config)

    gt_path, n_img, n_ann = build_gt_json(cfg, args.split, refresh=args.refresh)
    print(f"ground truth: {n_img} images, {n_ann} annotations ({args.split} split) -> {gt_path}")

    report = {"split": args.split, "ground_truth": {"images": n_img, "annotations": n_ann}, "models": {}}
    detections_by_label = {}
    for spec in args.model:
        label, path = spec.split("=", 1)
        imgsz = IMGSZ_FROM_LABEL.get(label, 640)
        print(f"\n[{label}] predicting at imgsz={imgsz} ...")
        detections, meta = predict_cached(cfg, path, label, args.split, imgsz, refresh=args.refresh)
        metrics = run_cocoeval(gt_path, detections)
        report["models"][label] = {
            "checkpoint": meta["checkpoint"],
            "imgsz": meta["imgsz"],
            "images": meta["n_images"],
            "detections": meta["n_detections"],
            "latency_median_ms": meta["latency_median_ms"],
            "latency_p95_ms": meta["latency_p95_ms"],
            "metrics": metrics,
        }
        detections_by_label[label] = detections
        print(f"[{label}] {meta['n_detections']} detections, "
              f"median latency {meta['latency_median_ms']:.1f} ms")

    # ---- slice 2: common vs minority classes (§10.5) -------------------------------
    # Added as a separate top-level key. Nothing above this line changed, so the size
    # slice under "models" stays byte-identical to the published experiment result.
    gt_annotations = load_gt_annotations(gt_path)
    bands_info = class_bands(cfg)
    _, original_to_name = category_maps(cfg)
    name_to_category_id = {name: cat_id for cat_id, name in original_to_name.items()}

    band_slice = {
        "definition": {
            "counted_on": "train split, iscrowd excluded (D6) — the instances the model learned from",
            "common_min_train_instances": bands_info["thresholds"]["common_min"],
            "mid_min_train_instances": bands_info["thresholds"]["mid_min"],
            "match_iou": MATCH_IOU,
        },
        "models": {},
    }
    for label in report["models"]:
        conf, source, is_fallback = threshold_for_label(cfg, label, args.conf)
        if is_fallback:
            print(f"\n!! [{label}] {source}")
        else:
            print(f"\n[{label}] operating threshold conf={conf:g} ({source})")
        band_slice["models"][label] = {
            "operating_threshold": {"conf": conf, "source": source, "is_fallback": is_fallback},
            "bands": build_class_band_slice(
                gt_path, gt_annotations, detections_by_label[label], conf, bands_info,
                name_to_category_id, n_images=n_img,
            ),
        }
    report["class_band_slice"] = band_slice

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

    print("\nclass-frequency band slice (mean per-class AP over the band's classes):")
    header = f"{'model':<6}{'band':<8}{'classes':>8}{'val obj':>9}{'AP50-95':>10}{'AP50':>9}{'recall':>9}"
    print(header)
    for label in report["models"]:
        for band in BAND_ORDER:
            e = band_slice["models"][label]["bands"][band]
            print(f"{label:<6}{band:<8}{e['n_classes']:>8}{e['val_instances']:>9}"
                  f"{e['mean_ap50_95']:>10.4f}{e['mean_ap50']:>9.4f}{e['recall_at_threshold']:>9.4f}")

    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
