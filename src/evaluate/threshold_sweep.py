"""Confidence-threshold selection on the validation split (assignment §10.4).

§10.4 says: "Do not accept a default threshold without analysis. Use validation data
to select or justify the final operating threshold for each model." and "Do not select
thresholds using the held-out test partition." This script is that analysis.

What it does: sweeps the confidence threshold from 0.05 to 0.90 in 0.01 steps over the
CACHED detections (produced once at conf=0.001 by coco_eval.predict_cached), matches
detections to ground truth at IoU 0.5 with coco_eval.match_detections at every grid
point, and picks the threshold that maximises F1. Nothing here re-runs inference and
nothing here reads the test split.

Why max F1: no business cost asymmetry between a false positive and a false negative
has been specified for this assignment, so the two errors are treated as equally
expensive and F1 — their equal-weight harmonic mean — is the criterion. That assumption
is written into reports/metrics/thresholds.json (key "assumption"), not just into this
comment, because the choice is only defensible while the assumption holds. The full
sweep is written out alongside the chosen point so a later cost statement can be
re-applied (F-beta, or a precision/recall floor) without re-running anything.

Protocol notes:
  - VALIDATION ONLY. The split is a module constant, not a CLI flag: there is no way to
    point this script at test, because a threshold chosen on test would invalidate the
    final test numbers (§10 preamble, §10.4).
  - Matching is the explicit greedy matcher in coco_eval.match_detections at IoU 0.5:
    iscrowd=1 regions are ignore regions (never false positives, never misses), and a
    second box on an already-matched object counts as a duplicate AND as a false
    positive. See that docstring for the full rule order.
  - Precision/recall here come from that matcher, NOT from Ultralytics' val(). The two
    differ in matcher, IoU handling and threshold, so the numbers are the same order of
    magnitude rather than identical — that comparison is the sanity check, not a target.

Usage:
    python -m src.evaluate.threshold_sweep \
        --model 640=artifacts/.../detect-300ep-640/weights/best.pt \
        --model 960=artifacts/.../detect-300ep-960/weights/best.pt
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: we only save files, never open windows
import matplotlib.pyplot as plt

from src.data.common import DEFAULT_CONFIG, load_config
from src.evaluate.coco_eval import (
    build_gt_json,
    load_gt_annotations,
    match_detections,
    predict_cached,
)

# Thresholds are chosen on validation. Not a CLI option on purpose (§10.4).
SPLIT = "val"

# The sweep grid, from the assignment brief: 0.05 to 0.90 inclusive, 0.01 steps.
CONF_MIN = 0.05
CONF_MAX = 0.90
CONF_STEP = 0.01

# IoU required to call a detection a match. 0.5 is the usual reporting point for
# precision/recall (mAP50-95 is COCOeval's job, and is threshold-free).
MATCH_IOU = 0.5

# The two reference points the trade-off table is quoted at, either side of the choice:
# a permissive threshold and a conservative one.
LOW_CONF = 0.10
HIGH_CONF = 0.50

# How close to the best F1 still counts as "as good as the best", for the plateau report.
# 1% relative: on 500 val images the F1 differences inside that band are far smaller than
# the sampling noise, so the exact argmax is not a meaningful distinction.
PLATEAU_TOLERANCE = 0.01

SELECTION_CRITERION = "max F1 on the validation split"
SELECTION_ASSUMPTION = (
    "No business cost asymmetry between false positives and false negatives has been "
    "specified for this task, so a missed object and a spurious box are treated as "
    "equally expensive; F1, their equal-weight harmonic mean, is therefore the "
    "selection criterion. If a deployment later states a cost ratio, re-select from "
    "the sweep in this file with F-beta (beta > 1 when misses cost more, beta < 1 when "
    "false alarms cost more) or with a precision/recall floor -- no re-running of "
    "inference is needed."
)

# Colour-blind-safe categorical slots (blue / orange / aqua), validated for adjacent-pair
# separation under deuteranopia and tritanopia. Fixed order, never cycled.
SERIES_COLORS = {"precision": "#2a78d6", "recall": "#eb6834", "f1": "#1baf7a"}
GRID_COLOR = "#CCCCCC"
INK_COLOR = "#52514e"


def imgsz_for_label(label, override=None):
    """Inference size for a model label: its own training resolution.

    Each run is evaluated at the resolution it was trained at, and our two detection
    runs are named after that resolution ("640", "960"), so a numeric label IS the
    imgsz. --imgsz overrides for any future model whose label is not a number.
    """
    if override is not None:
        return override
    return int(label) if label.isdigit() else 640


def conf_grid():
    """The confidence values to evaluate: 0.05, 0.06, ... 0.90 (86 points).

    Built by integer arithmetic and rounded to 2 dp so the grid is exactly the printed
    values; 0.05 + 0.01 * i in plain floats drifts (0.35000000000000003) and would make
    the chosen threshold ugly to quote and awkward to reuse.
    """
    n_steps = round((CONF_MAX - CONF_MIN) / CONF_STEP) + 1
    return [round(CONF_MIN + CONF_STEP * i, 2) for i in range(n_steps)]


def sweep_model(gt_annotations, detections, n_images):
    """Evaluate every grid point for one model; return the rows of the sweep.

    One row per confidence value, with the counts §10.4 asks about (false positives,
    false negatives, duplicates, downstream output volume) and the rates derived from
    them. The heavy per-detection bookkeeping match_detections also returns (which
    detection matched which box) is dropped here — the error gallery re-runs the matcher
    at the single chosen threshold when it needs that.
    """
    rows = []
    for conf in conf_grid():
        result = match_detections(
            gt_annotations, detections, conf=conf, iou_thr=MATCH_IOU, n_images=n_images
        )
        rows.append({
            "conf": conf,
            "tp": result["tp"],
            "fp": result["fp"],
            "fn": result["fn"],
            "duplicates": result["duplicates"],
            "ignored": result["ignored"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "n_detections": result["n_detections"],
            "detections_per_image": result["detections_per_image"],
        })
    return rows


def choose_operating_point(rows):
    """Return the sweep row with the highest F1 (the selection rule, §10.4).

    Ties are broken toward the LOWER confidence, i.e. toward recall: at equal F1 the
    lower threshold finds more objects, and a missed object cannot be recovered
    downstream while a false positive can still be filtered. Exact ties are not expected
    on an 0.01 grid; the rule is stated so the choice is deterministic either way.
    """
    return max(rows, key=lambda row: (row["f1"], -row["conf"]))


def f1_plateau(rows, chosen):
    """The band of thresholds whose F1 is within PLATEAU_TOLERANCE of the best.

    Reported because the honest reading of a flat F1 curve is "any threshold in this
    band performs the same", not "the argmax is the one true threshold". A reader who
    knows how wide the band is can pick inside it on other grounds — fewer boxes
    downstream, a precision floor — without giving up measurable F1.
    """
    cutoff = chosen["f1"] * (1 - PLATEAU_TOLERANCE)
    inside = [row for row in rows if row["f1"] >= cutoff]
    return {
        "tolerance_relative": PLATEAU_TOLERANCE,
        "min_f1_in_band": cutoff,
        "conf_min": inside[0]["conf"],
        "conf_max": inside[-1]["conf"],
        "n_grid_points": len(inside),
        # True when every grid point between conf_min and conf_max is inside the band,
        # i.e. the band really is one plateau and not two peaks with a dip between them.
        "contiguous": len(inside) == round((inside[-1]["conf"] - inside[0]["conf"]) / CONF_STEP) + 1,
        "precision_at_conf_min": inside[0]["precision"],
        "precision_at_conf_max": inside[-1]["precision"],
        "recall_at_conf_min": inside[0]["recall"],
        "recall_at_conf_max": inside[-1]["recall"],
        "detections_per_image_at_conf_min": inside[0]["detections_per_image"],
        "detections_per_image_at_conf_max": inside[-1]["detections_per_image"],
    }


def row_at(rows, conf):
    """The sweep row for one confidence value (grid values only, e.g. 0.10, 0.50)."""
    for row in rows:
        if row["conf"] == conf:
            return row
    raise ValueError(f"conf {conf} is not on the sweep grid")


def tradeoff_notes(rows, chosen):
    """Measured values for all six trade-offs §10.4 lists, at three thresholds.

    §10.4 asks for a discussion of false positives, false negatives, precision, recall,
    duplicate predictions and downstream output volume. Prose about those is cheap;
    what follows is the measured value of each at a permissive threshold (0.10), at the
    chosen threshold, and at a conservative one (0.50), so the trade-off is a table of
    real numbers from this model on this split rather than an opinion.
    """
    low = row_at(rows, LOW_CONF)
    high = row_at(rows, HIGH_CONF)
    points = [("at_conf_0.10", low), ("at_chosen_conf", chosen), ("at_conf_0.50", high)]

    def series(key):
        return {name: row[key] for name, row in points}

    return {
        "measured_at": {
            "conf_low": LOW_CONF,
            "conf_chosen": chosen["conf"],
            "conf_high": HIGH_CONF,
        },
        "false_positives": series("fp"),
        "false_negatives": series("fn"),
        "precision": series("precision"),
        "recall": series("recall"),
        "duplicate_predictions": series("duplicates"),
        "downstream_output_volume": {
            "detections_total": series("n_detections"),
            "detections_per_image": series("detections_per_image"),
        },
        "deltas_vs_chosen": {
            "conf_0.10_minus_chosen": {
                "false_positives": low["fp"] - chosen["fp"],
                "false_negatives": low["fn"] - chosen["fn"],
                "true_positives": low["tp"] - chosen["tp"],
                "duplicates": low["duplicates"] - chosen["duplicates"],
                "detections_per_image": low["detections_per_image"] - chosen["detections_per_image"],
                "precision": low["precision"] - chosen["precision"],
                "recall": low["recall"] - chosen["recall"],
                "f1": low["f1"] - chosen["f1"],
            },
            "conf_0.50_minus_chosen": {
                "false_positives": high["fp"] - chosen["fp"],
                "false_negatives": high["fn"] - chosen["fn"],
                "true_positives": high["tp"] - chosen["tp"],
                "duplicates": high["duplicates"] - chosen["duplicates"],
                "detections_per_image": high["detections_per_image"] - chosen["detections_per_image"],
                "precision": high["precision"] - chosen["precision"],
                "recall": high["recall"] - chosen["recall"],
                "f1": high["f1"] - chosen["f1"],
            },
        },
        "fp_cost_per_extra_tp_going_from_chosen_to_conf_0.10": (
            (low["fp"] - chosen["fp"]) / (low["tp"] - chosen["tp"])
            if low["tp"] != chosen["tp"] else None
        ),
        "tp_lost_per_fp_removed_going_from_chosen_to_conf_0.50": (
            (chosen["tp"] - high["tp"]) / (chosen["fp"] - high["fp"])
            if chosen["fp"] != high["fp"] else None
        ),
    }


def spread_labels(entries, min_gap):
    """Nudge end-of-line label positions apart so they stay readable when curves meet.

    entries is a list of (y, text, colour); returns the same list with y values pushed
    up in ascending order until neighbours are at least min_gap apart. Only the label
    moves, never the line.
    """
    ordered = sorted(entries, key=lambda item: item[0])
    spread = []
    previous_y = None
    for y, text, colour in ordered:
        if previous_y is not None and y - previous_y < min_gap:
            y = previous_y + min_gap
        spread.append((y, text, colour))
        previous_y = y
    return spread


def plot_curves(rows, chosen, plateau, label, out_path):
    """Precision, recall and F1 against confidence, with the chosen point marked.

    One y axis (all three series are rates in 0-1, so they share it honestly), thin
    marks, hairline y grid, no top/right spines. The three curves are labelled twice —
    legend and direct end-of-line labels — so identity never depends on colour alone.
    The chosen threshold is a dashed vertical rule with a ringed marker where it meets
    the F1 curve; the pale band behind it is the plateau within 1% of the best F1.
    """
    conf = [row["conf"] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Plateau first, so it sits behind everything and reads as background, not data.
    ax.axvspan(plateau["conf_min"], plateau["conf_max"], color=GRID_COLOR, alpha=0.45,
               linewidth=0, zorder=0)

    for key, name in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]:
        ax.plot(conf, [row[key] for row in rows], color=SERIES_COLORS[key],
                linewidth=2.4 if key == "f1" else 1.8, label=name, zorder=3)

    # The chosen operating point: vertical rule plus a ringed marker on the F1 curve.
    ax.axvline(chosen["conf"], color=INK_COLOR, linewidth=1.0, linestyle="--", zorder=2)
    ax.plot([chosen["conf"]], [chosen["f1"]], marker="o", markersize=8,
            color=SERIES_COLORS["f1"], markeredgecolor="white", markeredgewidth=2, zorder=4)

    # The label sits on the empty side of the rule, low down where no curve runs, so it
    # needs no leader line crossing the precision curve.
    on_the_right = chosen["conf"] < (CONF_MIN + CONF_MAX) / 2
    ax.text(
        chosen["conf"] + (0.008 if on_the_right else -0.008), 0.05,
        f"chosen conf {chosen['conf']:.2f}\n"
        f"F1 {chosen['f1']:.3f}   P {chosen['precision']:.3f}   R {chosen['recall']:.3f}\n"
        f"shaded: F1 within {PLATEAU_TOLERANCE:.0%} of best "
        f"(conf {plateau['conf_min']:.2f}-{plateau['conf_max']:.2f})",
        transform=ax.get_xaxis_transform(),   # x in data units, y in axes fraction
        ha="left" if on_the_right else "right", va="bottom",
        fontsize=9, color=INK_COLOR, zorder=5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=GRID_COLOR, linewidth=0.8),
    )

    # Direct labels at the right-hand end, spread apart where the curves converge.
    end_labels = [(rows[-1][key], name, SERIES_COLORS[key])
                  for key, name in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]]
    for y, text, colour in spread_labels(end_labels, min_gap=0.05):
        ax.text(CONF_MAX + 0.008, y, text, color=colour, fontsize=9,
                va="center", ha="left", clip_on=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(CONF_MIN, CONF_MAX)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([round(0.1 * i, 1) for i in range(1, 10)])
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel(f"Precision / recall / F1 (IoU {MATCH_IOU})")
    ax.set_title(f"Confidence-threshold sweep - {label} model, {SPLIT} split\n"
                 f"operating point selected by max F1", loc="left", fontsize=12)
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def print_summary(label, rows, chosen, plateau):
    """Human-readable echo of the chosen point and the three reference thresholds."""
    print(f"\n[{label}] chosen conf {chosen['conf']:.2f}  "
          f"P {chosen['precision']:.4f}  R {chosen['recall']:.4f}  F1 {chosen['f1']:.4f}")
    print(f"[{label}] F1 within {PLATEAU_TOLERANCE:.0%} of best over conf "
          f"{plateau['conf_min']:.2f}-{plateau['conf_max']:.2f} "
          f"({plateau['n_grid_points']} grid points, contiguous={plateau['contiguous']})")
    header = (f"{'conf':>6}{'tp':>7}{'fp':>8}{'fn':>7}{'dup':>6}{'ign':>6}"
              f"{'P':>9}{'R':>9}{'F1':>9}{'det/img':>10}")
    print(header)
    for tag, row in [("low", row_at(rows, LOW_CONF)), ("chosen", chosen),
                     ("high", row_at(rows, HIGH_CONF))]:
        print(f"{row['conf']:>6.2f}{row['tp']:>7}{row['fp']:>8}{row['fn']:>7}"
              f"{row['duplicates']:>6}{row['ignored']:>6}"
              f"{row['precision']:>9.4f}{row['recall']:>9.4f}{row['f1']:>9.4f}"
              f"{row['detections_per_image']:>10.2f}   ({tag})")


def main():
    parser = argparse.ArgumentParser(
        description="Select the confidence operating point on validation by max F1 (section 10.4).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=PATH",
                        help="repeatable, e.g. --model 640=path/to/best.pt")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="override the inference size (default: the numeric model label)")
    parser.add_argument("--out", default=None, help="where to write the JSON report")
    parser.add_argument("--refresh", action="store_true",
                        help="re-run inference instead of reading the prediction cache")
    args = parser.parse_args()

    cfg = load_config(args.config)

    gt_path, n_images, n_annotations = build_gt_json(cfg, SPLIT, refresh=args.refresh)
    gt_annotations = load_gt_annotations(gt_path)
    n_crowd = sum(1 for ann in gt_annotations if int(ann.get("iscrowd", 0)) == 1)
    print(f"ground truth: {n_images} images, {n_annotations} annotations "
          f"({n_annotations - n_crowd} non-crowd + {n_crowd} iscrowd) -> {gt_path}")

    report = {
        "split": SPLIT,
        "criterion": SELECTION_CRITERION,
        "assumption": SELECTION_ASSUMPTION,
        "iou_threshold_for_matching": MATCH_IOU,
        "grid": {"min": CONF_MIN, "max": CONF_MAX, "step": CONF_STEP, "n_points": len(conf_grid())},
        "ground_truth": {
            "images": n_images,
            "annotations": n_annotations,
            "non_crowd_annotations": n_annotations - n_crowd,
            "iscrowd_annotations": n_crowd,
        },
        "models": {},
    }

    for spec in args.model:
        label, path = spec.split("=", 1)
        imgsz = imgsz_for_label(label, args.imgsz)
        detections, meta = predict_cached(cfg, path, label, SPLIT, imgsz, refresh=args.refresh)

        rows = sweep_model(gt_annotations, detections, meta["n_images"])
        chosen = choose_operating_point(rows)
        plateau = f1_plateau(rows, chosen)

        figure_path = cfg["paths"]["figures_dir"] / f"threshold_curves_{label}.png"
        plot_curves(rows, chosen, plateau, label, figure_path)

        report["models"][label] = {
            "checkpoint": meta["checkpoint"],
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "imgsz": meta["imgsz"],
            "detections_scored": meta["n_detections"],   # the conf=0.001 pool the sweep filters
            "chosen": {
                "conf": chosen["conf"],
                "precision": chosen["precision"],
                "recall": chosen["recall"],
                "f1": chosen["f1"],
                "tp": chosen["tp"],
                "fp": chosen["fp"],
                "fn": chosen["fn"],
                "duplicates": chosen["duplicates"],
                "detections_per_image": chosen["detections_per_image"],
            },
            "f1_plateau": plateau,
            "sweep": rows,
            "tradeoff_notes": tradeoff_notes(rows, chosen),
            "figure": str(figure_path),
        }
        print_summary(label, rows, chosen, plateau)

    out = Path(args.out) if args.out else cfg["paths"]["metrics_dir"] / "thresholds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
