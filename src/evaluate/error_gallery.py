"""Qualitative error analysis on the validation split (assignment §10.6).

§10.6 asks for failure cases to be looked at, not just counted: "collect examples of
failures, group them into categories, and say what you think is causing each group."
This script does the mechanical half of that job — it finds every failure the 640
baseline makes at the chosen operating point, sorts each one into exactly one category,
counts the categories, and renders side-by-side ground-truth vs prediction panels so a
human can actually look at them.

**The three explanatory columns of the taxonomy are deliberately left EMPTY.**
`likely_cause`, `layer` and `next_experiment` in reports/error_examples/taxonomy.csv are
written BY HAND after opening the rendered panels. A script cannot see why a box is
wrong — it can only see that it is wrong — so inventing a cause here would be a guess
dressed up as a measurement, and the manual pass is the deliverable §10.6 actually asks
for. The same warning is repeated in the CSV's header comment.

The taxonomy is a PARTITION, not a set of overlapping tags. Every false positive and
every false negative takes the FIRST category it matches, tested in this fixed order:

  false positives
    1 duplicate_detection       extra box on a ground-truth object that is already matched
    2 class_confusion           overlaps a DIFFERENT-class object at IoU >= 0.5
    3 loose_box                 overlaps a SAME-class object at 0.3 <= IoU < 0.5
    4 high_conf_false_positive  score > 0.5 and no ground-truth overlap (max IoU < 0.3)
    5 low_conf_false_positive   any other false positive
  false negatives (missed objects)
    6 missed_small              ground-truth area < 32^2 px (the COCO "small" definition)
    7 missed_rare_class         class in the rare band (< 100 train instances, §10.5)
    8 missed_border             ground-truth box touches an image edge (within 2 px)
    9 missed_other              any other miss

Because the order is fixed, counts add up: categories 1-5 sum to the matcher's false
positive count and 6-9 sum to its false negative count. Order carries meaning —
missed_small is tested before missed_rare_class, so a small rare object is reported as a
size failure, not a frequency failure. That is the conservative reading: size is the
measured property, rarity is the interesting hypothesis, and the hypothesis should not
be handed the ambiguous cases.

Dense scenes are handled separately. "This image has 34 objects and the model found 3"
is a property of the SCENE, not of any one box, so it cannot live in a per-object
partition without double counting. It is reported as its own line: mean per-image recall
in dense images (> 20 ground-truth instances) against sparse ones.

Protocol notes:
  - VALIDATION ONLY. The split is a module constant, not a CLI flag, so this script
    cannot be pointed at the frozen test split (§10 preamble). Test is touched exactly
    once, later, by a separate script.
  - The 640 model is the principal baseline, so the manual error analysis is done on it
    (a second model can be passed with --model, but the report describes 640).
  - Detections come from the shared prediction cache (conf 0.001, NMS IoU 0.7,
    max_det 300) and are then filtered at the operating threshold chosen on validation
    in §10.4 (reports/metrics/thresholds.json). No inference is re-run here.
  - Matching, including the iscrowd ignore rule, is coco_eval.match_detections at
    IoU 0.5. Categories are re-derived from that matcher's own output, so a box called a
    duplicate here is exactly a box the matcher counted as a duplicate.
  - Category tests are made against NON-CROWD ground truth only. iscrowd=1 boxes are
    ignore regions covering unlabelled groups (D6); the matcher has already removed the
    detections they excuse, and the ones it kept should not be explained by a region
    that was never an individual object.

Outputs (all under reports/error_examples/):
    taxonomy.csv                 one row per category; three columns left for the human
    index.md                     every rendered panel with its category and description
    {category}_{image_id}.png    the panels themselves

Usage:
    python -m src.evaluate.error_gallery \
        --model 640=artifacts/coco5k-detect-300ep/runs/detect/runs/detect-300ep-640/weights/best.pt
"""

import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: we only save files, never open windows
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from src.data.common import DEFAULT_CONFIG, load_config
from src.evaluate.coco_eval import (
    build_gt_json,
    category_maps,
    class_bands,
    iou_xywh,
    load_gt_annotations,
    match_detections,
    predict_cached,
)
from src.evaluate.eval_slices import IMGSZ_FROM_LABEL, MATCH_IOU, threshold_for_label

# Error analysis is a validation activity. Not a CLI option on purpose (§10 preamble):
# the test split is frozen until every decision, including this one, has been made.
SPLIT = "val"

# The principal baseline. §10.6 is a manual pass, so it is done on one model.
DEFAULT_MODEL = (
    "640=artifacts/coco5k-detect-300ep/runs/detect/runs/detect-300ep-640/weights/best.pt"
)

# --- category thresholds --------------------------------------------------------------
# COCO's own "small" definition, so missed_small lines up with AP_small in the size slice.
SMALL_AREA_MAX = 32 * 32
# A box within this many pixels of an image edge counts as touching it. 2 px absorbs the
# rounding in COCO's float coordinates without catching objects that are merely near it.
BORDER_TOLERANCE_PX = 2
# Below this IoU the boxes barely touch, so "no ground-truth overlap" means max IoU under
# it. Same floor as loose_box's lower bound, so the two rules meet without a gap.
NO_OVERLAP_IOU = 0.3
# "High confidence" for a false positive: the model was sure and it was wrong.
HIGH_CONF_SCORE = 0.5
# A scene counts as dense above this many non-crowd ground-truth instances.
DENSE_SCENE_MIN_GT = 20

# How many panels to render. §10.6 asks for at least 15; one per non-empty category
# first, then the most illustrative examples of the biggest categories.
TARGET_PANELS = 18

FP_CATEGORIES = (
    "duplicate_detection",
    "class_confusion",
    "loose_box",
    "high_conf_false_positive",
    "low_conf_false_positive",
)
FN_CATEGORIES = (
    "missed_small",
    "missed_rare_class",
    "missed_border",
    "missed_other",
)
CATEGORY_ORDER = FP_CATEGORIES + FN_CATEGORIES

# What the reader should look for in a panel of this category. Printed in the panel
# title and in index.md, so the human doing the manual pass knows what the claim is.
CATEGORY_LOOK_AT = {
    "duplicate_detection": "two prediction boxes on one object — the gold one is the extra "
                           "box, the red one beside it already matched the dashed ground truth",
    "class_confusion": "the gold box sits on a real object (dashed ground truth) but names "
                       "the wrong class",
    "loose_box": "right class, badly placed box — gold prediction against the dashed "
                 "green ground truth it should have covered",
    "high_conf_false_positive": "confident gold box with no ground-truth object under it",
    "low_conf_false_positive": "gold box no earlier category explains — read the line below "
                               "for what is (or is not) under it",
    "missed_small": "gold ground-truth object is tiny and has no prediction on the right",
    "missed_rare_class": "gold ground-truth object is a rare class and has no prediction on the right",
    "missed_border": "gold ground-truth object is cut off by the image edge and was not found",
    "missed_other": "gold ground-truth object is normal-sized, a well-represented class and "
                    "clear of the edges — it was simply not found",
}

# Colours: ground truth green, predictions red (§10.6 panels), the error itself gold.
COLOR_GT = "#1baf7a"
COLOR_PRED = "#d6392b"
COLOR_ERROR = "#f2b705"


# --------------------------------------------------------------------------------------
# categorisation
# --------------------------------------------------------------------------------------

def categorise_false_positive(detection, image_gt, matched_gt_ids):
    """Sort one false positive into the first FP category it matches.

    Returns (category, partner_gt, iou) where partner_gt is the ground-truth box that
    explains the category (the object being duplicated, the wrongly-classified object,
    the loosely-boxed object) or None when the category is "nothing is there".

    Rule 1 is written to agree with coco_eval.match_detections exactly: that matcher
    calls a detection a duplicate when it overlaps an ALREADY-MATCHED same-class box at
    IoU >= 0.5, so the same test on the same matched_gt_ids reproduces its duplicate
    count rather than estimating it.

    image_gt holds the non-crowd ground truth of this image only — crowd regions are
    ignore regions (D6) and the matcher has already dropped the detections they excuse.
    """
    same_class = [g for g in image_gt if g["category_id"] == detection["category_id"]]
    other_class = [g for g in image_gt if g["category_id"] != detection["category_id"]]

    # 1. duplicate: a second box on an object that some higher-scoring box already claimed
    partner, partner_iou, _ = _best_overlap(
        detection, [g for g in same_class if g["id"] in matched_gt_ids], min_iou=MATCH_IOU)
    if partner is not None:
        return "duplicate_detection", partner, partner_iou

    # 2. class confusion: found the object, named it wrong
    partner, partner_iou, _ = _best_overlap(detection, other_class, min_iou=MATCH_IOU)
    if partner is not None:
        return "class_confusion", partner, partner_iou

    # 3. loose box: right class, box too sloppy to count as a hit at IoU 0.5
    partner, partner_iou, _ = _best_overlap(
        detection, same_class, min_iou=NO_OVERLAP_IOU, max_iou=MATCH_IOU)
    if partner is not None:
        return "loose_box", partner, partner_iou

    # 4/5. nothing meaningful under the box; the score decides which of the two it is
    _, _, max_iou = _best_overlap(detection, image_gt, min_iou=0.0)
    if detection["score"] > HIGH_CONF_SCORE and max_iou < NO_OVERLAP_IOU:
        return "high_conf_false_positive", None, max_iou
    return "low_conf_false_positive", None, max_iou


def _best_overlap(detection, ground_truths, min_iou, max_iou=None):
    """Search `ground_truths` for the highest-IoU box inside the band [min_iou, max_iou).

    Returns (best_gt_in_band, its_iou, highest_iou_over_all_candidates). The third value
    ignores the band and is what rule 4 needs to say "nothing is under this box".

    Split out because all four false-positive rules are the same search with different
    bounds, and writing it once keeps them provably consistent.
    """
    best_gt = None
    best_gt_iou = 0.0
    highest_iou = 0.0
    for gt in ground_truths:
        overlap = iou_xywh(detection["bbox"], gt["bbox"])
        highest_iou = max(highest_iou, overlap)
        if overlap < min_iou or (max_iou is not None and overlap >= max_iou):
            continue
        if overlap > best_gt_iou:
            best_gt = gt
            best_gt_iou = overlap
    return best_gt, best_gt_iou, highest_iou


def categorise_false_negative(gt_annotation, image_info, rare_category_ids):
    """Sort one missed object into the first FN category it matches.

    Order is deliberate: size is tested before rarity, so a small rare object counts as
    missed_small. Both properties are true of it, and only one bucket may claim it; the
    measured, mundane explanation gets priority over the interesting hypothesis.

    Area comes from the annotation's own `area` field — the segmentation area COCO uses
    to define small/medium/large — so this category matches AP_small in the size slice
    rather than a second, differently-computed notion of "small".
    """
    if gt_annotation["area"] < SMALL_AREA_MAX:
        return "missed_small"
    if gt_annotation["category_id"] in rare_category_ids:
        return "missed_rare_class"
    if touches_border(gt_annotation["bbox"], image_info["width"], image_info["height"]):
        return "missed_border"
    return "missed_other"


def touches_border(bbox, width, height, tolerance=BORDER_TOLERANCE_PX):
    """True when a COCO [x, y, w, h] box reaches an image edge (within `tolerance` px).

    Border objects are usually truncated by the frame, so the model sees a fragment of
    the object and the annotation covers only that fragment too.
    """
    x, y, w, h = bbox
    return (x <= tolerance or y <= tolerance
            or x + w >= width - tolerance or y + h >= height - tolerance)


def collect_errors(gt_annotations, images_by_id, detections, result, rare_category_ids,
                   original_to_name):
    """Turn one matcher result into a flat list of categorised error records.

    Each record carries everything both later steps need: the counting step needs the
    category, and the rendering step needs the box, the partner box and a sentence
    describing what went wrong.

    `rank` orders examples within a category by "most illustrative first". For false
    positives that is the score (the model's most confident mistakes are the ones worth
    looking at); for misses it is the object area (a large missed object is a more
    legible panel and a more serious failure than a 20-pixel one).
    """
    non_crowd_by_image = {}
    for ann in gt_annotations:
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        non_crowd_by_image.setdefault(ann["image_id"], []).append(ann)
    gt_by_id = {ann["id"]: ann for ann in gt_annotations}
    matched_gt_ids = {gt_id for _, gt_id, _ in result["matches"]}

    records = []

    for det_idx in result["fp_indices"]:
        detection = detections[det_idx]
        image_gt = non_crowd_by_image.get(detection["image_id"], [])
        category, partner, overlap = categorise_false_positive(detection, image_gt, matched_gt_ids)
        name = original_to_name[detection["category_id"]]
        partner_name = original_to_name[partner["category_id"]] if partner else None
        records.append({
            "category": category,
            "kind": "fp",
            "image_id": detection["image_id"],
            # det_index / gt_id identify the offending box exactly, so the renderer
            # highlights THAT box and not another one that happens to have equal
            # coordinates.
            "det_index": det_idx,
            "gt_id": None,
            "box": detection["bbox"],
            "class_name": name,
            "score": detection["score"],
            "partner_box": partner["bbox"] if partner else None,
            "partner_name": partner_name,
            "iou": overlap,
            "rank": detection["score"],
            "description": _fp_description(category, name, detection["score"], partner_name, overlap),
        })

    for gt_id in result["fn_gt_ids"]:
        ann = gt_by_id[gt_id]
        image_info = images_by_id[ann["image_id"]]
        category = categorise_false_negative(ann, image_info, rare_category_ids)
        name = original_to_name[ann["category_id"]]
        records.append({
            "category": category,
            "kind": "fn",
            "image_id": ann["image_id"],
            "det_index": None,
            "gt_id": ann["id"],
            "box": ann["bbox"],
            "class_name": name,
            "score": None,
            "partner_box": None,
            "partner_name": None,
            "iou": None,
            "rank": ann["area"],
            "description": _fn_description(category, name, ann["area"]),
        })

    return records


def _fp_description(category, name, score, partner_name, overlap):
    """One sentence saying what this false positive is, for index.md and the panel.

    The last branch is wordy on purpose. low_conf_false_positive is the catch-all bucket,
    so it also collects the one awkward case the fixed category order creates: a
    CONFIDENT box that grazes an object of another class at IoU 0.3-0.5, which is neither
    class confusion (that needs IoU >= 0.5) nor a loose box (that needs the same class)
    nor a high-confidence false positive (that needs no overlap at all). Calling such a
    box "weak" because of the bucket it landed in would be a lie the panel would expose.
    """
    if category == "duplicate_detection":
        return (f"second '{name}' box (score {score:.2f}) on a '{name}' that was already "
                f"matched, IoU {overlap:.2f}")
    if category == "class_confusion":
        return (f"predicted '{name}' (score {score:.2f}) on a real '{partner_name}', "
                f"IoU {overlap:.2f}")
    if category == "loose_box":
        return (f"'{name}' (score {score:.2f}) on the right object but only IoU "
                f"{overlap:.2f} — too loose to count at 0.5")
    if category == "high_conf_false_positive":
        return (f"confident '{name}' (score {score:.2f}) with no ground-truth object under "
                f"it (best IoU {overlap:.2f})")
    if score > HIGH_CONF_SCORE:
        return (f"'{name}' at score {score:.2f} — confident, but it only grazes ground truth "
                f"(best IoU {overlap:.2f}): too little overlap for class confusion, wrong "
                f"class for a loose box, so it lands in the catch-all bucket")
    if overlap < NO_OVERLAP_IOU:
        return f"weak '{name}' (score {score:.2f}) with no ground-truth object under it"
    return (f"weak '{name}' (score {score:.2f}) grazing a ground-truth object at IoU "
            f"{overlap:.2f}")


def _fn_description(category, name, area):
    """One sentence saying what this miss is, for index.md and the panel."""
    if category == "missed_small":
        return f"missed '{name}', area {area:.0f} px^2 (COCO small is < {SMALL_AREA_MAX})"
    if category == "missed_rare_class":
        return f"missed '{name}', a rare class (< 100 train instances), area {area:.0f} px^2"
    if category == "missed_border":
        return f"missed '{name}' touching an image edge, area {area:.0f} px^2"
    return f"missed '{name}', area {area:.0f} px^2, not small, not rare, not on the border"


# --------------------------------------------------------------------------------------
# dense-scene statistic
# --------------------------------------------------------------------------------------

def dense_scene_recall(gt_annotations, result, min_gt=DENSE_SCENE_MIN_GT):
    """Mean per-image recall in dense scenes vs sparse ones (§10.6, scene-level).

    A crowded image is a different failure mode from any single bad box: the objects
    overlap, NMS suppresses neighbours, and max_det caps the output. That is a property
    of the scene, so it is measured per image and averaged over images (each image counts
    once, whether it holds 3 objects or 40) rather than folded into the per-object
    taxonomy, where it would double-count boxes already categorised as misses.

    Dense means more than `min_gt` non-crowd ground-truth instances. Crowd regions are
    excluded from the count for the same reason they are excluded from recall: they were
    never individual objects to find (D6).
    """
    gt_by_id = {ann["id"]: ann for ann in gt_annotations}
    per_image = {}
    for ann in gt_annotations:
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        entry = per_image.setdefault(ann["image_id"], {"n_gt": 0, "tp": 0})
        entry["n_gt"] += 1
    for _, gt_id, _ in result["matches"]:
        per_image[gt_by_id[gt_id]["image_id"]]["tp"] += 1

    dense, sparse = [], []
    for image_id, entry in per_image.items():
        recall = entry["tp"] / entry["n_gt"]
        (dense if entry["n_gt"] > min_gt else sparse).append(
            {"image_id": image_id, "n_gt": entry["n_gt"], "tp": entry["tp"], "recall": recall})

    def mean_recall(group):
        return sum(row["recall"] for row in group) / len(group) if group else 0.0

    dense.sort(key=lambda row: (row["recall"], -row["n_gt"]))
    return {
        "dense_definition": f"more than {min_gt} non-crowd ground-truth instances in the image",
        "dense_images": len(dense),
        "sparse_images": len(sparse),
        "dense_objects": sum(row["n_gt"] for row in dense),
        "sparse_objects": sum(row["n_gt"] for row in sparse),
        "mean_recall_dense": mean_recall(dense),
        "mean_recall_sparse": mean_recall(sparse),
        "worst_dense_images": dense[:5],
    }


# --------------------------------------------------------------------------------------
# taxonomy table
# --------------------------------------------------------------------------------------

def build_taxonomy(records):
    """Count the categories and pick up to five example images for each.

    Prevalence is a percentage of ALL errors (false positives plus false negatives), so
    the nine rows sum to 100%: the question the table answers is "of everything this
    model got wrong, how much is each kind", and splitting the denominator would make
    the FP and FN rows incomparable.
    """
    rows = []
    total = len(records)
    for category in CATEGORY_ORDER:
        in_category = sorted((r for r in records if r["category"] == category),
                             key=lambda r: -r["rank"])
        examples = []
        for record in in_category:                 # most illustrative first, one per image
            if record["image_id"] not in examples:
                examples.append(record["image_id"])
            if len(examples) == 5:
                break
        rows.append({
            "category": category,
            "count": len(in_category),
            "prevalence_pct": 100.0 * len(in_category) / total if total else 0.0,
            "example_image_ids": examples,
        })
    return rows


def write_taxonomy_csv(path, taxonomy_rows, header_comment_lines):
    """Write taxonomy.csv with the three explanatory columns EMPTY, on purpose.

    likely_cause / layer / next_experiment are the human's columns. The header comment
    says so in the file itself, because a CSV gets opened years later by someone who
    never read this docstring, and an empty column with no explanation looks like a bug.
    """
    with open(path, "w", newline="") as f:
        for line in header_comment_lines:
            f.write(f"# {line}\n")
        writer = csv.writer(f)
        writer.writerow(["category", "count", "prevalence_pct", "example_image_ids",
                         "likely_cause", "layer", "next_experiment"])
        for row in taxonomy_rows:
            writer.writerow([
                row["category"],
                row["count"],
                f"{row['prevalence_pct']:.2f}",
                ";".join(str(image_id) for image_id in row["example_image_ids"]),
                "", "", "",          # filled in by hand after looking at the panels
            ])


# --------------------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------------------

def choose_panels(records, target=TARGET_PANELS):
    """Pick which errors get rendered: one per non-empty category, then fill up.

    Three passes. The first guarantees coverage — every category that occurred at all
    gets a panel, so the reader can see what a "loose box" or a "class confusion" looks
    like even when there are only a handful. The second adds the MEDIAN member of the
    catch-all bucket: ranking by score would otherwise represent 438 mostly-weak boxes by
    their two most confident and least typical members. The third fills the remaining
    slots by taking the next most illustrative example from the biggest categories in
    turn, so the extra panels are spent where most of the errors actually are.

    Only one panel per (category, image_id) pair, because the file name is
    {category}_{image_id}.png and a second one would silently overwrite the first.
    """
    by_category = {}
    for category in CATEGORY_ORDER:
        seen_images = set()
        ordered = []
        for record in sorted((r for r in records if r["category"] == category),
                             key=lambda r: -r["rank"]):
            if record["image_id"] in seen_images:
                continue
            seen_images.add(record["image_id"])
            ordered.append(record)
        by_category[category] = ordered

    chosen = []
    taken = {category: set() for category in CATEGORY_ORDER}

    def take(category, position):
        """Add the example at `position` in a category's ranking, if it is still free."""
        if position in taken[category] or position >= len(by_category[category]):
            return
        taken[category].add(position)
        chosen.append(by_category[category][position])

    for category in CATEGORY_ORDER:                        # pass 1: coverage
        take(category, 0)
    take("low_conf_false_positive",                        # pass 2: a typical catch-all
         len(by_category["low_conf_false_positive"]) // 2)

    biggest_first = sorted(CATEGORY_ORDER, key=lambda c: -len(by_category[c]))
    longest = max((len(ordered) for ordered in by_category.values()), default=0)
    position = 1
    while len(chosen) < target and position < longest:     # pass 3: volume
        for category in biggest_first:
            if len(chosen) >= target:
                break
            take(category, position)
        position += 1
    return chosen


def draw_box(ax, bbox, color, label=None, linewidth=1.6, linestyle="solid", label_at="top"):
    """Draw one COCO [x, y, w, h] box with an optional label at a corner.

    label_at="bottom" puts the caption under the box instead of over it. Used for the
    dashed reference boxes, whose caption would otherwise land exactly where the
    prediction sitting on the same object already wrote its own.
    """
    x, y, w, h = bbox
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color,
                           linewidth=linewidth, linestyle=linestyle))
    if label:
        text_y = min(y + h + 3, ax.get_ylim()[0] - 2) if label_at == "bottom" else max(y - 3, 8)
        ax.text(x + 1, text_y, label, color="white", fontsize=6.5,
                va="top" if label_at == "bottom" else "bottom", ha="left",
                bbox={"facecolor": color, "edgecolor": "none", "pad": 1.0, "alpha": 0.85})


def point_arrow(ax, bbox, text):
    """Arrow pointing at the error box, labelled with the category.

    The panel is useless if the reader has to guess which of forty boxes is the one being
    talked about, so the offending box is both thickened and pointed at.
    """
    x, y, w, h = bbox
    ax.annotate(
        text, xy=(x + w / 2, y + h), xycoords="data",
        xytext=(0, -34), textcoords="offset points",
        color="black", fontsize=8, fontweight="bold", ha="center", va="top",
        bbox={"facecolor": COLOR_ERROR, "edgecolor": "black", "linewidth": 0.5, "pad": 2.0},
        arrowprops={"arrowstyle": "-|>", "color": COLOR_ERROR, "linewidth": 2.0,
                    "shrinkA": 1, "shrinkB": 2},
        annotation_clip=False,
    )


def render_panel(record, image_path, image_gt, image_dets, conf, out_path,
                 original_to_name):
    """Render one error panel: ground truth on the left, predictions on the right.

    Left  = every non-crowd ground-truth box in green with its class name.
    Right = every prediction at or above the operating threshold in red with class and
            score. That is the honest comparison — the reader sees the whole scene both
            ways, not a crop that has already decided what the answer is.

    The error itself is drawn thick in gold with an arrow. For a missed object the gold
    ground-truth box is also repeated dashed on the prediction side, because the failure
    is an ABSENCE: showing where nothing was predicted is the only way to see it.

    Draw order on the right panel matters and is deliberate. The ground-truth box that
    explains a false positive (the object being duplicated, mislabelled or loosely boxed)
    is drawn dashed in GREEN and drawn FIRST, so the red predictions land on top of it.
    Drawn last and in gold it hid the very box the panel is claiming exists — on a
    duplicate, the true-positive box sits within a pixel or two of the ground truth, so
    painting the ground truth over it left one visible box in a panel captioned "two".

    image_dets is a list of (index_into_the_full_detection_list, detection) pairs so the
    offending box is identified by index, not by comparing coordinates.
    """
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    width, height = image.size

    fig_width = 14.0
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_width * height / width / 2 + 1.5))
    for ax in axes:
        ax.imshow(image)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")

    error_box = record["box"]

    # ---- left: ground truth
    for ann in image_gt:
        is_error = ann["id"] == record["gt_id"]
        draw_box(axes[0], ann["bbox"],
                 COLOR_ERROR if is_error else COLOR_GT,
                 original_to_name[ann["category_id"]],
                 linewidth=3.5 if is_error else 1.6)
    if record["kind"] == "fn":
        point_arrow(axes[0], error_box, "missed object")
    axes[0].set_title(f"ground truth — {len(image_gt)} objects", fontsize=10)

    # ---- right: predictions above the operating threshold
    # The explaining ground-truth box goes down first so predictions draw over it.
    if record["partner_box"] is not None:
        partner_label = ("ground truth, already matched"
                         if record["category"] == "duplicate_detection"
                         else f"ground truth: {record['partner_name']}")
        draw_box(axes[1], record["partner_box"], COLOR_GT, partner_label,
                 linewidth=2.0, linestyle="dashed", label_at="bottom")
    for det_index, det in image_dets:
        is_error = det_index == record["det_index"]
        label = f"{original_to_name[det['category_id']]} {det['score']:.2f}"
        draw_box(axes[1], det["bbox"],
                 COLOR_ERROR if is_error else COLOR_PRED, label,
                 linewidth=3.5 if is_error else 1.6)
    if record["kind"] == "fp":
        point_arrow(axes[1], error_box, record["category"].replace("_", " "))
    else:
        draw_box(axes[1], error_box, COLOR_ERROR,
                 f"missed: {record['class_name']}", linewidth=2.5, linestyle="dashed")
        point_arrow(axes[1], error_box, "no prediction here")
    axes[1].set_title(f"predictions at conf >= {conf:g} — {len(image_dets)} boxes", fontsize=10)

    # Wrapped by hand: matplotlib does not reflow a suptitle, and an unwrapped sentence
    # runs off the edge of the figure where nobody can read it.
    title = "\n".join(
        textwrap.fill(line, width=120) for line in (
            f"image {record['image_id']} — {record['category']}",
            f"look at: {CATEGORY_LOOK_AT[record['category']]}",
            record["description"],
        )
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def write_index(path, panels, label, conf, taxonomy_rows, dense_stats, ground_truth):
    """List every rendered panel with its category and a one-line description."""
    lines = [
        f"# Error gallery — model {label} — {SPLIT} split",
        "",
        "Assignment §10.6. Generated by `src/evaluate/error_gallery.py`.",
        "",
        f"- operating threshold: conf {conf:g}, matcher IoU {MATCH_IOU} "
        f"(chosen on validation in §10.4)",
        f"- ground truth: {ground_truth['images']} images, {ground_truth['annotations']} "
        f"annotations ({ground_truth['non_crowd']} non-crowd + {ground_truth['crowd']} iscrowd "
        f"kept as ignore regions)",
        "- left panel = ground truth (green), right panel = predictions above the threshold "
        "(red); the error itself is gold, thickened and arrowed",
        "- `taxonomy.csv` counts every error; its `likely_cause`, `layer` and "
        "`next_experiment` columns are **left empty for the human** to fill in after "
        "looking at these panels",
        "- one real-world mistake can produce two rows: a wrong-class box is both a spurious "
        "detection (`class_confusion`) and an object nobody found (a `missed_*` row)",
        "",
        "## Dense-scene statistic (scene-level, not a per-object category)",
        "",
        f"Mean per-image recall in dense images ({dense_stats['dense_definition']}) is "
        f"**{dense_stats['mean_recall_dense']:.4f}** over {dense_stats['dense_images']} images, "
        f"against **{dense_stats['mean_recall_sparse']:.4f}** over "
        f"{dense_stats['sparse_images']} sparse images.",
        "",
        "## Panels",
        "",
        "One panel per category first, then the most illustrative examples of the biggest "
        "categories: false positives ranked by score (the model's most confident mistakes), "
        "misses ranked by object area (the most conspicuous failures, and the most legible "
        "panels). The catch-all `low_conf_false_positive` bucket also gets its median-scoring "
        "member, so it is not represented only by its least typical ones.",
        "",
        "| panel | category | image_id | what it shows |",
        "|---|---|---:|---|",
    ]
    for record in panels:
        file_name = f"{record['category']}_{record['image_id']}.png"
        lines.append(f"| [`{file_name}`]({file_name}) | {record['category']} "
                     f"| {record['image_id']} | {record['description']} |")
    lines += [
        "",
        "## Category counts",
        "",
        "| category | count | prevalence of all errors |",
        "|---|---:|---:|",
    ]
    for row in taxonomy_rows:
        lines.append(f"| {row['category']} | {row['count']} | {row['prevalence_pct']:.2f}% |")
    total = sum(row["count"] for row in taxonomy_rows)
    lines += ["", f"Total errors categorised: {total}.", ""]
    Path(path).write_text("\n".join(lines))


# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Qualitative error analysis and example gallery (§10.6), validation only.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model", default=DEFAULT_MODEL, metavar="LABEL=PATH",
                        help="the model to analyse; defaults to the 640 baseline")
    parser.add_argument("--conf", type=float, default=None,
                        help="override the operating threshold; by default it is read "
                             "from reports/metrics/thresholds.json (§10.4)")
    parser.add_argument("--panels", type=int, default=TARGET_PANELS,
                        help=f"how many panels to render (§10.6 asks for >= 15, "
                             f"default {TARGET_PANELS})")
    parser.add_argument("--refresh", action="store_true",
                        help="re-run inference instead of reading the prediction cache")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    cfg = load_config(args.config)
    label, model_path = args.model.split("=", 1)
    imgsz = IMGSZ_FROM_LABEL.get(label, 640)
    out_dir = Path(cfg["paths"]["metrics_dir"]).parent / "error_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path, n_images, n_annotations = build_gt_json(cfg, SPLIT, refresh=args.refresh)
    gt_annotations = load_gt_annotations(gt_path)
    n_crowd = sum(1 for a in gt_annotations if int(a.get("iscrowd", 0)) == 1)
    print(f"ground truth: {n_images} images, {n_annotations} annotations "
          f"({n_annotations - n_crowd} non-crowd + {n_crowd} iscrowd) -> {gt_path}")

    with open(gt_path) as f:
        images_by_id = {img["id"]: img for img in json.load(f)["images"]}

    print(f"\n[{label}] predicting at imgsz={imgsz} ...")
    detections, meta = predict_cached(cfg, model_path, label, SPLIT, imgsz,
                                      refresh=args.refresh, device=args.device)

    conf, source, is_fallback = threshold_for_label(cfg, label, args.conf)
    if is_fallback:
        print(f"!! [{label}] {source}")
    else:
        print(f"[{label}] operating threshold conf={conf:g} ({source})")

    result = match_detections(gt_annotations, detections, conf=conf, iou_thr=MATCH_IOU,
                              n_images=n_images)
    print(f"[{label}] at conf {conf:g}: tp {result['tp']} fp {result['fp']} fn {result['fn']} "
          f"duplicates {result['duplicates']} ignored-on-crowd {result['ignored']}")

    # Rare band comes from TRAIN instance counts (§10.5), not from val: "rare" is a
    # property of what the model was trained on, not of what it is being tested on.
    bands_info = class_bands(cfg, verbose=False)
    _, original_to_name = category_maps(cfg)
    name_to_category_id = {name: cat_id for cat_id, name in original_to_name.items()}
    rare_category_ids = {name_to_category_id[name] for name in bands_info["bands"]["rare"]}

    records = collect_errors(gt_annotations, images_by_id, detections, result,
                             rare_category_ids, original_to_name)

    # The taxonomy is a partition, so the category counts must reproduce the matcher's
    # own totals exactly. If they do not, a rule is overlapping or leaking and every
    # number below it is wrong — so this is an assertion, not a printout.
    n_fp = sum(1 for r in records if r["kind"] == "fp")
    n_fn = sum(1 for r in records if r["kind"] == "fn")
    assert n_fp == result["fp"], f"categorised {n_fp} false positives, matcher found {result['fp']}"
    assert n_fn == result["fn"], f"categorised {n_fn} misses, matcher found {result['fn']}"
    n_duplicate = sum(1 for r in records if r["category"] == "duplicate_detection")
    assert n_duplicate == result["duplicates"], (
        f"categorised {n_duplicate} duplicates, matcher counted {result['duplicates']}")

    taxonomy_rows = build_taxonomy(records)
    dense_stats = dense_scene_recall(gt_annotations, result)

    # The one honest caveat in the taxonomy: "high_conf_false_positive" requires BOTH a
    # high score and no overlap, so a confident box that grazes an object at IoU 0.3-0.5
    # falls through to low_conf_false_positive. Count those rather than hide them, and
    # measure the catch-all bucket's median score so "mostly weak boxes" is a number
    # rather than an impression.
    low_bucket_scores = sorted(r["score"] for r in records
                               if r["category"] == "low_conf_false_positive")
    confident_in_low_bucket = sum(1 for s in low_bucket_scores if s > HIGH_CONF_SCORE)
    low_bucket_median_score = (low_bucket_scores[len(low_bucket_scores) // 2]
                               if low_bucket_scores else 0.0)

    print(f"\n{'category':<26}{'count':>7}{'% of errors':>13}")
    for row in taxonomy_rows:
        print(f"{row['category']:<26}{row['count']:>7}{row['prevalence_pct']:>12.2f}%")
    print(f"{'TOTAL':<26}{len(records):>7}{100.0:>12.2f}%")
    print(f"\ndense-scene recall: mean per-image recall "
          f"{dense_stats['mean_recall_dense']:.4f} in {dense_stats['dense_images']} dense images "
          f"(> {DENSE_SCENE_MIN_GT} objects) vs {dense_stats['mean_recall_sparse']:.4f} in "
          f"{dense_stats['sparse_images']} sparse images")

    header_comment_lines = [
        f"Error taxonomy for model {label} on the {SPLIT} split at conf {conf:g} "
        f"(assignment section 10.6).",
        "Generated by src/evaluate/error_gallery.py. Do not edit the first four columns by hand.",
        "",
        "likely_cause, layer and next_experiment are INTENTIONALLY EMPTY. They are filled in",
        "BY HAND after opening the panels in this directory. A script can see that a box is",
        "wrong but not why, so any cause written here automatically would be a guess presented",
        "as a measurement. layer should be one of: data / representation / training /",
        "threshold / deployment.",
        "",
        f"Categories are a partition: each error takes the FIRST matching category in the "
        f"order {', '.join(CATEGORY_ORDER)}.",
        f"prevalence_pct is a percentage of all {len(records)} errors "
        f"({result['fp']} false positives + {result['fn']} misses), so the rows sum to 100.",
        "One real-world mistake can produce two rows: a wrong-class box is BOTH a spurious "
        "detection (class_confusion) AND an object nobody found (a missed_* row). That is the "
        "matcher's accounting, not double counting inside a bucket -- see missed_other_5169.png, "
        "where a bus predicted as a train appears on both sides.",
        f"example_image_ids are semicolon-separated, most illustrative first "
        f"(false positives by score, misses by object area).",
        f"Caveat: high_conf_false_positive needs score > {HIGH_CONF_SCORE} AND max IoU < "
        f"{NO_OVERLAP_IOU} with any ground truth; {confident_in_low_bucket} confident boxes "
        f"that graze an object instead sit in low_conf_false_positive, whose median score "
        f"is {low_bucket_median_score:.2f}.",
        f"Dense-scene statistic (scene-level, not a category): mean per-image recall "
        f"{dense_stats['mean_recall_dense']:.4f} in {dense_stats['dense_images']} dense images "
        f"(> {DENSE_SCENE_MIN_GT} objects) vs {dense_stats['mean_recall_sparse']:.4f} in "
        f"{dense_stats['sparse_images']} sparse images.",
    ]
    write_taxonomy_csv(out_dir / "taxonomy.csv", taxonomy_rows, header_comment_lines)

    # ---- panels ----------------------------------------------------------------------
    panels = choose_panels(records, target=args.panels)
    non_crowd_by_image = {}
    for ann in gt_annotations:
        if int(ann.get("iscrowd", 0)) == 0:
            non_crowd_by_image.setdefault(ann["image_id"], []).append(ann)
    # (index, detection) pairs: the index is how a panel knows which box is the error.
    dets_by_image = {}
    for det_index, det in enumerate(detections):
        if det["score"] >= conf:
            dets_by_image.setdefault(det["image_id"], []).append((det_index, det))

    images_dir = Path(cfg["paths"]["images_dir"])
    print(f"\nrendering {len(panels)} panels ...")
    for record in panels:
        image_info = images_by_id[record["image_id"]]
        out_path = out_dir / f"{record['category']}_{record['image_id']}.png"
        render_panel(
            record,
            images_dir / image_info["file_name"],
            non_crowd_by_image.get(record["image_id"], []),
            dets_by_image.get(record["image_id"], []),
            conf,
            out_path,
            original_to_name,
        )
        print(f"  {out_path.name}: {record['description']}")

    write_index(out_dir / "index.md", panels, label, conf, taxonomy_rows, dense_stats,
                {"images": n_images, "annotations": n_annotations,
                 "non_crowd": n_annotations - n_crowd, "crowd": n_crowd})

    # A machine-readable copy of everything the CSV summarises, for the report generator.
    summary = {
        "assignment_section": "10.6",
        "split": SPLIT,
        "label": label,
        "checkpoint": meta["checkpoint"],
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "operating_threshold": {"conf": conf, "source": source, "is_fallback": is_fallback},
        "match_iou": MATCH_IOU,
        "counts": {k: result[k] for k in ("tp", "fp", "fn", "duplicates", "ignored")},
        "category_definitions": {
            "order": list(CATEGORY_ORDER),
            "small_area_max_px2": SMALL_AREA_MAX,
            "border_tolerance_px": BORDER_TOLERANCE_PX,
            "loose_box_iou_range": [NO_OVERLAP_IOU, MATCH_IOU],
            "high_conf_score": HIGH_CONF_SCORE,
            "rare_band": "train instances < 100 (§10.5 bands)",
        },
        "categories": taxonomy_rows,
        "confident_boxes_in_low_conf_bucket": confident_in_low_bucket,
        "low_conf_bucket_median_score": low_bucket_median_score,
        "dense_scene": dense_stats,
        "panels": [{"file": f"{r['category']}_{r['image_id']}.png", "category": r["category"],
                    "image_id": r["image_id"], "description": r["description"]} for r in panels],
        "manual_columns_note": (
            "likely_cause, layer and next_experiment in taxonomy.csv are left empty by this "
            "script and filled in by hand after looking at the panels (§10.6)."),
    }
    with open(out_dir / "error_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"\nwritten: {out_dir / 'taxonomy.csv'} "
          f"(likely_cause / layer / next_experiment left EMPTY for the manual pass)")
    print(f"written: {out_dir / 'index.md'}")
    print(f"written: {out_dir / 'error_summary.json'}")
    print(f"written: {len(panels)} panels in {out_dir}")


if __name__ == "__main__":
    main()
