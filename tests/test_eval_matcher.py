"""The greedy detection matcher in src/evaluate/coco_eval.py (§10.4, §10.6).

These are the rules the whole error analysis rests on, so they are tested on tiny
hand-built boxes: no model, no checkpoint, no real data. Every expected number below
can be worked out on paper, which is the point — if the matcher drifts, the arithmetic
in the comments says what the answer should have been.

Boxes are COCO format [x, y, w, h] throughout.
"""

from src.evaluate.coco_eval import intersection_over_detection, iou_xywh, match_detections


def gt(ann_id, category_id, bbox, iscrowd=0, image_id=1):
    """One ground-truth annotation in the shape build_gt_json writes."""
    return {"id": ann_id, "image_id": image_id, "category_id": category_id,
            "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": iscrowd}


def det(category_id, bbox, score, image_id=1):
    """One detection in the shape predict_cached writes."""
    return {"image_id": image_id, "category_id": category_id, "bbox": bbox, "score": score}


def test_detection_on_crowd_region_is_ignored():
    """A detection inside an iscrowd region is neither a true nor a false positive.

    Crowd regions mean "many objects here, not labelled individually" (D6). The
    detection below sits fully inside the crowd box, so intersection-over-detection-area
    = 1.0 >= 0.5 and the COCO rule says ignore it. Note plain IoU would be
    100*100 / (200*200) = 0.25 -- below threshold -- which is exactly why crowds use the
    detection area as the denominator.
    """
    ground_truth = [gt(ann_id=1, category_id=1, bbox=[0, 0, 200, 200], iscrowd=1)]
    detections = [det(category_id=1, bbox=[50, 50, 100, 100], score=0.9)]

    result = match_detections(ground_truth, detections, conf=0.05)

    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["ignored"] == 1
    assert result["fn"] == 0          # crowd ground truth is never a miss
    assert result["fp_indices"] == []


def test_second_detection_on_same_gt_is_duplicate_and_false_positive():
    """Two boxes on one object: the better one is a TP, the other is a duplicate FP.

    Both detections overlap the single ground-truth box at IoU 1.0. The higher-scoring
    one (0.9) is matched first because the matcher is greedy on score; the second finds
    that ground truth already used and is booked as a duplicate -- and also as a false
    positive, because a spurious extra box on the same object is a real error.
    """
    ground_truth = [gt(ann_id=1, category_id=1, bbox=[10, 10, 50, 50])]
    detections = [
        det(category_id=1, bbox=[10, 10, 50, 50], score=0.6),
        det(category_id=1, bbox=[10, 10, 50, 50], score=0.9),
    ]

    result = match_detections(ground_truth, detections, conf=0.05)

    assert result["tp"] == 1
    assert result["duplicates"] == 1
    assert result["fp"] == 1          # the duplicate is counted as a false positive too
    assert result["fn"] == 0
    assert result["matches"][0][0] == 1        # index 1 = the score-0.9 detection
    assert result["fp_indices"] == [0]         # index 0 = the score-0.6 detection


def test_duplicate_rule_is_checked_before_the_crowd_rule():
    """A box that is BOTH a duplicate and on a crowd is a duplicate, not ignored.

    This pins the order of the cascade, because the two rules can fire on the same
    detection: the second box below duplicates the already-matched person AND sits
    inside the crowd-of-people region. The matcher checks "duplicate" first, so it is
    booked as duplicate + false positive rather than ignored.

    Rationale: only the crowd's BOUNDING BOX is available here, not its RLE mask, and
    that box often swallows individually-labelled objects nearby. Checking crowds first
    would let a wide crowd box excuse genuine double-detections of labelled objects.
    Ignoring is reserved for detections with no other explanation.

    On real val data at conf 0.25 this affects 10 of 876 false positives, and every
    single false positive that touches a crowd region is one of these duplicates --
    no "pure" false positive is ever charged against a crowd.
    """
    ground_truth = [
        gt(ann_id=1, category_id=1, bbox=[50, 50, 60, 60]),
        gt(ann_id=2, category_id=1, bbox=[0, 0, 400, 400], iscrowd=1),
    ]
    detections = [
        det(category_id=1, bbox=[50, 50, 60, 60], score=0.9),
        det(category_id=1, bbox=[50, 50, 60, 60], score=0.7),
    ]

    result = match_detections(ground_truth, detections, conf=0.05)

    assert result["tp"] == 1
    assert result["duplicates"] == 1
    assert result["fp"] == 1
    assert result["ignored"] == 0

    # The same detection with no ground-truth object under it IS ignored: the crowd
    # rule still applies to everything that is not explained as a duplicate.
    lone = match_detections([ground_truth[1]], [detections[0]], conf=0.05)
    assert lone["ignored"] == 1 and lone["fp"] == 0


def test_wrong_class_overlap_is_never_a_match():
    """A well-placed box with the wrong label is a false positive AND a false negative.

    The detection overlaps the ground truth at IoU 0.9 but is class 2 against a class 1
    object, so it cannot match: the model both hallucinated a class-2 object (FP) and
    missed the class-1 one (FN). Class confusion must not be scored as a near-miss.

    IoU arithmetic: gt [0, 0, 100, 100], det [0, 0, 100, 90].
      intersection = 100 * 90                    = 9000
      union        = 100*100 + 100*90 - 9000     = 10000
      IoU          = 9000 / 10000                = 0.90
    """
    ground_truth = [gt(ann_id=1, category_id=1, bbox=[0, 0, 100, 100])]
    detections = [det(category_id=2, bbox=[0, 0, 100, 90], score=0.9)]

    assert abs(iou_xywh(detections[0]["bbox"], ground_truth[0]["bbox"]) - 0.90) < 1e-12

    result = match_detections(ground_truth, detections, conf=0.05)

    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["tp"] == 0
    assert result["duplicates"] == 0
    assert result["ignored"] == 0
    assert result["fn_gt_ids"] == [1]


def test_perfect_match_gives_precision_and_recall_one():
    """One object, one correct detection: TP=1, precision=1.0, recall=1.0, F1=1.0."""
    ground_truth = [gt(ann_id=1, category_id=3, bbox=[20, 30, 40, 60])]
    detections = [det(category_id=3, bbox=[20, 30, 40, 60], score=0.8)]

    result = match_detections(ground_truth, detections, conf=0.05)

    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["matches"] == [(0, 1, 1.0)]


def test_iou_hand_worked_example():
    """IoU on boxes worked out by hand, so the geometry itself is pinned down.

    A = [0, 0, 10, 10]  (area 100), B = [5, 5, 10, 10]  (area 100).
      overlap region = x in [5, 10], y in [5, 10]  ->  5 * 5   = 25
      union          = 100 + 100 - 25                        = 175
      IoU            = 25 / 175                              = 1/7 = 0.142857...
    """
    assert abs(iou_xywh([0, 0, 10, 10], [5, 5, 10, 10]) - 25 / 175) < 1e-12

    # Boxes that only touch along an edge have zero overlap, not a sliver.
    assert iou_xywh([0, 0, 10, 10], [10, 0, 10, 10]) == 0.0
    # Identical boxes are 1.0.
    assert iou_xywh([3, 4, 7, 8], [3, 4, 7, 8]) == 1.0

    # Intersection-over-detection-area for the crowd rule: a 10x10 detection half
    # inside a 100x100 crowd box covers 5*10 = 50 of its own 100 -> 0.5.
    assert abs(intersection_over_detection([95, 0, 10, 10], [0, 0, 100, 100]) - 0.5) < 1e-12


def test_confidence_threshold_filters_low_scoring_detections():
    """Detections below conf never enter the matcher, so they cost nothing and gain nothing.

    Two objects, two detections, but only the 0.9 one clears conf=0.5: one TP, one
    missed object, precision 1/1 = 1.0, recall 1/2 = 0.5.
    """
    ground_truth = [
        gt(ann_id=1, category_id=1, bbox=[0, 0, 50, 50]),
        gt(ann_id=2, category_id=1, bbox=[100, 100, 50, 50]),
    ]
    detections = [
        det(category_id=1, bbox=[0, 0, 50, 50], score=0.9),
        det(category_id=1, bbox=[100, 100, 50, 50], score=0.2),
    ]

    result = match_detections(ground_truth, detections, conf=0.5)

    assert result["n_detections"] == 1
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 1
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5


def test_matching_is_per_image():
    """A detection cannot claim an identical box in a different image."""
    ground_truth = [gt(ann_id=1, category_id=1, bbox=[0, 0, 50, 50], image_id=1)]
    detections = [det(category_id=1, bbox=[0, 0, 50, 50], score=0.9, image_id=2)]

    result = match_detections(ground_truth, detections, conf=0.05)

    assert result["tp"] == 0
    assert result["fp"] == 1
    assert result["fn"] == 1


def test_greedy_matcher_gives_best_gt_to_highest_scoring_detection():
    """With two objects and two overlapping detections, the higher score claims first.

    Detection A (score 0.9) overlaps gt 1 exactly. Detection B (score 0.7) also overlaps
    gt 1 (IoU 1.0) and gt 2 partially, but gt 1 is taken by then. B's IoU with gt 2 is
    below 0.5, so B ends up a duplicate rather than a second TP.
    """
    ground_truth = [
        gt(ann_id=1, category_id=1, bbox=[0, 0, 100, 100]),
        gt(ann_id=2, category_id=1, bbox=[400, 400, 100, 100]),
    ]
    detections = [
        det(category_id=1, bbox=[0, 0, 100, 100], score=0.7),
        det(category_id=1, bbox=[0, 0, 100, 100], score=0.9),
    ]

    result = match_detections(ground_truth, detections, conf=0.05)

    assert result["matches"] == [(1, 1, 1.0)]
    assert result["duplicates"] == 1
    assert result["fn_gt_ids"] == [2]


def test_detections_per_image_uses_split_size_when_given():
    """detections_per_image is output volume, so empty images must count in the denominator."""
    ground_truth = [gt(ann_id=1, category_id=1, bbox=[0, 0, 50, 50], image_id=1)]
    detections = [det(category_id=1, bbox=[0, 0, 50, 50], score=0.9, image_id=1)]

    seen_only = match_detections(ground_truth, detections, conf=0.05)
    whole_split = match_detections(ground_truth, detections, conf=0.05, n_images=10)

    assert seen_only["detections_per_image"] == 1.0
    assert whole_split["detections_per_image"] == 0.1
