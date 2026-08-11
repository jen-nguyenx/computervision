"""Geometry helpers: bbox_to_yolo / yolo_to_bbox / clip_bbox (D5, D10).

Hand-computed values, exact inversion, random round-trips, and clipping
behavior at the image border.
"""

import random

from src.data.convert_coco import bbox_to_yolo, clip_bbox, yolo_to_bbox


def test_bbox_to_yolo_hand_computed():
    # COCO bbox [x=10, y=20, w=30, h=40] in a 100x200 image (D5):
    # cx = (10 + 30/2) / 100 = 0.25, cy = (20 + 40/2) / 200 = 0.20
    # w  = 30 / 100 = 0.30,          h  = 40 / 200 = 0.20
    cx, cy, w, h = bbox_to_yolo([10, 20, 30, 40], 100, 200)
    assert abs(cx - 0.25) < 1e-12
    assert abs(cy - 0.20) < 1e-12
    assert abs(w - 0.30) < 1e-12
    assert abs(h - 0.20) < 1e-12


def test_yolo_to_bbox_inverts_hand_computed():
    yolo = bbox_to_yolo([10, 20, 30, 40], 100, 200)
    back = yolo_to_bbox(yolo, 100, 200)
    for got, expected in zip(back, [10, 20, 30, 40]):
        assert abs(got - expected) < 1e-9


def test_round_trip_random_boxes():
    rng = random.Random(0)
    for _ in range(50):
        img_w = rng.randint(50, 2000)
        img_h = rng.randint(50, 2000)
        x = rng.uniform(0, img_w - 2)
        y = rng.uniform(0, img_h - 2)
        w = rng.uniform(0.5, img_w - x)
        h = rng.uniform(0.5, img_h - y)
        back = yolo_to_bbox(bbox_to_yolo([x, y, w, h], img_w, img_h), img_w, img_h)
        for got, expected in zip(back, [x, y, w, h]):
            assert abs(got - expected) < 1e-9


def test_clip_bbox_clamps_out_of_bounds():
    # Box [90, 180, 30, 40] in a 100x200 image sticks out past the right and
    # bottom edges: x2 = 120 -> 100, y2 = 220 -> 200 (D10).
    clipped, was_clipped = clip_bbox([90, 180, 30, 40], 100, 200)
    assert was_clipped is True
    assert clipped == [90.0, 180.0, 10.0, 20.0]


def test_clip_bbox_clamps_negative_origin():
    # Box starting before (0, 0): x1 = -5 -> 0, y1 = -10 -> 0.
    clipped, was_clipped = clip_bbox([-5, -10, 30, 40], 100, 200)
    assert was_clipped is True
    assert clipped == [0.0, 0.0, 25.0, 30.0]


def test_clip_bbox_leaves_interior_box_untouched():
    clipped, was_clipped = clip_bbox([10, 20, 30, 40], 100, 200)
    assert was_clipped is False
    assert clipped == [10.0, 20.0, 30.0, 40.0]


def test_clip_bbox_fully_outside_becomes_zero_area():
    # A box entirely past the border collapses to zero width; the converter
    # then drops it under excluded.zero_area (D12).
    clipped, was_clipped = clip_bbox([150, 50, 30, 40], 100, 200)
    assert was_clipped is True
    assert clipped[2] == 0.0
