import math
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from tagsort import ModelError
from tagsort.models import DEFAULT_MODEL, load_manifest, model_files
from tagsort.pipeline.detect import (
    DetectedLine,
    Detector,
    OnnxDetector,
    _components,
    _convex_hull,
    _rect_corners,
    boxes_from_map,
    min_area_rect,
)


def test_components_follow_8_connectivity() -> None:
    mask = np.array(
        [
            [1, 1, 0, 0, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    groups = sorted(sorted(g) for g in _components(mask))
    assert groups == [[(0, 0, 1), (1, 2, 2)], [(1, 4, 4), (2, 4, 4)], [(3, 0, 0)]]


def test_components_of_an_empty_mask() -> None:
    assert _components(np.zeros((4, 4), dtype=bool)) == []


def test_u_shape_is_one_component() -> None:
    mask = np.zeros((5, 7), dtype=bool)
    mask[0:5, 0] = mask[0:5, 6] = mask[4, :] = True
    assert len(_components(mask)) == 1


def test_convex_hull() -> None:
    points = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1], [1, 0]], dtype=float)
    assert sorted(map(tuple, _convex_hull(points))) == [(0, 0), (0, 2), (2, 0), (2, 2)]
    assert len(_convex_hull(np.array([[1.0, 1.0], [1.0, 1.0]]))) == 1


@pytest.mark.parametrize("degrees", [0, 15, 30, 45, -20, 89])
def test_min_area_rect_recovers_a_rotated_rectangle(degrees: float) -> None:
    angle = math.radians(degrees)
    corners = _rect_corners((50.0, 40.0), (30.0, 8.0), angle)
    center, (w, h), found = min_area_rect(np.array(corners))
    assert center == pytest.approx((50.0, 40.0))
    assert sorted((w, h)) == pytest.approx([8.0, 30.0])
    assert w * h == pytest.approx(240.0)
    assert -math.pi / 2 < found <= math.pi / 2


def test_min_area_rect_of_one_point() -> None:
    assert min_area_rect(np.array([[3.0, 4.0]])) == ((3.0, 4.0), (0.0, 0.0), 0.0)


def test_rect_corners_are_clockwise_from_top_left() -> None:
    tl, tr, br, bl = _rect_corners((10.0, 10.0), (8.0, 4.0), 0.0)
    assert (tl, tr, br, bl) == ((6.0, 8.0), (14.0, 8.0), (14.0, 12.0), (6.0, 12.0))


def test_boxes_from_map() -> None:
    probability = np.zeros((40, 60))
    probability[10:16, 5:45] = 0.9  # a text line
    probability[30:32, 50:52] = 0.9  # too small
    probability[25:28, 5:20] = 0.25  # above threshold but low score
    (line,) = boxes_from_map(probability, threshold=0.2, box_threshold=0.4, unclip_ratio=1.5)
    xs = [x for x, _ in line.quad]
    ys = [y for _, y in line.quad]
    assert line.score == pytest.approx(0.9)
    # The 40 x 6 region grows by area * ratio / perimeter = 240 * 1.5 / 92 on each side.
    offset = 240 * 1.5 / 92
    assert min(xs) == pytest.approx(5 - offset)
    assert max(xs) == pytest.approx(45 + offset)
    assert min(ys) == pytest.approx(10 - offset)
    assert max(ys) == pytest.approx(16 + offset)


def model() -> Any:
    try:
        return model_files(DEFAULT_MODEL)["det"]
    except ModelError:
        pytest.skip("run `tagsort models download` to test with the real model")


def test_detects_rendered_lines_with_the_real_model() -> None:
    detector = OnnxDetector(model(), load_manifest(DEFAULT_MODEL).detection, limit_side=1600)
    assert isinstance(detector, Detector)
    image = Image.new("RGB", (1200, 900), (235, 230, 220))
    font = ImageFont.load_default(size=48)
    draw = ImageDraw.Draw(image)
    draw.text((100, 120), "GJ07966", fill="black", font=font)
    draw.text((600, 650), "0086", fill="black", font=font)
    lines = detector.detect(image)
    assert all(isinstance(line, DetectedLine) for line in lines)
    centers = sorted(
        (round(sum(x for x, _ in ln.quad) / 4), round(sum(y for _, y in ln.quad) / 4))
        for ln in lines
    )
    assert len(centers) == 2
    assert 150 < centers[0][0] < 350
    assert 100 < centers[0][1] < 200
    assert 600 < centers[1][0] < 800
    assert 640 < centers[1][1] < 740
    assert detector.detect(Image.new("RGB", (300, 200), "white")) == []
