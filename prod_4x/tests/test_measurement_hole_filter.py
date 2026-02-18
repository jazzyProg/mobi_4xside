from __future__ import annotations

from app.services.measurement import (
    _hole_far_outside_board,
    _is_concentric_duplicate,
    _is_nested_hole,
    _is_oval_hole,
    _contour_circularity,
    _is_overlap_duplicate,
)

import numpy as np


def test_corner_hole_is_not_rejected_when_overlap_covers_jitter():
    # Corner/edge hole: center can be slightly outside due to OBB jitter,
    # but hole itself still intersects the board.
    assert not _hole_far_outside_board(
        dist_left=-3.0,
        dist_right=30.0,
        dist_bot=-2.5,
        dist_top=28.0,
        radius_mm=4.0,
    )


def test_hole_is_rejected_when_center_too_far_outside_even_with_radius_margin():
    assert _hole_far_outside_board(
        dist_left=-7.0,
        dist_right=20.0,
        dist_bot=15.0,
        dist_top=10.0,
        radius_mm=4.0,
    )


def test_nested_hole_is_detected_and_can_be_collapsed_to_larger_one():
    assert _is_nested_hole(
        inner_center_mm=np.array([10.0, 10.0]),
        inner_dia_mm=6.0,
        outer_center_mm=np.array([10.2, 10.0]),
        outer_dia_mm=12.0,
    )


def test_not_nested_when_centers_are_far_apart():
    assert not _is_nested_hole(
        inner_center_mm=np.array([10.0, 10.0]),
        inner_dia_mm=6.0,
        outer_center_mm=np.array([18.0, 10.0]),
        outer_dia_mm=12.0,
    )


def test_oval_hole_is_ignored_by_default_ratio_threshold():
    assert _is_oval_hole(maj_mm=12.0, min_mm=8.0)
    assert not _is_oval_hole(maj_mm=10.0, min_mm=9.5)


def test_concentric_duplicate_is_detected():
    assert _is_concentric_duplicate(
        center_a_mm=np.array([10.0, 10.0]),
        dia_a_mm=10.0,
        center_b_mm=np.array([10.2, 10.1]),
        dia_b_mm=11.5,
    )


def test_concentric_duplicate_rejects_large_diameter_gap():
    assert not _is_concentric_duplicate(
        center_a_mm=np.array([10.0, 10.0]),
        dia_a_mm=6.0,
        center_b_mm=np.array([10.1, 10.1]),
        dia_b_mm=15.0,
    )



def test_contour_circularity_penalizes_elongated_shape():
    ang = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    circle = np.stack([50 + 20 * np.cos(ang), 50 + 20 * np.sin(ang)], axis=1).astype(np.float32).reshape(-1, 1, 2)
    ellipse = np.stack([50 + 35 * np.cos(ang), 50 + 12 * np.sin(ang)], axis=1).astype(np.float32).reshape(-1, 1, 2)
    assert _contour_circularity(circle) > _contour_circularity(ellipse)


def test_overlap_duplicate_detected_for_mostly_overlapping_circles():
    assert _is_overlap_duplicate(
        center_a_mm=np.array([10.0, 10.0]),
        dia_a_mm=20.0,
        center_b_mm=np.array([12.0, 10.0]),
        dia_b_mm=19.0,
    )


def test_overlap_duplicate_not_detected_for_separate_circles():
    assert not _is_overlap_duplicate(
        center_a_mm=np.array([10.0, 10.0]),
        dia_a_mm=10.0,
        center_b_mm=np.array([24.0, 10.0]),
        dia_b_mm=10.0,
    )
