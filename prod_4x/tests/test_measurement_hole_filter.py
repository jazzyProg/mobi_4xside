from __future__ import annotations

from app.services.measurement import (
    _hole_far_outside_board,
    _is_concentric_duplicate,
    _is_nested_hole,
    _is_oval_hole,
    _contour_circularity,
    _is_overlap_duplicate,
    _passes_circularity_filter,
    _passes_axis_ratio_filter,
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
    # default threshold is intentionally relaxed to reduce false negatives
    # from segmentation-induced ovality.
    assert not _is_oval_hole(maj_mm=12.0, min_mm=8.0)
    assert _is_oval_hole(maj_mm=12.0, min_mm=4.2)
    assert not _is_oval_hole(maj_mm=10.0, min_mm=9.5)


def test_large_hole_axis_ratio_filter_is_more_tolerant():
    # ratio ~= 1.35 fails strict small-hole check but passes large-hole gate
    assert not _passes_axis_ratio_filter(major_mm=13.5, minor_mm=10.0, diameter_mm=10.0)
    assert _passes_axis_ratio_filter(major_mm=13.5, minor_mm=10.0, diameter_mm=35.0)


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


def test_large_hole_allows_lower_circularity():
    assert _passes_circularity_filter(circularity=0.20, diameter_mm=30.0)


def test_small_hole_keeps_stricter_circularity():
    assert not _passes_circularity_filter(circularity=0.25, diameter_mm=10.0)


def test_overlap_duplicate_requires_close_centers():
    assert not _is_overlap_duplicate(
        center_a_mm=np.array([10.0, 10.0]),
        dia_a_mm=30.0,
        center_b_mm=np.array([20.0, 10.0]),
        dia_b_mm=28.0,
    )


def test_small_hole_concentric_duplicate_needs_closer_diameter_match():
    # 12 mm-class hole should not be dropped as concentric duplicate
    # when alternative contour diameter diverges too much.
    assert not _is_concentric_duplicate(
        center_a_mm=np.array([10.0, 10.0]),
        dia_a_mm=12.0,
        center_b_mm=np.array([10.1, 10.0]),
        dia_b_mm=16.0,
    )


def test_small_hole_overlap_duplicate_needs_closer_diameter_match():
    assert not _is_overlap_duplicate(
        center_a_mm=np.array([10.0, 10.0]),
        dia_a_mm=12.0,
        center_b_mm=np.array([10.2, 10.1]),
        dia_b_mm=17.0,
    )
