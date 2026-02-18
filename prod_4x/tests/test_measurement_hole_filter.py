from __future__ import annotations

from app.services.measurement import _hole_far_outside_board, _is_nested_hole, _is_oval_hole

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
