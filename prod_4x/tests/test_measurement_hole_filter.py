from __future__ import annotations

from app.services.measurement import _hole_far_outside_board


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
