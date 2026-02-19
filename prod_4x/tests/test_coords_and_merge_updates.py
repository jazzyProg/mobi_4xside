from __future__ import annotations

from app.utils import mergecoords
from app.utils import tilestocoords as t2c


def test_denormalize_keeps_out_of_bounds_for_later_filtering() -> None:
    coords = [0.0, 0.0, 1.0, 1.0]
    out = t2c.denormalize(coords, x_off=-160, y_off=-160)
    assert out[0] < 0.0
    assert out[1] < 0.0
    assert out[2] > 0.0
    assert out[3] > 0.0


def test_collect_labels_filters_out_of_frame_pairs(tmp_path) -> None:
    p = tmp_path / "tile_0-0.txt"
    p.write_text("1 0.0 0.0 0.2 0.2 0.3 0.3 0.4 0.4\n", encoding="utf-8")

    lines = t2c.collect_labels(tmp_path)

    assert len(lines) == 1
    parts = lines[0].split()
    # after filtering, only in-frame points remain => 3 pairs
    assert parts[0] == "1"
    assert len(parts) == 1 + 6


def test_mergecoords_filters_small_class1_after_combine() -> None:
    # Two tiny class-1 contours should be clustered but discarded (< 10 points after combine).
    input_lines = [
        "0 0 0 100 0 100 100 0 100",
        "1 10 10 11 10 11 11",
        "1 13 10 14 10 14 11",
    ]

    cls0, cls1, cls3 = mergecoords.main(inputpath=None, inputlines=input_lines)

    assert len(cls0) >= 3
    assert cls1 == []
    assert cls3 == []
