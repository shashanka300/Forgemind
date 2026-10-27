"""Bench region: crop + mask applied to static frames only."""
import numpy as np
import pytest
from PIL import Image

from conftest import COLORS, scene
from forgemind.perception import bench
from forgemind.providers.base import Detection


def test_parse_and_validate():
    assert bench.parse_roi("") is None
    assert bench.parse_roi("0.1,0.2,0.9,0.8") == (0.1, 0.2, 0.9, 0.8)
    with pytest.raises(ValueError):
        bench.parse_roi("0.9,0.2,0.1,0.8")
    assert bench.parse_masks('[[0, 0.6, 1, 1]]') == [(0.0, 0.6, 1.0, 1.0)]


def test_apply_crops_and_paints_masks_in_crop_coords():
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    out = bench.apply(img, (0.5, 0.0, 1.0, 1.0), [(0.75, 0.5, 1.2, 1.0)])
    a = np.asarray(out)
    assert out.size == (50, 100)
    assert tuple(a[10, 10]) == (255, 0, 0)            # kept
    assert tuple(a[80, 40]) == bench.MASK_FILL         # masked (x 0.75..1 -> 25..50 in crop)
    assert bench.apply(img) is img                     # nothing configured -> untouched


def test_static_frames_lose_masked_objects_mobile_frames_dont(rt):
    rt.settings.bench_masks = [(0.0, 0.6, 1.0, 1.0)]  # bottom 40% is the "keyboard"
    img = scene([("multimeter", .4, .1, .55, .3), ("soldering iron", .4, .7, .6, .9)])
    out = rt.perception.process("static", img, ts=1000.0)
    assert sorted(d["concept"] for d in out["detections"]) == ["multimeter"]
    out = rt.perception.process("mobile", img, ts=1001.0)
    assert sorted(d["concept"] for d in out["detections"]) == ["multimeter", "soldering iron"]


def test_suggest_masks_keyboard_but_not_a_laptop_with_boards_on_it():
    class Seg:
        def segment(self, image, concepts):
            return [Detection("desk", (0.1, 0.2, 0.9, 1.0), 0.9),
                    Detection("keyboard", (0.3, 0.7, 0.7, 0.9), 0.95),
                    Detection("laptop", (0.5, 0.0, 0.95, 0.45), 0.9),
                    Detection("circuit board", (0.55, 0.05, 0.7, 0.3), 0.9),
                    Detection("computer monitor", (0.0, 0.0, 0.2, 0.2), 0.3)]   # too unsure
    roi, masks, dropped = bench.suggest(Image.new("RGB", (64, 64)), Seg(), ["circuit board"])
    assert roi is None                                   # cropping is opt-in
    assert masks == [(0.29, 0.69, 0.71, 0.91)]
    assert dropped == [("laptop", (0.49, 0.0, 0.96, 0.46))]
    roi, _, _ = bench.suggest(Image.new("RGB", (64, 64)), Seg(), ["circuit board"], with_roi=True)
    assert roi == (0.09, 0.19, 0.91, 1.0)
