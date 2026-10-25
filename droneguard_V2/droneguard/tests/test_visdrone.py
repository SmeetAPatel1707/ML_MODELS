from pathlib import Path

import pytest
from PIL import Image

from droneguard.visdrone import CLASS_NAMES, convert_dataset, parse_line


def test_pedestrian_maps_to_class_zero():
    box = parse_line("10,20,30,40,1,1,0,0", 100, 100)
    assert box is not None
    cls, cx, cy, w, h = box
    assert cls == 0 and CLASS_NAMES[cls] == "pedestrian"
    assert (cx, cy, w, h) == pytest.approx((0.25, 0.40, 0.30, 0.40))


def test_motor_maps_to_last_class():
    assert parse_line("0,0,10,10,1,10,0,0", 100, 100)[0] == 9


@pytest.mark.parametrize("line", [
    "0,0,10,10,1,0,0,0",   # ignored region
    "0,0,10,10,1,11,0,0",  # others
    "0,0,10,10,0,4,0,0",   # score 0 -> ignore
    "0,0,0,10,1,4,0,0",    # zero width
    "200,200,10,10,1,4,0,0",  # fully outside image
    "1,2,3",               # malformed
    "",
])
def test_skipped_lines(line):
    assert parse_line(line, 100, 100) is None


def test_box_is_clipped_to_image():
    cls, cx, cy, w, h = parse_line("90,90,20,20,1,4,0,0", 100, 100)
    assert (cx, cy, w, h) == pytest.approx((0.95, 0.95, 0.10, 0.10))


def _make_split(root: Path, name: str, nested: bool = False):
    d = root / name / name if nested else root / name
    (d / "images").mkdir(parents=True)
    (d / "annotations").mkdir(parents=True)
    Image.new("RGB", (200, 100)).save(d / "images" / "0001.jpg")
    (d / "annotations" / "0001.txt").write_text(
        "10,10,20,20,1,1,0,0\n"   # pedestrian -> kept
        "50,50,20,20,0,4,0,0\n"   # score 0 -> skipped
        "0,0,5,5,1,0,0,0\n"       # ignored region -> skipped
        "60,20,40,30,1,4,0,0\n"   # car -> kept
    )


def test_convert_dataset_end_to_end(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_split(src, "VisDrone2019-DET-train", nested=True)  # Kaggle-style nesting
    _make_split(src, "VisDrone2019-DET-val")

    stats = convert_dataset(src, dst)

    assert set(stats) == {"train", "val"}
    assert stats["train"].boxes_kept == 2 and stats["train"].boxes_skipped == 2
    assert stats["train"].per_class["pedestrian"] == 1 and stats["train"].per_class["car"] == 1

    labels = (dst / "labels" / "val" / "0001.txt").read_text().splitlines()
    assert [l.split()[0] for l in labels] == ["0", "3"]
    assert (dst / "images" / "val" / "0001.jpg").exists()

    yaml = (dst / "data.yaml").read_text()
    assert "val: images/val" in yaml and "nc: 10" in yaml and "0: pedestrian" in yaml


def test_refuses_without_val_split(tmp_path):
    src = tmp_path / "src"
    _make_split(src, "VisDrone2019-DET-train")
    with pytest.raises(RuntimeError):
        convert_dataset(src, tmp_path / "dst")
