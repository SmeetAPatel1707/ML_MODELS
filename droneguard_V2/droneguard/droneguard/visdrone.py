"""Convert VisDrone2019-DET annotations to YOLO format.

VisDrone annotation line format (one object per line):
    bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion

object_category:
    0 = ignored region   (skipped)
    1 = pedestrian ... 10 = motor   (kept, mapped to YOLO ids 0..9)
    11 = others          (skipped)

In ground-truth files, score == 0 marks a box to be ignored, so those are skipped too.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

# VisDrone split folder name -> YOLO split name
SPLITS = {
    "VisDrone2019-DET-train": "train",
    "VisDrone2019-DET-val": "val",
    "VisDrone2019-DET-test-dev": "test",
}


@dataclass
class SplitStats:
    images: int = 0
    boxes_kept: int = 0
    boxes_skipped: int = 0
    missing_images: int = 0
    per_class: dict[str, int] = field(default_factory=lambda: {n: 0 for n in CLASS_NAMES})


def parse_line(line: str, img_w: int, img_h: int) -> tuple[int, float, float, float, float] | None:
    """Parse one VisDrone annotation line into a YOLO box, or None if it should be skipped.

    Returns (class_id, x_center, y_center, width, height), all coordinates normalised to 0..1.
    """
    parts = [p for p in line.strip().split(",") if p != ""]
    if len(parts) < 6:
        return None

    x, y, w, h, score, category = (int(float(v)) for v in parts[:6])

    if score == 0 or category < 1 or category > len(CLASS_NAMES):
        return None

    # Clip the box to the image and drop degenerate boxes.
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return None

    bw, bh = x2 - x1, y2 - y1
    return (
        category - 1,
        (x1 + bw / 2) / img_w,
        (y1 + bh / 2) / img_h,
        bw / img_w,
        bh / img_h,
    )


def find_split_dir(root: Path, split_name: str) -> Path | None:
    """Locate a split folder that contains `images/` and `annotations/`.

    Handles both `root/<split>/images` and the nested Kaggle layout `root/<split>/<split>/images`.
    """
    for candidate in (root / split_name / split_name, root / split_name):
        if (candidate / "images").is_dir() and (candidate / "annotations").is_dir():
            return candidate
    return None


def convert_split(src: Path, dst_root: Path, yolo_split: str, copy_images: bool = True) -> SplitStats:
    """Convert one VisDrone split into `dst_root/images/<split>` and `dst_root/labels/<split>`."""
    img_out = dst_root / "images" / yolo_split
    lbl_out = dst_root / "labels" / yolo_split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    stats = SplitStats()
    for ann_path in sorted((src / "annotations").glob("*.txt")):
        img_path = src / "images" / f"{ann_path.stem}.jpg"
        if not img_path.exists():
            stats.missing_images += 1
            continue

        with Image.open(img_path) as im:  # reads the header only, not the pixels
            img_w, img_h = im.size

        lines = []
        for raw in ann_path.read_text().splitlines():
            box = parse_line(raw, img_w, img_h)
            if box is None:
                if raw.strip():
                    stats.boxes_skipped += 1
                continue
            cls, cx, cy, bw, bh = box
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stats.per_class[CLASS_NAMES[cls]] += 1

        (lbl_out / ann_path.name).write_text("\n".join(lines))
        target = img_out / img_path.name
        if not target.exists():
            if copy_images:
                shutil.copy2(img_path, target)
            else:
                target.symlink_to(img_path.resolve())

        stats.images += 1
        stats.boxes_kept += len(lines)

    return stats


def write_data_yaml(dst_root: Path, splits_present: list[str]) -> Path:
    """Write the Ultralytics dataset config. Validation uses the real val split, never train."""
    if "val" not in splits_present:
        raise RuntimeError("VisDrone val split not found; refusing to validate on the training set.")

    lines = [f"path: {dst_root.resolve().as_posix()}", "train: images/train", "val: images/val"]
    if "test" in splits_present:
        lines.append("test: images/test")
    lines.append(f"nc: {len(CLASS_NAMES)}")
    lines.append("names:")
    lines += [f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)]

    out = dst_root / "data.yaml"
    out.write_text("\n".join(lines) + "\n")
    return out


def convert_dataset(src_root: Path, dst_root: Path, copy_images: bool = True) -> dict[str, SplitStats]:
    """Convert every VisDrone split found under `src_root`. Returns stats per YOLO split."""
    src_root, dst_root = Path(src_root), Path(dst_root)
    results: dict[str, SplitStats] = {}
    for vis_name, yolo_name in SPLITS.items():
        split_dir = find_split_dir(src_root, vis_name)
        if split_dir is None:
            print(f"[skip] {vis_name}: not found under {src_root}")
            continue
        stats = convert_split(split_dir, dst_root, yolo_name, copy_images)
        results[yolo_name] = stats
        print(
            f"[ok] {yolo_name}: {stats.images} images, {stats.boxes_kept} boxes kept, "
            f"{stats.boxes_skipped} skipped, {stats.missing_images} missing images"
        )

    write_data_yaml(dst_root, list(results))
    return results
