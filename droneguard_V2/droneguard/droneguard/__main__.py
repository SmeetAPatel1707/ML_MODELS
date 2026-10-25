"""Command-line entry point.

Examples:
    python -m droneguard prepare --src /data/visdrone --dst datasets/visdrone-yolo
    python -m droneguard train --data datasets/visdrone-yolo/data.yaml --model yolo11s.pt --imgsz 1024
    python -m droneguard evaluate --weights runs/detect/train/weights/best.pt --data datasets/visdrone-yolo/data.yaml
    python -m droneguard predict --weights runs/detect/train/weights/best.pt --source samples/
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def cmd_prepare(args: argparse.Namespace) -> None:
    from .visdrone import convert_dataset

    stats = convert_dataset(Path(args.src), Path(args.dst), copy_images=not args.symlink)
    summary = {split: vars(s) for split, s in stats.items()}
    out = Path(args.dst) / "conversion_stats.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {Path(args.dst) / 'data.yaml'} and {out}")


def cmd_train(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
        seed=args.seed,
        deterministic=True,
        plots=True,
    )
    print(f"Best weights: {Path(model.trainer.save_dir) / 'weights' / 'best.pt'}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, split=args.split, plots=True)

    names = model.names
    rows = []
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        rows.append({"class": names[int(cls_idx)], "precision": p, "recall": r, "mAP50": ap50, "mAP50-95": ap})
    rows.append({
        "class": "all",
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})

    print(f"\n{'class':<18}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for row in rows:
        print(f"{row['class']:<18}{row['precision']:>8.3f}{row['recall']:>8.3f}{row['mAP50']:>8.3f}{row['mAP50-95']:>10.3f}")
    print(f"\nSaved per-class metrics to {out}")


def cmd_predict(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        save=True,
        project=args.project,
        name=args.name,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="droneguard", description="Aerial intruder detection on VisDrone.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("prepare", help="Convert VisDrone2019-DET to YOLO format.")
    s.add_argument("--src", required=True, help="Folder containing the VisDrone2019-DET-* split folders.")
    s.add_argument("--dst", default="datasets/visdrone-yolo")
    s.add_argument("--symlink", action="store_true", help="Symlink images instead of copying (saves disk).")
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("train", help="Train a YOLO model.")
    s.add_argument("--data", default="datasets/visdrone-yolo/data.yaml")
    s.add_argument("--model", default="yolo11s.pt", help="Pretrained checkpoint, e.g. yolov8n.pt (baseline) or yolo11s.pt.")
    s.add_argument("--imgsz", type=int, default=1024)
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--batch", type=int, default=-1, help="-1 = auto batch size for available GPU memory.")
    s.add_argument("--patience", type=int, default=20)
    s.add_argument("--project", default="runs/detect")
    s.add_argument("--name", default="train")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("evaluate", help="Per-class metrics on the val (or test) split.")
    s.add_argument("--weights", required=True)
    s.add_argument("--data", default="datasets/visdrone-yolo/data.yaml")
    s.add_argument("--imgsz", type=int, default=1024)
    s.add_argument("--split", default="val", choices=["val", "test"])
    s.add_argument("--out", default="results/metrics_val.csv")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("predict", help="Run detection on images or a video.")
    s.add_argument("--weights", required=True, help="Use your trained best.pt, not a generic COCO checkpoint.")
    s.add_argument("--source", required=True)
    s.add_argument("--imgsz", type=int, default=1024)
    s.add_argument("--conf", type=float, default=0.25)
    s.add_argument("--project", default="runs/predict")
    s.add_argument("--name", default="demo")
    s.set_defaults(func=cmd_predict)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
