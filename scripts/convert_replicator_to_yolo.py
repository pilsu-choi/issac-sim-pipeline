"""Convert Isaac Sim Replicator BasicWriter output to YOLO detection format."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import yaml


def log(message: str) -> None:
    print(f"[yolo-converter] {message}", flush=True)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Dataset generation config path.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Replicator BasicWriter output directory. Defaults to dataset_root in config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/yolo/humanoid_egocentric_perception"),
        help="YOLO dataset output directory.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory before conversion.",
    )
    parser.add_argument(
        "--no-enhance-rgb",
        action="store_true",
        help="Copy RGB images without auto exposure correction.",
    )
    return parser.parse_args()


def frame_id_from_rgb_path(path: Path) -> str:
    return path.stem.replace("rgb_", "")


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def to_yolo_line(
    row: np.void,
    class_id: int,
    image_width: int,
    image_height: int,
) -> str | None:
    x_min = clamp(float(row["x_min"]), 0.0, image_width - 1.0)
    y_min = clamp(float(row["y_min"]), 0.0, image_height - 1.0)
    x_max = clamp(float(row["x_max"]), 0.0, image_width - 1.0)
    y_max = clamp(float(row["y_max"]), 0.0, image_height - 1.0)

    box_width = x_max - x_min
    box_height = y_max - y_min
    if box_width <= 1 or box_height <= 1:
        return None

    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    normalized_width = box_width / image_width
    normalized_height = box_height / image_height

    return (
        f"{class_id} "
        f"{x_center:.6f} {y_center:.6f} "
        f"{normalized_width:.6f} {normalized_height:.6f}"
    )


def write_dataset_yaml(output_dir: Path, classes: list[str]) -> None:
    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: class_name for index, class_name in enumerate(classes)},
    }
    with (output_dir / "dataset.yaml").open("w", encoding="utf-8") as yaml_file:
        yaml.safe_dump(dataset_yaml, yaml_file, sort_keys=False, allow_unicode=True)


def save_rgb_image(source_path: Path, target_path: Path, enhance_rgb: bool) -> None:
    if not enhance_rgb:
        shutil.copy2(source_path, target_path)
        return

    image = Image.open(source_path).convert("RGB")
    pixels = np.asarray(image).astype(np.float32)
    max_value = float(pixels.max())

    if max_value > 0:
        # Isaac Sim RGB can be slightly under-exposed depending on renderer state.
        # Keep this conservative; if the source is only 0/1 colors, the generation
        # material scale should be fixed instead of amplifying it here.
        if max_value < 32:
            pixels *= min(96.0 / max_value, 8.0)
        mean_value = float(pixels.mean())
        if 0 < mean_value < 70:
            pixels *= min(80.0 / mean_value, 2.0)

    enhanced_image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    enhanced_image.save(target_path)


def convert_frame(
    image_path: Path,
    source_dir: Path,
    output_dir: Path,
    split: str,
    class_to_id: dict[str, int],
    image_width: int,
    image_height: int,
    enhance_rgb: bool,
) -> tuple[int, int]:
    frame_id = frame_id_from_rgb_path(image_path)
    labels_path = source_dir / f"bounding_box_2d_tight_labels_{frame_id}.json"
    boxes_path = source_dir / f"bounding_box_2d_tight_{frame_id}.npy"

    semantic_labels = load_json(labels_path)
    boxes = np.load(boxes_path, allow_pickle=True)

    yolo_lines = []
    for row in boxes:
        semantic_id = str(int(row["semanticId"]))
        class_name = semantic_labels.get(semantic_id, {}).get("class")
        if class_name not in class_to_id:
            continue

        yolo_line = to_yolo_line(
            row=row,
            class_id=class_to_id[class_name],
            image_width=image_width,
            image_height=image_height,
        )
        if yolo_line:
            yolo_lines.append(yolo_line)

    target_image_path = output_dir / "images" / split / image_path.name
    target_label_path = output_dir / "labels" / split / f"{image_path.stem}.txt"
    save_rgb_image(image_path, target_image_path, enhance_rgb=enhance_rgb)
    target_label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
    return 1, len(yolo_lines)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    source_dir = args.source_dir or Path(config["dataset_root"])
    output_dir = args.output_dir
    classes = list(config["classes"])
    class_to_id = {class_name: index for index, class_name in enumerate(classes)}
    image_width = int(config["scene"]["image_width"])
    image_height = int(config["scene"]["image_height"])
    enhance_rgb = not args.no_enhance_rgb

    image_paths = sorted(source_dir.glob("rgb_*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No rgb_*.png files found in {source_dir}")

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    random.shuffle(image_paths)

    val_count = max(1, int(len(image_paths) * args.val_ratio))
    val_images = set(image_paths[:val_count])

    split_counts = {"train": 0, "val": 0}
    label_counts = {"train": 0, "val": 0}

    log(
        f"Converting {len(image_paths)} images from {source_dir} to {output_dir}. "
        f"enhance_rgb={enhance_rgb}"
    )
    for image_path in image_paths:
        split = "val" if image_path in val_images else "train"
        image_count, label_count = convert_frame(
            image_path=image_path,
            source_dir=source_dir,
            output_dir=output_dir,
            split=split,
            class_to_id=class_to_id,
            image_width=image_width,
            image_height=image_height,
            enhance_rgb=enhance_rgb,
        )
        split_counts[split] += image_count
        label_counts[split] += label_count

    write_dataset_yaml(output_dir, classes)

    log(f"Train images: {split_counts['train']}, labels: {label_counts['train']}")
    log(f"Val images: {split_counts['val']}, labels: {label_counts['val']}")
    log(f"YOLO dataset yaml: {output_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
