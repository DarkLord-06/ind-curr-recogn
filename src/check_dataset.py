"""Lightweight validation for a YOLO OBB-format Indian currency dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main(config_path: Path) -> int:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = (config_path.parent / config["path"]).resolve()
    names = config["names"]
    class_count = len(names)
    errors: list[str] = []
    histogram: Counter[int] = Counter()
    for split in ("train", "val", "test"):
        if split not in config:
            continue
        images_dir = (root / config[split]).resolve()
        # Roboflow stores each split as split/images and split/labels.
        labels_dir = images_dir.parent / "labels"
        images = [path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES] if images_dir.exists() else []
        print(f"{split}: {len(images)} images ({images_dir})")
        if not images:
            errors.append(f"No images found for '{split}'")
        for image in images:
            label = labels_dir / f"{image.stem}.txt"
            if not label.exists():
                errors.append(f"Missing label: {label}")
                continue
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
                try:
                    values = [float(value) for value in line.split()]
                    class_id = int(values[0])
                    if (len(values) != 9 or class_id < 0 or class_id >= class_count
                            or any(value < 0 or value > 1 for value in values[1:])):
                        raise ValueError
                    histogram[class_id] += 1
                except (ValueError, IndexError):
                    errors.append(f"Invalid label: {label}:{line_number}")
    print("Class distribution:")
    for class_id, name in (names.items() if isinstance(names, dict) else enumerate(names)):
        print(f"  {name}: {histogram[int(class_id)]}")
    if errors:
        print("\nErrors (first 20):")
        print("\n".join(f"- {error}" for error in errors[:20]))
        print(f"Dataset check failed with {len(errors)} issue(s).")
        return 1
    print("Dataset check passed.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/indian_currency.yaml"))
    raise SystemExit(main(parser.parse_args().data))
