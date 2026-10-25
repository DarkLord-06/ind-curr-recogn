"""Clip invalid YOLO OBB corner coordinates to the image boundary.

Prefer re-exporting corrected annotations from Roboflow. This utility is a
practical fallback for OBB points that extend a small distance beyond an image.
It defaults to a dry run and changes files only when --apply is supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def sanitize_label(path: Path, apply: bool) -> int:
    changed = 0
    output: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if not values:
            continue
        if len(values) != 9:
            raise ValueError(f"Expected 9 values in {path}, got {len(values)}")
        class_id = values[0]
        points = [float(value) for value in values[1:]]
        clipped = [min(1.0, max(0.0, value)) for value in points]
        changed += points != clipped
        output.append(" ".join([class_id, *(f"{value:.8f}" for value in clipped)]))
    if changed and apply:
        path.write_text("\n".join(output) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair out-of-range YOLO OBB coordinates.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--apply", action="store_true", help="Write clipped labels. Without this flag, only report changes.")
    args = parser.parse_args()
    changed_files = 0
    changed_annotations = 0
    for label in args.dataset.glob("*/labels/*.txt"):
        changed = sanitize_label(label, args.apply)
        changed_files += bool(changed)
        changed_annotations += changed
    action = "Updated" if args.apply else "Would update"
    print(f"{action} {changed_annotations} annotations in {changed_files} label files.")
    if not args.apply and changed_annotations:
        print("Review the dataset if possible, then run again with --apply to clip coordinates to 0..1.")


if __name__ == "__main__":
    main()
