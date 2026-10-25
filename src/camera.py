"""Run real-time Indian banknote detection and denomination counting."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    from .speech import Speaker
except ImportError:  # Allows `python src/camera.py` as well as `python -m src.camera`.
    from speech import Speaker


def denomination(label: str) -> int:
    """Extract the amount from a label such as INR_500."""
    match = re.search(r"\d+", label)
    return int(match.group()) if match else 0


def draw_summary(frame, counts: Counter[str], fps: float) -> None:
    lines = [f"{name}: {count}" for name, count in sorted(counts.items(), key=lambda item: denomination(item[0]))]
    total = sum(denomination(name) * count for name, count in counts.items())
    panel_height = max(80, 38 + 26 * len(lines))
    cv2.rectangle(frame, (8, 8), (270, panel_height), (20, 20, 20), -1)
    cv2.putText(frame, f"Notes: {sum(counts.values())} | Total: INR {total}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (18, 60 + 25 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"{fps:.1f} FPS | s: save analysis | q: quit", (18, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)


def summary(counts: Counter[str]) -> dict[str, object]:
    ordered_counts = dict(sorted(counts.items(), key=lambda item: denomination(item[0])))
    return {
        "notes": sum(counts.values()),
        "denominations": ordered_counts,
        "total_inr": sum(denomination(name) * count for name, count in counts.items()),
    }


def spoken_summary(counts: Counter[str]) -> str:
    result = summary(counts)
    if not result["notes"]:
        return "No currency detected"
    denominations = ", ".join(f"{count} note{'s' if count != 1 else ''} of {denomination(name)} rupees" for name, count in result["denominations"].items())
    return f"Detected {denominations}. Total {result['total_inr']} rupees"


def save_snapshot(raw_frame, annotated_frame, counts: Counter[str], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}"
    base_path = output_dir / f"currency_{timestamp}"
    cv2.imwrite(str(base_path.with_name(f"{base_path.name}_raw.jpg")), raw_frame)
    cv2.imwrite(str(base_path.with_name(f"{base_path.name}_analysis.jpg")), annotated_frame)
    base_path.with_suffix(".json").write_text(json.dumps(summary(counts), indent=2), encoding="utf-8")
    return base_path


def run(weights: Path, camera_index: int, confidence: float, image_size: int, snapshot_dir: Path, speech_enabled: bool) -> None:
    model = YOLO(str(weights))
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}. Try --camera 1 or check camera permissions.")
    last_time = time.perf_counter()
    speaker = Speaker(enabled=speech_enabled)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera frame could not be read.")
            result = model(frame, conf=confidence, imgsz=image_size, verbose=False)[0]
            annotated = result.plot()
            counts: Counter[str] = Counter()
            # This project uses YOLO OBB: notes are represented by rotated boxes.
            if result.obb is not None:
                for class_id in result.obb.cls.int().tolist():
                    counts[str(result.names[class_id])] += 1
            now = time.perf_counter()
            fps = 1 / max(now - last_time, 1e-6)
            last_time = now
            draw_summary(annotated, counts, fps)
            cv2.imshow("Indian Currency Recognition", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")):
                saved = save_snapshot(frame, annotated, counts, snapshot_dir)
                print(f"Saved analysis: {saved}")
                speaker.say(f"Analysis saved. {spoken_summary(counts)}")
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        speaker.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognise and count Indian banknotes from a camera.")
    parser.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.70, help="Minimum detection confidence; raise it to reduce false positives.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--snapshot-dir", type=Path, default=Path("captures"))
    parser.add_argument("--no-speech", action="store_true", help="Disable spoken currency results.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if not arguments.weights.is_file():
        raise FileNotFoundError(f"Model not found: {arguments.weights}. Train first or pass --weights path/to/best.pt")
    run(arguments.weights, arguments.camera, arguments.confidence, arguments.imgsz, arguments.snapshot_dir, not arguments.no_speech)
