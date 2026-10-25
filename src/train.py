"""Train a YOLO detector, then optionally export it to TensorFlow SavedModel."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

from ultralytics import YOLO


def available_memory_gb() -> float:
    """Return currently available physical memory without another dependency."""
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullAvailPhys / 1024**3
    pages = os.sysconf("SC_AVPHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return pages * page_size / 1024**3


def recommended_workers() -> int:
    """Choose enough parallel loading without consuming most system RAM."""
    cpu_limit = max(1, (os.cpu_count() or 2) // 2)
    # Reserve 4 GB for Windows, Python, the IDE, and GPU driver; allow ~2 GB/worker.
    memory_limit = max(1, int(max(0, available_memory_gb() - 4) // 2))
    return min(4, cpu_limit, memory_limit)


def worker_count(value: str) -> int:
    if value.lower() == "auto":
        return recommended_workers()
    count = int(value)
    if count < 0:
        raise argparse.ArgumentTypeError("workers must be a non-negative integer or 'auto'")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Indian currency YOLO OBB model.")
    parser.add_argument("--data", type=Path, default=Path("data/indian_currency.yaml"))
    parser.add_argument("--model", default="yolo11n-obb.pt", help="Base model; use yolo11s-obb.pt for more accuracy.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1, help="-1 automatically chooses a batch size.")
    parser.add_argument("--workers", type=worker_count, default="auto", help="Data-loader workers, or 'auto' (recommended).")
    parser.add_argument("--device", default=None, help="e.g. 0 for first CUDA GPU, cpu for CPU.")
    parser.add_argument("--export-tf", action="store_true", help="Export best model as TensorFlow SavedModel after training.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")
    print(f"Using {args.workers} data-loader worker(s) (available RAM: {available_memory_gb():.1f} GB).")
    model = YOLO(args.model)
    results = model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, device=args.device,
    )
    best_weights = results.save_dir / "weights" / "best.pt"
    print(f"Training complete. Best weights: {best_weights}")
    if args.export_tf:
        print("TensorFlow SavedModel:", YOLO(str(best_weights)).export(format="saved_model"))
