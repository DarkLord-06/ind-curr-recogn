# Indian Currency Recognition

Detects every visible Indian banknote from a webcam, displays its denomination, counts notes by denomination, and calculates the total in INR.

The live application uses OpenCV for the camera and display, and YOLO OBB (oriented bounding boxes) for multi-note detection. OBB is the correct YOLO task for this dataset because annotations contain the four corners of each rotated note. TensorFlow is used for deployment: pass `--export-tf` when training to export the trained YOLO model to TensorFlow SavedModel. The `.pt` model is recommended for the supplied desktop camera application.

## Supported labels

Your supplied configuration contains: `INR_2000`, `INR_10`, `INR_100`, `INR_20`, `INR_200`, `INR_5`, `INR_50`, and `INR_500`. The ₹5 class has only one annotated example and only in the test split, so it cannot be learned; add it to training/validation or remove/relabel that annotation. Class IDs must match the IDs in label files.

## Dataset layout

The dataset currently lives in `dataset/`. Its absolute path is set in the configuration to prevent Ultralytics resolving it from the shell's working directory. If you move the project, update the `path:` value in `data/indian_currency.yaml`.

```
dataset/
  train/images/, train/labels/
  valid/images/, valid/labels/
  test/images/, test/labels/
```

Each label line is `class_id x1 y1 x2 y2 x3 y3 x4 y4`: the four normalised corner points of a note. Multiple lines in one label file represent multiple notes in the same image.

### Important dataset repair before retraining

The supplied export contains out-of-range OBB coordinates, which Ultralytics may ignore as corrupt labels. The best fix is to correct/re-export the annotations from Roboflow. As a fallback, inspect the dry-run result and then clip out-of-frame vertices to the image boundary:

```powershell
python src/sanitize_obb_labels.py
python src/sanitize_obb_labels.py --apply
python src/check_dataset.py
```

Also add empty camera/background images with matching empty `.txt` files to `train/labels` and `valid/labels`. These negative examples are necessary for the model to learn that an empty frame contains no currency.

## Setup and run

Your machine has an NVIDIA RTX 4060, but the current virtual environment has a CPU-only PyTorch build. Install the CUDA-enabled PyTorch wheel before training. Your NVIDIA driver supports CUDA 13.0 wheels:

```powershell
.\.venv\Scripts\Activate.ps1
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The verification command must print `True` and your NVIDIA GPU name. PyTorch wheels bundle their own CUDA runtime; the installed NVIDIA driver must be sufficiently new, but a separate local CUDA Toolkit version does not have to exactly match the wheel.

Then install the project dependencies and train on GPU 0:

```powershell
pip install -r requirements.txt
python src/check_dataset.py
python src/train.py --epochs 100 --imgsz 640 --device 0 --workers auto
```

Copy `runs/obb/train/weights/best.pt` to `models/best.pt`, then start the camera:

```powershell
python src/camera.py --weights models/best.pt --camera 0
```

Use `q` or `Esc` to close the camera window. If the wrong camera opens, try `--camera 1`. For TensorFlow deployment, add `--export-tf` to training. Increase `--imgsz` to 960 or use `--model yolo11s.pt` when small or overlapping notes need more accuracy.

While the camera is running, press `S` to save three files in `captures/`: the raw camera image, the annotated analysis image, and a JSON report containing the detected denomination counts and INR total. It speaks that saved analysis only after `S` is pressed; add `--no-speech` to disable it.

The camera starts at a 0.70 confidence threshold to reduce false positives. If it still detects a ₹2000 note with an empty camera scene, try `--confidence 0.80`. The durable fix is to add several empty/background images with matching empty label files to the **training** and **validation** splits, then retrain. This teaches the model that many scenes contain no note at all.

`--workers auto` selects a conservative worker count from available RAM and CPU cores (maximum four on Windows). Each data-loader worker is a separate Python process and may substantially increase RAM use. Use `nvidia-smi` to monitor the real CUDA utilisation; Windows Task Manager's default 3D graph does not accurately show CUDA training work.

## Dataset quality checklist

- Include fronts, backs, rotations, folds, partial occlusion, varied lighting, and multi-note images.
- Label every visible note; otherwise the model learns to ignore unlabelled notes.
- Keep classes balanced and reserve the test set for final evaluation.
- Do not include the withdrawn ₹2000 denomination unless it is intentionally part of your use case.
