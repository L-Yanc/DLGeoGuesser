# YOLOv8 Detector

This directory contains a wrapper for the [Ultralytics YOLOv8](https://docs.ultralytics.com/) model. It provides a standardised interface for training, validation, and prediction, tailored for the DLGeoGuesser project.

The `YOLOv8Detector` class is the core component, designed to be used both as a standalone tool via its command-line interface and as a building block in larger pipelines.

## Command-Line Interface

The `main.py` script provides a CLI to interact with the YOLOv8 model. All commands should be run from the project root.

### Training

You can train a model from scratch using a `.yaml` configuration or fine-tune an existing model from a `.pt` weights file. Training runs save all artifacts (checkpoints, logs, and metrics) to a unique directory inside `runs/yolo/`.

**Training Flags**

-   `--model <path>`: Path to a model config (e.g., `yolov8n.yaml`) for scratch training, or weights (e.g., `yolov8n.pt`) for fine-tuning.
-   `--data <path>`: Path to the dataset YAML file (e.g., `configs/vistas_yolo.yaml`).
-   `--epochs <int>`: Number of training epochs.
-   `--batch <int>`: Batch size.
-   `--workers <int>`: Number of data loading threads.
-   `--cache <ram|disk>`: Cache dataset for faster training.
-   `--resume`: Resume an interrupted training run.
-   `--name <run_name>`: A unique name for the training run.
-   `--device <name>`: Hardware to run on (`cpu`, `mps`, `0` for CUDA).

**Example Command**

```bash
python -m src.dl_geoguesser.vision.yolo_detector.main train \
  --model yolov8n.pt \
  --data configs/vistas_yolo.yaml \
  --epochs 50 \
  --batch 16 \
  --name yolo_run_1 \
  --device mps
```

### Validation

Validate a trained model on the `val` or `test` split of your dataset.

**Example Command**

```bash
python -m src.dl_geoguesser.vision.yolo_detector.main validate \
  --weights "runs/yolo/yolo_run_1/weights/best.pt" \
  --data configs/vistas_yolo.yaml \
  --split test
```

### Prediction

Run inference on a single image or a directory of images. The output is a structured dictionary of detected objects.

**Example Command**

```bash
python -m src.dl_geoguesser.vision.yolo_detector.main predict \
  --weights "runs/yolo/yolo_run_1/weights/best.pt" \
  --source "path/to/your/image.jpg" \
  --conf 0.4
```

### Visualising Performance

The Ultralytics library generates several plots and logs during training. This command provides quick access to the paths of these files.

**Example Command**

```bash
python -m src.dl_geoguesser.vision.yolo_detector.main visualize \
  --run "runs/yolo/yolo_run_1"
```

## Usage (as a Library Component)

The `YOLOv8Detector` class can be imported and used within other Python scripts.

```python
import cv2
from dl_geoguesser.vision.yolo_detector.model import YOLOv8Detector

# --- Configuration ---
WEIGHTS_PATH = "runs/yolo/yolo_run_1/weights/best.pt"
IMAGE_PATH = "path/to/an/image.jpg"

# --- Instantiate the Detector ---
detector = YOLOv8Detector(model_path=WEIGHTS_PATH)

# --- Run Prediction ---
# The image can be a path, a NumPy array, or a PIL Image
image = cv2.imread(IMAGE_PATH)
predictions = detector.predict(image, conf=0.4)

# --- Process Results ---
for class_name, instances in predictions.items():
    print(f"Found {len(instances)} instances of '{class_name}':")
    for instance in instances:
        print(f"  - BBox: {instance['bbox_crop']}, Confidence: {instance['confidence']:.2f}")

```

The returned dictionary is structured as follows:

```python
{
    "traffic_light": [
        {"confidence": 0.95, "bbox_crop": [100, 150, 120, 180], "scale": 600},
    ],
    "utility_pole": [
        {"confidence": 0.88, "bbox_crop": [250, 100, 260, 250], "scale": 1500}
    ]
}
```
