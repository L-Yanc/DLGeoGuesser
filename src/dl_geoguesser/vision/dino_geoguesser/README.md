# DINO Geoguesser Model

This directory contains the implementation of a DINOv2-based country classifier. The model is designed to take cropped images of objects (detected by a model like YOLO) and predict the country of origin.

## Training

The model is trained via the command-line interface in `main.py`. The training pipeline is highly flexible and supports two main modes:

1.  **On-the-Fly Mode (Default)**: Loads images and computes DINO embeddings in real-time during the training loop. This is simpler but much slower. It allows for image-based data augmentation.
2.  **Pre-computed Mode**: Performs a one-time pass to compute all DINO embeddings for the dataset and saves them to the run folder. Subsequent training epochs load these saved embeddings, resulting in extremely fast training of the classifier head.

All training artifacts, including checkpoints and any pre-computed embeddings, are saved to a run-specific directory inside `runs/dino/`.

### Training Flags

-   `--name <run_name>`: A unique name for the training run.
-   `--device <name>`: The hardware to run on (`cpu`, `mps`, `cuda`).
-   `--precompute-embeddings`: If present, activates the pre-computed training mode.
-   `--augmentations <type>`: Chooses the augmentation strategy.
    -   `none`: No augmentations.
    -   `image`: (On-the-fly mode only) Applies image-based augmentations like random cropping and color jitter.
    -   `embedding`: Adds Gaussian noise to embeddings (works in both modes).
    -   `both`: Applies both `image` and `embedding` augmentations (on-the-fly mode only).
-   `--resume`: If present, resumes training from the `last.pt` checkpoint within the run's directory (`--name`).
-   `--weights <path>`: Starts a new run but initializes the model's weights from a specified `.pt` file for fine-tuning.

### Example Commands

All commands should be run from the project root.

**Scenario 1: Fast training with pre-computed embeddings and noise augmentation.**
This is the recommended approach for quick experiments.
```bash
python -m src.dl_geoguesser.vision.dino_geoguesser.main train \
  --name dino_precomputed_run \
  --device mps \
  --precompute-embeddings \
  --augmentations embedding
```

**Scenario 2: Slower training with on-the-fly image processing and augmentation.**
Use this if image-based augmentations are critical and you have time.
```bash
python -m src.dl_geoguesser.vision.dino_geoguesser.main train \
  --name dino_image_aug_run \
  --device mps \
  --augmentations image
```

**Scenario 3: Resuming an interrupted run.**
This will automatically use pre-computed embeddings if they were created by the original run.
```bash
python -m src.dl_geoguesser.vision.dino_geoguesser.main train \
  --name dino_precomputed_run \
  --resume \
  --device mps
```

**Scenario 4: Fine-tuning from existing weights.**
Starts a new training session using weights from a previous `best.pt` file.
```bash
python -m src.dl_geoguesser.vision.dino_geoguesser.main train \
  --name dino_finetune_run \
  --weights "runs/dino/dino_precomputed_run/best.pt" \
  --device mps
```

## Detailed Evaluation

To diagnose model performance and investigate issues like class imbalance, you can use the `evaluate` command. This mode will run your trained model on a dataset split (`val` or `test`) and provide detailed, per-class metrics.

This will:
1.  Print a **Classification Report** to the console, showing precision, recall, and F1-score for every country.
2.  Generate and save a **Confusion Matrix** heatmap to the model's run directory (e.g., `runs/dino/your_run_name/confusion_matrix_val.png`).

**Note:** This feature requires `matplotlib` and `seaborn`. If you don't have them installed, run:
`pip install matplotlib seaborn`

### Command

```bash
python -m src.dl_geoguesser.vision.dino_geoguesser.main evaluate \
  --weights "runs/dino/your_run_name/best.pt" \
  --split val \
  --device mps
```

## Usage (as a Library Component)

The `DinoGeoguesser` class is designed to be a component in a larger prediction pipeline. You can specify the device for inference when you instantiate the class.

Below is an example of how to use it in a script:

```python
from PIL import Image
from dl_geoguesser.vision.yolo_detector import YOLOv8Detector
from dl_geoguesser.vision.dino_geoguesser import DinoGeoguesser

# --- Configuration ---
# Select the device for inference ('mps', 'cuda', or 'cpu')
DEVICE = "mps" 

# 1. Define paths to the trained models
dino_weights_path = "runs/dino/dino_country_classifier/best.pt"
yolo_weights_path = "path/to/your/yolo_weights.pt" # e.g., runs/detect/train/weights/best.pt
image_path = "path/to/an/image.jpg"

# 2. Instantiate the models on the desired device
dino_classifier = DinoGeoguesser(weights_path=dino_weights_path, device=DEVICE)
# Note: The YOLOv8Detector can also be assigned a device on init
yolo_detector = YOLOv8Detector(model_path=yolo_weights_path, device=DEVICE)

# 3. Load the image and get object detections
image = Image.open(image_path).convert("RGB")
detections = yolo_detector.predict(image)

# 4. Get country scores from the DINO classifier using the crops
if detections:
    country_scores = dino_classifier.predict_from_crops(image, detections)
    
    # Sort and print the top 5 predictions
    sorted_scores = sorted(country_scores.items(), key=lambda item: item[1], reverse=True)
    
    print("--- Top 5 Country Predictions ---")
    for country, score in sorted_scores[:5]:
        print(f"{country}: {score:.3f}")
    print("---")
else:
    print("No objects detected in the image.")

```
