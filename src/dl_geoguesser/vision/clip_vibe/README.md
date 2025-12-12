# CLIP Vibe Model

This directory contains the implementation of a CLIP-based classifier. The model is designed to take images and predict which class they belong to (e.g., "forest", "beach", "city").

## Training

The model is trained via the command-line interface in `main.py`. The training pipeline is highly flexible and supports two main modes:

1.  **On-the-Fly Mode (Default)**: Loads images and computes CLIP embeddings in real-time during the training loop. This is simpler but much slower. It allows for image-based data augmentation.
2.  **Pre-computed Mode**: Performs a one-time pass to compute all CLIP embeddings for the dataset and saves them to the run folder. Subsequent training epochs load these saved embeddings, resulting in extremely fast training of the classifier head.

All training artifacts, including checkpoints and any pre-computed embeddings, are saved to a run-specific directory inside `runs/clip_vibe/`.

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
python -m src.dl_geoguesser.vision.clip_vibe.main train \
  --name clip_vibe_precomputed_run \
  --device mps \
  --precompute-embeddings \
  --augmentations embedding
```

**Scenario 2: Slower training with on-the-fly image processing and augmentation.**
Use this if image-based augmentations are critical and you have time.
```bash
python -m src.dl_geoguesser.vision.clip_vibe.main train \
  --name clip_vibe_image_aug_run \
  --device mps \
  --augmentations image
```

**Scenario 3: Resuming an interrupted run.**
This will automatically use pre-computed embeddings if they were created by the original run.
```bash
python -m src.dl_geoguesser.vision.clip_vibe.main train \
  --name clip_vibe_precomputed_run \
  --resume \
  --device mps
```

**Scenario 4: Fine-tuning from existing weights.**
Starts a new training session using weights from a previous `best.pt` file.
```bash
python -m src.dl_geoguesser.vision.clip_vibe.main train \
  --name clip_vibe_finetune_run \
  --weights "runs/clip_vibe/clip_vibe_precomputed_run/best.pt" \
  --device mps
```

## Class Imbalance

The dataset has a significant class imbalance (some classes have many more images than others). To combat this, the training pipeline includes two features:

1.  **Stratified Splitting**: By default, the dataset is split into train, validation, and test sets in a stratified manner. This can be disabled for ablation studies.
2.  **Weighted Sampling**: You can enable weighted sampling for the training set to give samples from rarer classes a higher probability of being selected in each batch. This helps the model see a more balanced distribution of classes during training.

To control these features, set the following flags in `configs/clip_vibe.yaml`:
```yaml
training:
  use_weighted_sampling: true
  use_stratification: true
```

## Detailed Evaluation

To diagnose model performance and investigate issues like class imbalance, you can use the `evaluate` command. This mode will run your trained model on a dataset split (`val` or `test`) and provide detailed, per-class metrics.

This will:
1.  Print a **Classification Report** to the console, showing precision, recall, and F1-score for every class.
2.  Generate and save a **Confusion Matrix** heatmap to the model's run directory (e.g., `runs/clip_vibe/your_run_name/confusion_matrix_val.png`).

**Note:** This feature requires `matplotlib` and `seaborn`. If you don't have them installed, run:
`pip install matplotlib seaborn`

### Command

```bash
python -m src.dl_geoguesser.vision.clip_vibe.main evaluate \
  --weights "runs/clip_vibe/your_run_name/best.pt" \
  --split val \
  --device mps
```

## Usage (as a Library Component)

The `ClipVibe` class is designed to be a component in a larger prediction pipeline. You can specify the device for inference when you instantiate the class.

Below is an example of how to use it in a script:

```python
from PIL import Image
from dl_geoguesser.vision.clip_vibe import ClipVibe

# --- Configuration ---
DEVICE = "mps" 
clip_vibe_weights_path = "runs/clip_vibe/your_run_name/best.pt"
image_path = "path/to/an/image.jpg"

# --- Instantiate the Model ---
clip_vibe_classifier = ClipVibe(weights_path=clip_vibe_weights_path, device=DEVICE)

# --- Get Class Scores ---
image = Image.open(image_path).convert("RGB")
class_scores = clip_vibe_classifier.predict(image)
    
# --- Print Top 5 Predictions ---
if class_scores:
    sorted_scores = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
    print("--- Top 5 Class Predictions ---")
    for class_name, score in sorted_scores[:5]:
        print(f"{class_name}: {score:.3f}")
    print("---")
else:
    print("Could not make a prediction.")
```

## Explainability

To understand why the model makes a certain prediction, you can use the `explain` command to generate a visual heatmap overlay on an image. This helps to highlight the regions of the image that were most influential for the model's decision.

There are two available methods for generating explanations:

1.  **`attention`**: This is a fast method that visualises the gradient-weighted attention from the CLIP model. It's useful for a quick look at what the model might be focusing on, but it is less rigorous because the underlying attention weights were not trained for the classification task.
2.  **`integrated_gradients`**: This is a more rigorous, but slower, method that computes pixel-level attributions. It provides a more faithful explanation of which pixels caused the prediction. This method works correctly even if the CLIP backbone is frozen.

### Command

```bash
python -m src.dl_geoguesser.vision.clip_vibe.main explain \
  --weights "runs/clip_vibe/your_run_name/best.pt" \
  --image "path/to/your/image.jpg" \
  --class_name "forest" \
  --output "explanation.jpg" \
  --method "integrated_gradients" \
  --device mps
```

This will save an image named `explanation.jpg` with the heatmap overlaid on the original image.
