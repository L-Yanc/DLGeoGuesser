# CLIP Vibe Scripts

This directory contains scripts for creating a dataset, and for running predictions and visualisations with the `ClipVibe` model.

## Script Usage

### 1. Scrape Dataset

This script scrapes images from DuckDuckGo based on a predefined set of search terms. It downloads the images into a specified directory and creates a `metadata.csv` file to be used for training.

**Command:**
```bash
python scripts/clip_vibe/scrape_ddg.py --num_images 100 --output_dir data/processed/clip_vibe_dataset
```

### 2. Regenerate Metadata

This is a utility script to regenerate the `metadata.csv` file if it gets lost or corrupted. It scans the image filenames in a directory to determine the class for each image. Note that the original `search_term` information will be lost.

**Command:**
```bash
python scripts/clip_vibe/regenerate_metadata.py --images_dir data/processed/clip_vibe_dataset
```

### 3. Predict from Images

This script runs batch predictions on a directory of images using a trained `ClipVibe` model. It prints the top-K predicted classes and their scores for each image to the console.

**Command:**
```bash
python scripts/clip_vibe/predict_from_images.py \
  --weights "runs/clip_vibe/your_run_name/best.pt" \
  --images_dir "test_images" \
  --top_k 3
```

### 4. Predict and Visualise Explanations

This script runs predictions on a directory of images and generates explanation heatmaps for the top predicted class of each image. It saves the visualisations as new images. You can choose between two explanation methods: `attention` (fast) and `integrated_gradients` (rigorous).

**Command:**
```bash
# Generate explanations using the default Integrated Gradients method
python scripts/clip_vibe/predict_and_visualise.py \
  --weights "runs/clip_vibe/your_run_name/best.pt" \
  --source "test_images" \
  --output_dir "test_images_explained"

# Use the faster 'attention' method instead
python scripts/clip_vibe/predict_and_visualise.py \
  --weights "runs/clip_vibe/your_run_name/best.pt" \
  --source "test_images" \
  --output_dir "test_images_explained" \
  --method "attention"
```
