# CLIP Landscape Model Scripts

This directory contains scripts for training and evaluating the CLIP-based landscape classification model.

**Note:** All scripts in this directory should be run from the project root.

## Execution Order

### 1. Train the Model

This script trains the classification head on top of the frozen CLIP backbone. It uses the settings from `configs/clip_landscape.yaml` and saves the best model to `models/clip_landscape/classifier_head/best.pt`.

**Command:**
```bash
python scripts/clip_landscape/run_clip_training.py
```

### 2. Extract Embeddings

This script uses the trained model's backbone to extract image embeddings for the train, validation, and test sets. These are required for the k-NN evaluation. The embeddings are saved in `models/clip_landscape/knn_baseline/`.

**Command:**
```bash
python scripts/clip_landscape/run_clip_embeddings.py
```

### 3. Evaluate with k-NN

This script loads the pre-computed embeddings and runs a k-Nearest Neighbors (k-NN) classification. This provides a baseline evaluation of the feature quality. Results are printed to the console and saved in `models/clip_landscape/knn_baseline/knn_results.txt`.

**Command:**
```bash
python scripts/clip_landscape/run_clip_knn_eval.py
```

### 4. Visualise with Grad-CAM (Optional)

The `run_clip_gradcam.py` script is currently a placeholder. When implemented, it will be used to generate Grad-CAM visualisations to see what parts of an image the model is focusing on.
