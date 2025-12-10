# YOLOv8 Object Detection Scripts

This directory contains scripts for preparing the Mapillary Vistas dataset, and for training and visualising a YOLOv8 object detector.

## Execution Order

The scripts should be run in the following order:

### 1. Download and Organise Data

This script downloads and unzips the necessary data from the URLs specified in `yolo_data_urls.txt`. It places the data in the `data/raw/` directory.

**Command:**
```bash
bash scripts/yolo/download_and_organise_vistas.sh
```

### 2. Prepare Vistas Dataset for YOLO

This script processes the raw Mapillary Vistas data into the format required for YOLOv8 training. It creates `train/`, `val/`, and `test/` splits with `images/` and `labels/` subdirectories, and generates a `data.yaml` file for YOLO.

**Command:**
You can process a full split (e.g., 'training') or create smaller sub-splits.

To process the full 'training' split:
```bash
python scripts/yolo/prepare_vistas_yolo.py --split training
```

To create a smaller, custom split for quick testing:
```bash
python scripts/yolo/prepare_vistas_yolo.py --split training --train-size 1000 --val-size 200 --test-size 100
```

### 3. Train the Model

Training is handled by the project's YOLOv8 wrapper script. This script uses the `data.yaml` file created by the preparation script and fine-tunes a pre-trained YOLO model.

**Command:**
```bash
python -m src.dl_geoguesser.vision.yolo_detector.main train \
--model yolov8n.pt \
--data configs/vistas_yolo.yaml \
--epochs 100 \
--batch 64 \
--workers 8 \
--device 0 \
--name yolo_run
```

### 4. Predict and Visualise

After training, this script can be used to run predictions on images and visualise the results. It will automatically find the latest trained model, or you can specify a weights file.

**Command:**
```bash
# Visualise predictions on the validation set
python scripts/yolo/predict_and_visualise.py --source data/processed/vistas_yolo/val/images

# Visualise prediction on a single image
python scripts/yolo/predict_and_visualise.py --source /path/to/your/image.jpg
```
