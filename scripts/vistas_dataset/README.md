# Vistas Dataset Scripts

This directory contains scripts for visualising the processed Mapillary Vistas dataset.

## Usage

### 1. Visualise YOLO Annotations

This script allows you to inspect the bounding box annotations that were created by the `prepare_vistas_yolo.py` script. It draws the boxes on the images so you can verify that the data was processed correctly.

**Command:**
```bash
# Visualise the training set
python scripts/vistas_dataset/visualise_data.py --split train

# Visualise the validation set
python scripts/vistas_dataset/visualise_data.py --split val
```
