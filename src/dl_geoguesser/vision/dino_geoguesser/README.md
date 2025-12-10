# DINO Geoguesser Model

This directory contains the implementation of a DINOv2-based country classifier. The model is designed to take cropped images of objects (detected by a model like YOLO) and predict the country of origin.

## Training

The model can be trained using the provided command-line interface. While the device can be set in `configs/dino_geoguesser.yaml`, it's recommended to override it with the `--device` command-line argument.

To start training, run the following command from the project root:

```bash
# To run on an Apple Silicon Mac
python -m src.dl_geoguesser.vision.dino_geoguesser.main train --name dino_mac_run --device mps

# To run on a an NVIDIA GPU
python -m src.dl_geoguesser.vision.dino_geoguesser.main train --name dino_gpu_run --device cuda
```

- `--name`: Specifies a unique name for the training run. All artifacts will be saved to `runs/dino/<name>`.
- `--device`: Specifies the hardware to run on (`cpu`, `mps`, `cuda`). This overrides any device setting in the config file.

The best performing model checkpoint will be saved as `runs/dino/<name>/best.pt`.

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
