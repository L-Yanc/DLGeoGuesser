# OCR Pipeline

This directory contains the Optical Character Recognition (OCR) pipeline. Its primary purpose is to extract text from images, with a focus on multilingual support.

The pipeline is built around the `easyocr` library and uses multiple models to detect text from different language families (e.g., Latin, Cyrillic, Arabic).

## Core Functionality

The `MultiLangOCR` class is the main component. It is designed to work in tandem with an object detector like the `YOLOv8Detector`. The typical workflow is:

1.  An object detector identifies regions of interest in an image (e.g., signs, banners).
2.  The `ocr_pipeline` receives these regions (bounding boxes).
3.  For each region, it crops the original image.
4.  It runs multiple `easyocr` models on the cropped image.
5.  It consolidates the results and attempts to detect the language of the extracted text using `langdetect`.

The `MultiLangOCR` class is memory-intensive to initialise because it loads several language models simultaneously.

## Command-Line Interface

A simple CLI is provided in `main.py` for testing the OCR pipeline on a single image.

### CLI Usage

You can run OCR on a full image or, if you provide YOLO weights, it will first detect objects and then run OCR on the relevant ones.

**Flags**

-   `--image <path>`: (Required) Path to the source image.
-   `--yolo_weights <path>`: (Optional) Path to YOLOv8 weights. If provided, OCR will be targeted at detected objects.

**Example 1: Run OCR on the full image**

```bash
python -m src.dl_geoguesser.vision.ocr_pipeline.main \
  --image "path/to/your/image.jpg"
```

**Example 2: Use YOLO to guide OCR**

```bash
python -m src.dl_geoguesser.vision.ocr_pipeline.main \
  --image "path/to/your/image.jpg" \
  --yolo_weights "runs/yolo/yolo_run_1/weights/best.pt"
```

## Usage (as a Library Component)

The main use case is to integrate `MultiLangOCR` into a larger analysis pipeline.

```python
import cv2
from dl_geoguesser.vision.ocr_pipeline.model import MultiLangOCR
from dl_geoguesser.vision.yolo_detector.model import YOLOv8Detector

# --- Configuration ---
YOLO_WEIGHTS = "runs/yolo/yolo_run_1/weights/best.pt"
IMAGE_PATH = "path/to/an/image.jpg"

# --- Initialise Models ---
# Note: MultiLangOCR is slow to initialise
ocr_detector = MultiLangOCR()
yolo_detector = YOLOv8Detector(model_path=YOLO_WEIGHTS)

# --- Run Pipeline ---
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 1. Get object detections from YOLO
yolo_predictions = yolo_detector.predict(image_rgb, 0.3)

# 2. Process relevant objects with OCR
# The function automatically filters for object classes that are likely to have text.
from dl_geoguesser.vision.ocr_pipeline.main import process_yolo_predictions
ocr_results = process_yolo_predictions(yolo_predictions, image_rgb, ocr_detector)

# --- Process Results ---
for bbox, data in ocr_results.items():
    print(f"Text found at {bbox}:")
    print(f"  - Content: '{data['text']}'")
    if data['languages']:
        top_lang = data['languages'][0]
        print(f"  - Detected Language: {top_lang['lang']} (Confidence: {top_lang['confidence']:.2f})")

```

The output dictionary is structured as follows:

```python
{
    (100, 150, 120, 180): { # Bounding box tuple
        "text": "Hello World",
        "languages": [
            {"lang": "en", "confidence": 0.99},
            {"lang": "de", "confidence": 0.01}
        ]
    }
}
```