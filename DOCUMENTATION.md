# DLGeoGuesser Project Documentation

This document provides a comprehensive overview of the DLGeoGuesser project, its structure, and how to use its various components.

## High-Level Overview

DLGeoGuesser is an AI project designed to predict the geographic location of a street-level image, similar to the game GeoGuessr. It uses a combination of computer vision and natural language processing models to analyse images and provide location guesses.

The project is composed of two main types of models:
1.  **Vision Models**: Analyse the image to extract features, detect objects, read text, and determine the general "vibe" or environment (e.g., "forest", "city").
2.  **Language Models**: Act as conversational agents or analysers that take the structured output from the vision models to reason about the location and interact with a user.

---

## Project Structure

The project is organised into several key directories.

```
DLGeoGuesser/
├── configs/            # YAML configuration files for models.
├── data/               # Raw and processed datasets.
├── models/             # Saved checkpoints for language models.
├── runs/               # Output directories for model training runs.
├── scripts/            # Standalone scripts for data processing, prediction, etc.
├── src/                # Main Python source code for models and libraries.
│   └── dl_geoguesser/
│       ├── language/   # Language model implementations.
│       └── vision/     # Vision model implementations.
└── ui/                 # Flask-based web interface for the project.
```

-   `configs/`: Holds configuration files, typically `.yaml`, that define parameters for datasets and models.
-   `data/`: All datasets, both raw and processed, reside here.
-   `models/`: Contains pre-trained model weights and tokenizer files, especially for the language models.
-   `runs/`: This directory is the default output location for all training jobs. Each run creates a new subdirectory containing model checkpoints, logs, and performance metrics.
-   `scripts/`: Contains helper scripts organized by function (e.g., `yolo/`, `clip_vibe/`, `geohints_dataset/`). These are used for tasks like downloading data, preparing datasets, and running predictions. Each sub-directory has its own `README.md`.
-   `src/dl_geoguesser/`: The main Python package. All core model implementations and reusable logic live here.
-   `ui/`: A Flask-based web application that provides an interactive "Game Show" interface to the full pipeline.

---

## Core Modules (`src/dl_geoguesser`)

This is the primary package containing the model implementations. Each module is designed to be usable as a library component and also provides a command-line interface for direct interaction.

For detailed instructions on how to use each module, please refer to the `README.md` file within its respective directory.

### Vision Modules (`src/dl_geoguesser/vision`)

-   `yolo_detector/`: A wrapper for the **YOLOv8** model used for object detection. It can be trained to find specific objects relevant to GeoGuessing, like signs and utility poles.
-   `clip_vibe/`: A classifier based on **CLIP** that predicts the general environment or "vibe" of an image (e.g., "beach", "mountain road", "suburb").
-   `dino_geoguesser/`: A location classifier based on the **DINOv2** model.
-   `ocr_pipeline/`: A multi-language **Optical Character Recognition (OCR)** pipeline built on `easyocr`. It is designed to find and read text from images, especially from objects detected by YOLO.
-   `pipeline/`: Contains the main vision pipeline that chains the other vision models together to produce a single, structured JSON output from an image.

### Language Modules (`src/dl_geoguesser/language`)

-   `nanochat_model/`: A from-scratch implementation of a **GPT-style language model** for dialogue and analysis.
-   `gemini_client/`: A client for interacting with the **Google Gemini** family of models.
-   `vla_client/` & `vla_server/`: A client-server implementation for a **Vision-Language-Action (VLA)** model, allowing for remote inference.

---

## Installation and Setup

This project uses a virtual environment to manage dependencies. Follow these steps from the project root directory (`DLGeoGuesser/`).

1.  **Create and activate the virtual environment:**
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install the project package in editable mode:**
    This allows you to import modules from `src` in your scripts.
    ```bash
    pip install -e .
    ```

---

## Example Workflows

Each module and script is self-contained and documented in its own `README.md`. Below are high-level examples of how to use the project.

### 1. Training a Vision Model

Most vision models can be trained via the CLI in their respective `main.py` files. For example, to train the YOLOv8 detector:

```bash
# See the full list of options in src/dl_geoguesser/vision/yolo_detector/README.md
python -m src.dl_geoguesser.vision.yolo_detector.main train \
    --model yolov8n.pt \
    --data configs/vistas_yolo.yaml \
    --epochs 50 \
    --name yolo_vistas_run_1 \
    --device mps
```
All results will be saved to `runs/yolo/yolo_vistas_run_1/`.

### 2. Preparing a Dataset

The `scripts` directory contains tools for data preparation. For example, to process the GeoHints dataset:

```bash
# See scripts/geohints_dataset/README.md for more details
python scripts/geohints_dataset/process_geohints.py
```

### 3. Running the Full Pipeline

The `ui/` directory contains a web-based "Game Show" to run the full end-to-end pipeline on an image.

To start the web server:
```bash
./ui/run_game_show.sh
```
Or, from the project root:
```bash
python ui/app.py
```
Then, open your web browser to `http://localhost:5001`. You can upload an image and see the combined analysis from all the vision and language models.