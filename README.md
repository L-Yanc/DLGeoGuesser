# DLGeoGuesser

DLGeoGuesser is an AI system that looks at a street-level photo and tries to work out where in the world it was taken, in the spirit of the game GeoGuessr. It combines several computer vision models with language models so that the system can both analyse an image and explain its reasoning in plain English.

There is a full video walkthrough where we explain how the whole system fits together and demo it end to end:

[Watch the project walkthrough](https://drive.google.com/file/d/1uFbpHYTAq128VjMg6Ixk2_TsSmEWHpf_/view?usp=sharing)

## What it does

Given a single image, the system runs it through a chain of vision models, gathers the evidence they produce, and then hands that evidence to a language model that reasons about the likely location. The result is a ranked list of guesses along with an explanation of why the model thinks so.

Under the hood there are two families of models working together:

- **Vision models** pull structured information out of the image. They detect objects, read any visible text, classify the general environment, and estimate the country of origin.
- **Language models** take that structured information and turn it into a conversational answer. They produce the top location guesses, explain the reasoning, and let you ask follow-up questions.

## The pipeline

```
User uploads an image
        |
Vision models
  |- CLIP  ->  scene type / vibe (urban, rural, coastal, historic, and so on)
  |- YOLO  ->  object detections (signs, poles, vehicles, buildings)
  |- OCR   ->  text read from signs and billboards
        |
DINOv2  ->  country prediction from the detected regions
        |
Language model  ->  ranked location guesses plus an explanation
        |
Results shown in the UI, with chat enabled for follow-up questions
```

## Models

### Vision

- **YOLOv8 detector** (`src/dl_geoguesser/vision/yolo_detector/`). A YOLOv8 object detector trained to find location-relevant objects such as road signs, poles, and other street furniture. These detections feed the country classifier and the OCR stage.
- **CLIP vibe classifier** (`src/dl_geoguesser/vision/clip_vibe/`). A CLIP-based classifier that predicts the overall environment or "vibe" of a scene, for example beach, mountain road, or suburb. It also produces attention heatmaps so you can see where the model is looking.
- **DINOv2 country classifier** (`src/dl_geoguesser/vision/dino_geoguesser/`). A country classifier built on a frozen `facebook/dinov2-base` backbone with a small trainable MLP head. It works on cropped regions passed to it by the YOLO detector, using the fine-grained visual features of DINOv2 to guess the country.
- **OCR pipeline** (`src/dl_geoguesser/vision/ocr_pipeline/`). A multi-language OCR pipeline built on `easyocr` that finds and reads text in an image, which is a strong signal for language and region.
- **Vision pipeline** (`src/dl_geoguesser/vision/pipeline/`). Chains the vision models together and produces a single structured output for an image.

### Language

- **VLA fine-tuned model** (`models/llama_finetune_1494/`). A LLaMA 3.1 8B model with LoRA adapters, fine-tuned on GeoGuesser data to generate location explanations from the vision output. It runs behind a small FastAPI server so inference can happen locally or on a remote GPU.
- **NanoChat models** (`src/dl_geoguesser/language/nanochat_model/`). Small GPT-style models trained from scratch, which is the from-scratch component of the project. There are three variants: a base checkpoint, a mid-training checkpoint, and a chat fine-tuned checkpoint. These are light enough to run on CPU.
- **Gemini client** (`src/dl_geoguesser/language/gemini_client.py`). A client for Google Gemini models, used as a strong pre-trained baseline for comparison.

Together these cover the three points the coursework asks for: a fine-tuned model, a model trained from scratch, and a state-of-the-art pre-trained model for comparison.

## Project layout

```
DLGeoGuesser/
├── configs/     YAML configs for the models and datasets
├── data/        Raw and processed datasets
├── models/      Saved language model checkpoints
├── runs/        Output of training runs (checkpoints, logs, metrics)
├── scripts/     Standalone scripts for data prep, prediction, and pipelines
├── src/         Main Python package (dl_geoguesser)
│   └── dl_geoguesser/
│       ├── language/   Language model implementations
│       └── vision/     Vision model implementations
└── ui/          Flask web app (the "Game Show" interface)
```

Most modules and script folders have their own README with the detailed options, so start there when you want to dig into a specific part.

## Setup

Run these from the project root.

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install the project package in editable mode so you can import from `src`:
   ```bash
   pip install -e .
   ```

If you want to use the Gemini client or the remote VLA server, copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

- `GEMINI_API_KEY` for the Gemini client.
- `VLA_SERVER_URL` for the VLA inference server (defaults to `http://localhost:8000`).

## Running the demo

The easiest way to see everything working is the web UI, which runs the full pipeline on an image and lets you chat with the model afterwards.

```bash
python ui/app.py
```

Or use the helper script:

```bash
./ui/run_game_show.sh
```

Then open `http://localhost:5001` in your browser, pick a language model, upload an image, and hit analyse. You will see the scene type, detected objects, extracted text, country prediction, the visualisation grid, and the language model's ranked guesses. Once analysis finishes you can ask follow-up questions in the chat box.

See `ui/README.md` for the full list of UI features, model options, and API endpoints.

## Working with the models directly

Each core module has a command-line interface, so you can train or run them on their own.

Train the YOLO detector:

```bash
python -m src.dl_geoguesser.vision.yolo_detector.main train \
    --model yolov8n.pt \
    --data configs/vistas_yolo.yaml \
    --epochs 50 \
    --name yolo_vistas_run_1 \
    --device mps
```

Train the DINO country classifier:

```bash
python -m src.dl_geoguesser.vision.dino_geoguesser.main train \
    --name my_dino_run \
    --device mps
```

Results are written to `runs/`, one subdirectory per run, with checkpoints and logs inside.

Run the full pipeline with VLA explanations from the command line:

```bash
python scripts/pipeline/run_with_vla.py test_images/sample.jpg
```

For the individual training and inference options, check the README inside each module and script folder.

## Language model server

The VLA fine-tuned model runs behind a FastAPI server. To start it locally:

```bash
export VLA_BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export VLA_LORA_CHECKPOINT="./models/llama_finetune_1494"

python -m uvicorn src.dl_geoguesser.language.vla_server.server:app --host 0.0.0.0 --port 8000
```

The 8B model wants roughly 16GB of GPU memory or a lot of system RAM, so for CPU-only setups the NanoChat models are the better choice. Full details, including remote deployment, are in `src/dl_geoguesser/language/README.md`.
