# Language Models for DLGeoGuesser

This module provides language model integrations for generating natural language explanations of GeoGuesser predictions.

## Available Models

### 1. VLA Fine-tuned Model (Recommended)
**Location**: `models/llama_finetune_1494/`  
**Type**: LLaMA 3.1 8B with LoRA adapters, fine-tuned on GeoGuesser data  
**Usage**: Remote inference via VLA server

The VLA (Vision-Language Assistant) model is a fine-tuned LLaMA model specifically trained to generate GeoGuesser explanations based on vision model outputs.

#### Architecture
```
Vision Pipeline → VLA Client → VLA Server (FastAPI) → Fine-tuned LLaMA → Explanation
```

#### Starting the VLA Server

**Local (requires GPU or good CPU):**

Option 1 - Using the startup script (recommended):
```bash
# Simple start with defaults
python scripts/start_vla_server.py

# Or with bash script
./scripts/start_vla_server.sh
```

Option 2 - Manual start:
```bash
# Set environment variables
export VLA_BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export VLA_LORA_CHECKPOINT="./models/llama_finetune_1494"

# Start server
python -m uvicorn src.dl_geoguesser.language.vla_server.server:app --host 0.0.0.0 --port 8000
```

**Remote (RunPod recommended):**
1. Deploy the VLA server on RunPod with GPU
2. Set the server URL:
```bash
export VLA_SERVER_URL="https://your-pod-id.proxy.runpod.net"
```

#### Using VLA Client

```python
from src.dl_geoguesser.language.vla_client import VLAInferenceClient

# Connect to server
client = VLAInferenceClient("http://localhost:8000")

# Check health
health = client.health_check()
print(health)

# Generate explanation
vision_data = {
    "country": "france",
    "country_confidence": 0.85,
    "vibe_top": "suburban residential",
    "vibe_distribution": {"suburban residential": 0.45, "urban": 0.25},
    "evidence": {
        "top_contents": ["road_sign", "architecture"],
        "detected_text": "Rue de la Paix",
        "detected_languages": ["fr"]
    }
}

explanation = client.generate_explanation(
    vision_data=vision_data,
    max_tokens=128,
    temperature=0.7
)
print(explanation)
```

### 2. NanoChat Models
**Location**: `models/d12_base_1k/`  
**Type**: Small GPT-style models trained from scratch  
**Usage**: Local inference (CPU-friendly)

Three variants available:
- `base_checkpoints`: Base pretrained model
- `mid_checkpoints/d12`: Mid-training checkpoint
- `chatsft_checkpoints/d12`: Chat-finetuned model

```python
from src.dl_geoguesser.language.nanochat_model import NanoChatModel

# Load model
model = NanoChatModel(
    checkpoint_dir="models/d12_base_1k/chatsft_checkpoints/d12",
    device="cpu"
)

# Generate text
text = model.generate(
    prompt="The capital of France is",
    max_tokens=50,
    temperature=0.8
)
print(text)
```

## Integration with Pipeline

### Command Line
```bash
# Run pipeline with VLA explanations
python scripts/pipeline/run_with_vla.py test_images/sample.jpg
```

### UI Integration
The UI (`ui/app.py`) supports both VLA and NanoChat models:

```bash
# Set VLA server URL (optional, defaults to localhost:8000)
export VLA_SERVER_URL="https://your-server.com"

# Start UI
python ui/app.py
```

The UI will:
1. Run vision pipeline on uploaded image
2. Generate initial analysis with selected model (VLA or NanoChat)
3. Enable chat for follow-up questions

### Programmatic Usage

```python
from src.dl_geoguesser.vision.pipeline import GeoLocationPipeline, PipelineConfig
from src.dl_geoguesser.language.vla_client import VLAInferenceClient

# Initialize pipeline
config = PipelineConfig(
    clip_weights="runs/clip_vibe/.../best.pt",
    yolo_weights="runs/yolo/.../best.pt",
    dino_weights="runs/dino/.../best.pt",
    device="cpu"
)
pipeline = GeoLocationPipeline(config)
pipeline.load_models()

# Initialize VLA client
vla_client = VLAInferenceClient("http://localhost:8000")

# Process image
from PIL import Image
image = Image.open("test.jpg")
result = pipeline.predict(image)

# Format for VLA
vision_data = {
    "country": result.top_country,
    "country_confidence": result.top_country_confidence,
    "vibe_top": result.top_vibe,
    "vibe_distribution": dict(sorted(
        result.vibe_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]),
    "evidence": {
        "top_contents": list(result.detections.keys()),
        "detected_text": result.extracted_text,
    }
}

# Generate explanation
explanation = vla_client.generate_explanation(vision_data)
print(explanation)
```

## Model Comparison

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| VLA Fine-tuned | 8B params | Slow (needs GPU) | High | Production, best explanations |
| NanoChat SFT | ~100M params | Fast (CPU) | Medium | Development, quick testing |
| NanoChat Base | ~100M params | Fast (CPU) | Low | Baseline comparison |

## Environment Variables

- `VLA_SERVER_URL`: URL of VLA inference server (default: `http://localhost:8000`)
- `VLA_BASE_MODEL`: HuggingFace model name (default: `meta-llama/Llama-3.1-8B-Instruct`)
- `VLA_LORA_CHECKPOINT`: Path to LoRA checkpoint (default: `./models/llama_finetune_1494`)

## Troubleshooting

### VLA Server Connection Issues
```python
# Test connection
from src.dl_geoguesser.language.vla_client import VLAInferenceClient
client = VLAInferenceClient("http://localhost:8000")
try:
    health = client.health_check()
    print("✅ Connected:", health)
except Exception as e:
    print("❌ Connection failed:", e)
```

### Model Loading Issues
- Ensure checkpoint paths are correct
- Check device availability (CUDA/MPS/CPU)
- Verify model files exist and are not corrupted

### Memory Issues
- VLA model requires ~16GB GPU memory or ~32GB RAM
- Use NanoChat models for CPU-only environments
- Consider using remote VLA server (RunPod) for production
