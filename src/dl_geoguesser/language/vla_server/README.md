# VLA-PEFT Inference Server

FastAPI server that exposes the fine-tuned LLaMA-3.1-8B model for GeoGuesser explanations.

## Running the Server

### Basic Usage

```bash
# From DLGeoGuesser root
python -m uvicorn src.dl_geoguesser.language.vla_server.server:app --host 0.0.0.0 --port 8000
```

### With Custom Configuration

```bash
# Set environment variables for custom model paths
export VLA_BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export VLA_LORA_CHECKPOINT="/path/to/checkpoint-1494"

# Run server
uvicorn src.dl_geoguesser.language.vla_server.server:app --host 0.0.0.0 --port 8000
```

### Running in Background (RunPod/Server)

```bash
# Start in background
nohup uvicorn src.dl_geoguesser.language.vla_server.server:app --host 0.0.0.0 --port 8000 > vla_server.log 2>&1 &

# Check if running
ps aux | grep uvicorn

# View logs
tail -f vla_server.log
```

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "ok": true,
  "device": "cuda:0",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "lora_checkpoint": "./checkpoints/checkpoint-1494"
}
```

### Chat Inference

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful GeoGuesser assistant."},
      {"role": "user", "content": "Based on the signs I see, where am I?"}
    ],
    "max_new_tokens": 256,
    "temperature": 0.7
  }'
```

Response:
```json
{
  "assistant": "Based on the road signs and architectural features..."
}
```

## Integration with DLGeoGuesser UI

The UI app can use `VLAInferenceClient` from `vla_client.py`:

```python
from src.dl_geoguesser.language import VLAInferenceClient

# Initialize client (pointing to RunPod or local server)
vla_client = VLAInferenceClient(base_url="https://your-runpod-url")

# Generate explanation
vision_data = {
    "yolo_detections": [...],
    "clip_predictions": {...},
    "dino_predictions": {...}
}

explanation = vla_client.generate_explanation(vision_data)
print(explanation)
```

## Requirements

- PyTorch 2.9.1+
- transformers
- peft
- fastapi
- uvicorn
- CUDA (optional, for GPU acceleration)

## Notes

- The server automatically falls back to base model if LoRA checkpoint is corrupted or missing
- For RunPod deployment, ensure port 8000 is exposed via the proxy URL
- Memory requirements: ~16GB GPU VRAM (with 4-bit quantization) or ~32GB CPU RAM
