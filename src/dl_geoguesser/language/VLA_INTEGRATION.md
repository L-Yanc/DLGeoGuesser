# VLA-PEFT Integration

This module provides integration with the VLA-PEFT (Vision-Language Action with Parameter-Efficient Fine-Tuning) model for GeoGuesser explanations.

## Components

### 1. VLA Server (`vla_server/`)
FastAPI server that exposes the fine-tuned LLaMA-3.1-8B model.

**Files:**
- `server.py`: Main FastAPI application with `/health` and `/chat` endpoints
- `README.md`: Server documentation and usage instructions

### 2. VLA Client (`vla_client.py`)
Python client for interfacing with the VLA server from the UI.

**Class:** `VLAInferenceClient`

**Methods:**
- `generate_explanation(vision_data, max_tokens, temperature, custom_prompt)`: Generate GeoGuessr explanation
- `health_check()`: Check if server is healthy and accessible

## Usage in UI

The VLA-PEFT model is integrated as one of the available chat models in the UI:

```python
from src.dl_geoguesser.language import VLAInferenceClient

# Initialize client (pointing to RunPod or local server)
vla_client = VLAInferenceClient(base_url="https://your-runpod-url")

# Check server status
if vla_client.health_check():
    print("VLA server ready!")

# Generate explanation
vision_data = {
    'yolo_detections': ['stop_sign', 'traffic_light'],
    'clip_predictions': {
        'top_vibe': 'urban',
        'countries': {'USA': 0.85, 'Canada': 0.10}
    },
    'ocr_text': 'Main Street'
}

explanation = vla_client.generate_explanation(vision_data)
```

## Configuration

Set the VLA server URL via environment variable:

```bash
export VLA_SERVER_URL="https://t5uuas4ux32flb-8000.proxy.runpod.net"
python ui/app.py
```

Or it defaults to the RunPod proxy URL configured in `app.py`.

## Running the Server

### On RunPod GPU Instance

```bash
# SSH into RunPod
ssh t5uuas4ux32flb-6441140d@ssh.runpod.io -i ~/.ssh/id_ed25519

# Navigate to project
cd /workspace/vla-peft

# Start server (runs in background)
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > vla_server.log 2>&1 &

# Check status
curl http://localhost:8000/health
```

### Using the Helper Script

```bash
# From DLGeoGuesser root
./scripts/run_vla_server.sh /path/to/checkpoint 8000
```

## Model Selection in UI

Users can select the VLA-PEFT model from the chat interface dropdown. The UI automatically:
1. Checks VLA server health on startup
2. Formats vision pipeline output for the VLA model
3. Handles both initial analysis and follow-up questions
4. Falls back gracefully if the server is unavailable

## Architecture

```
DLGeoGuesser UI (Flask)
    ↓
VLAInferenceClient (HTTP client)
    ↓ POST /chat
VLA Server (FastAPI on RunPod)
    ↓
LLaMA-3.1-8B + LoRA weights
    ↓
Natural language explanation
```

## Notes

- The VLA server requires GPU for optimal performance (16GB+ VRAM)
- Falls back to base model if LoRA checkpoint is corrupted/missing
- Server uses chat template format for proper prompt formatting
- Temperature and top_p parameters control generation randomness
