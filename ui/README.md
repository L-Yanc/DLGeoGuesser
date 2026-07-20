# GeoGuesser Game Show UI

An interactive web interface for the DLGeoGuesser AI system that analyses images to predict locations using computer vision and language models.

## Features

### Image analysis pipeline
- **CLIP (ClipVibe)**: scene classification (urban, rural, historic, and so on)
- **YOLO**: object detection (traffic signs, buildings, vehicles)
- **DINO**: country prediction from detected regions
- **OCR**: text extraction from signs and billboards

### AI analysis
- Automatic LLM analysis after image processing
- Three model options:
  - **Base**: pre-trained model
  - **Mid**: mid-training checkpoint
  - **SFT**: fine-tuned for chat (recommended)
- Provides the top two location guesses with explanations

### Interactive chat
- Ask follow-up questions about the analysed image
- Conversation history is maintained
- Context-aware responses using the analysis results

### Visualisations
- Combined grid view: original, detections, heatmap
- CLIP attention heatmaps showing focus areas
- YOLO bounding boxes on detected objects

## Quick start

### 1. Install dependencies
```bash
pip install -e .
```

### 2. Start the server
```bash
cd ui
python app.py
```

### 3. Open the browser
Navigate to `http://localhost:5001`.

## Usage

### Analyse an image

1. **Select an AI model** from the dropdown (Base, Mid, or SFT).
2. **Upload an image** with "UPLOAD IMAGE" or the camera.
3. **Analyse** by clicking the "ANALYZE NOW!" button.
4. **View the results**:
   - AI analysis (the LLM's location guesses)
   - Visualisation grid
   - Location predictions
   - Scene type (vibe)
   - Extracted text, if any
   - Detected objects

### Chat with the AI

After analysis completes:
1. The chat input becomes enabled.
2. Ask questions about the location.
3. The AI responds using the selected model.
4. Clear the conversation with the clear button.

## Model selection

### Self-trained models

**Base (pre-trained)**
- Raw pre-trained model
- Fast, unbiased predictions
- May not follow the format perfectly

**Mid (checkpoint)**
- Mid-training checkpoint
- Balance of speed and quality
- Good for testing

**SFT (fine-tuned), recommended**
- Fine-tuned for chat and instructions
- Best format adherence
- Most conversational

## Architecture

```
User upload
    |
Vision models (parallel)
    |- CLIP -> scene type
    |- YOLO -> objects
    |- OCR  -> text
    |
DINO -> country prediction
    |
Selected LLM -> analysis
    |
Display results and enable chat
```

## Configuration

### Model paths
Edit `ui/app.py`:
```python
DEFAULT_WEIGHTS = {
    "clip": "runs/clip_vibe/clip_vibe_precomputed_run/best.pt",
    "yolo": "runs/yolo/M1Pro_run/weights/best.pt",
    "dino": "runs/dino/dino_precomputed/best.pt",
}
```

### Generation parameters
In `generate_initial_analysis()`:
```python
max_tokens=100,      # Response length
temperature=0.6,     # Creativity (0.0-1.0)
top_k=40,            # Sampling diversity
```

### Enable or disable OCR
```python
enable_ocr=True,  # Set to False to skip text extraction
```

## Performance

### Timing (CPU)
- Vision models: 5-10 seconds
- LLM analysis: 1-2 seconds
- Total: 6-12 seconds

### Memory usage
- Vision models: around 2GB RAM
- LLM model: around 500MB RAM
- Total: around 2.5GB RAM

### Optimisation tips
- Use a GPU for faster processing
- Reduce `max_tokens` for quicker LLM responses
- Disable OCR if text extraction is not needed

## Troubleshooting

### Server will not start
```bash
# Check port 5001 is available
lsof -i :5001

# Kill the existing process
kill -9 <PID>
```

### Models not loading
```bash
# Verify model files exist
ls -la runs/*/best.pt
ls -la models/d12_base_1k/*/

# Check the tokenizer symlink
ls -la models/d12_base_1k/base_checkpoints/tokenizer
```

### Slow performance
- Use a GPU: change `device="cpu"` to `device="cuda"` or `device="mps"`
- Disable OCR: set `enable_ocr=False`
- Reduce the image size before upload

### Chat not working
- Ensure the image is analysed first
- Check the server console for model loading errors
- Verify the selected model loaded successfully

## File structure

```
ui/
├── app.py                 # Flask backend
├── templates/
│   └── index.html         # Main UI
├── static/
│   ├── style.css          # Styling
│   ├── script.js          # Frontend logic
│   └── results/           # Generated visualisations
├── run_game_show.sh       # Startup helper
└── README.md              # This file
```

## API endpoints

### POST /api/analyze
Analyse an uploaded image.

**Request:**
```json
{
  "image": "data:image/jpeg;base64,...",
  "model": "self_trained_sft",
  "session_id": "session_123"
}
```

**Response:**
```json
{
  "success": true,
  "vibe": {},
  "country": {},
  "detections": {},
  "ocr": {},
  "llm_analysis": {
    "analysis": "1st: Slovenia - ...\n2nd: Slovakia - ...",
    "model": "self_trained_sft"
  },
  "visualizations": {"grid": "/static/results/..."}
}
```

### POST /api/chat/generate
Generate a chat response.

**Request:**
```json
{
  "message": "Tell me more about the architecture",
  "model": "self_trained_sft",
  "session_id": "session_123",
  "context": {}
}
```

**Response:**
```json
{
  "success": true,
  "response": "The architecture shows...",
  "model": "self_trained_sft"
}
```

### POST /api/chat/clear
Clear the conversation history.

### GET /api/status
Check if the pipeline is ready.

### GET /api/chat/status
Check if the chat models are loaded.

## Development

### Adding new models
1. Add the checkpoint path to `initialize_chat_models()`.
2. Add the option to the dropdown in `index.html`.
3. Update the model name formatting in `script.js`.

### Modifying the prompt
Edit `generate_initial_analysis()` in `app.py`:
```python
prompt = f"""Your custom prompt here..."""
```

### Styling changes
Edit `static/style.css`, which uses a modern GeoGuessr-inspired design:
- Vibrant blue (#4A90E2)
- Bright coral (#FF6B6B)
- Sunshine yellow (#FFD93D)
- Cyan (#7FDBFF)
- Hot pink (#FF69B4)

## Credits

- **Vision models**: CLIP, YOLO, DINO
- **Language models**: NanoChat (self-trained)
- **UI design**: modern GeoGuessr aesthetic
- **Framework**: Flask and vanilla JS

## License

Part of the DLGeoGuesser project.
