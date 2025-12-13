# 🌍 GeoGuesser Game Show UI

An interactive web interface for the DLGeoGuesser AI system that analyzes images to predict locations using computer vision and language models.

## Features

### 🎯 Image Analysis Pipeline
- **CLIP (ClipVibe)**: Scene classification (urban, rural, historic, etc.)
- **YOLO**: Object detection (traffic signs, buildings, vehicles)
- **DINO**: Country prediction from detected regions
- **OCR**: Text extraction from signs and billboards

### 🤖 AI Analysis
- Automatic LLM analysis after image processing
- Three model options:
  - **Base**: Pre-trained model
  - **Mid**: Mid-training checkpoint
  - **SFT**: Fine-tuned for chat (recommended)
- Provides top 2 location guesses with explanations

### 💬 Interactive Chat
- Ask follow-up questions about the analyzed image
- Conversation history maintained
- Context-aware responses using analysis results

### 🎨 Visualizations
- Combined grid view: Original | Detections | Heatmap
- CLIP attention heatmaps showing focus areas
- YOLO bounding boxes on detected objects

## Quick Start

### 1. Install Dependencies
```bash
conda activate rlenv
pip install -e .
```

### 2. Start the Server
```bash
cd ui
python app.py
```

### 3. Open Browser
Navigate to: `http://localhost:5001`

## Usage

### Analyze an Image

1. **Select AI Model** - Choose from dropdown (Base, Mid, or SFT)
2. **Upload Image** - Click "UPLOAD IMAGE" or use camera
3. **Analyze** - Click "ANALYZE NOW!" button
4. **View Results**:
   - 🤖 AI Analysis (LLM's location guesses)
   - 🖼️ Visualization grid
   - 🌍 Location predictions
   - 🎨 Scene type (vibe)
   - 📝 Extracted text (if any)
   - 🔍 Detected objects

### Chat with AI

After analysis completes:
1. Chat input becomes enabled
2. Ask questions about the location
3. AI responds using selected model
4. Click 🗑️ to clear conversation

## Model Selection

### Self-Trained Models

**Base (Pre-trained)**
- Raw pre-trained model
- Fast, unbiased predictions
- May not follow format perfectly

**Mid (Checkpoint)**
- Mid-training checkpoint
- Balance of speed and quality
- Good for testing

**SFT (Fine-tuned)** ⭐ Recommended
- Fine-tuned for chat/instructions
- Best format adherence
- Most conversational


## Architecture

```
User Upload
    ↓
Vision Models (Parallel)
    ├─ CLIP → Scene Type
    ├─ YOLO → Objects
    └─ OCR → Text
    ↓
DINO → Country Prediction
    ↓
Selected LLM → Analysis
    ↓
Display Results + Enable Chat
```

## Configuration

### Model Paths
Edit `ui/app.py`:
```python
DEFAULT_WEIGHTS = {
    "clip": "runs/clip_vibe/clip_vibe_precomputed_run/best.pt",
    "yolo": "runs/yolo/M1Pro_run/weights/best.pt",
    "dino": "runs/dino/dino_precomputed/best.pt",
}
```

### Generation Parameters
In `generate_initial_analysis()`:
```python
max_tokens=100,      # Response length
temperature=0.6,     # Creativity (0.0-1.0)
top_k=40,           # Sampling diversity
```

### Enable/Disable OCR
```python
enable_ocr=True,  # Set to False to skip text extraction
```

## Performance

### Timing (CPU)
- Vision models: 5-10 seconds
- LLM analysis: 1-2 seconds
- Total: 6-12 seconds

### Memory Usage
- Vision models: ~2GB RAM
- LLM model: ~500MB RAM
- Total: ~2.5GB RAM

### Optimization Tips
- Use GPU for faster processing
- Reduce `max_tokens` for quicker LLM responses
- Disable OCR if text extraction not needed

## Troubleshooting

### Server Won't Start
```bash
# Check port 5001 is available
lsof -i :5001

# Kill existing process
kill -9 <PID>
```

### Models Not Loading
```bash
# Verify model files exist
ls -la runs/*/best.pt
ls -la models/d12_base_1k/*/

# Check tokenizer symlink
ls -la models/d12_base_1k/base_checkpoints/tokenizer
```

### Slow Performance
- Use GPU: Change `device="cpu"` to `device="cuda"` or `device="mps"`
- Disable OCR: Set `enable_ocr=False`
- Reduce image size before upload

### Chat Not Working
- Ensure image analyzed first
- Check server console for model loading errors
- Verify selected model loaded successfully

## File Structure

```
ui/
├── app.py                 # Flask backend
├── templates/
│   └── index.html        # Main UI
├── static/
│   ├── style.css         # Styling
│   ├── script.js         # Frontend logic
│   └── results/          # Generated visualizations
├── README.md             # This file
├── FEATURES.md           # Detailed features
├── STRUCTURE.md          # Code structure
└── CHAT_INTEGRATION.md   # Chat system docs
```

## API Endpoints

### POST /api/analyze
Analyze an uploaded image.

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
  "vibe": {...},
  "country": {...},
  "detections": {...},
  "ocr": {...},
  "llm_analysis": {
    "analysis": "1st: Slovenia - ...\n2nd: Slovakia - ...",
    "model": "self_trained_sft"
  },
  "visualizations": {"grid": "/static/results/..."}
}
```

### POST /api/chat/generate
Generate chat response.

**Request:**
```json
{
  "message": "Tell me more about the architecture",
  "model": "self_trained_sft",
  "session_id": "session_123",
  "context": {...}
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
Clear conversation history.

### GET /api/status
Check if pipeline is ready.

### GET /api/chat/status
Check if chat models are loaded.

## Development

### Adding New Models
1. Add checkpoint path to `initialize_chat_models()`
2. Add option to dropdown in `index.html`
3. Update model name formatting in `script.js`

### Modifying Prompt
Edit `generate_initial_analysis()` in `app.py`:
```python
prompt = f"""Your custom prompt here..."""
```

### Styling Changes
Edit `static/style.css` - uses modern GeoGuessr-inspired design:
- Vibrant blue (#4A90E2)
- Bright coral (#FF6B6B)
- Sunshine yellow (#FFD93D)
- Cyan (#7FDBFF)
- Hot pink (#FF69B4)

## Credits

- **Vision Models**: CLIP, YOLO, DINO
- **Language Models**: NanoChat (self-trained)
- **UI Design**: Modern GeoGuessr aesthetic
- **Framework**: Flask + Vanilla JS

## License

Part of the DLGeoGuesser project.
