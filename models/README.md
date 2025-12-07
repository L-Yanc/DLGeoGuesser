# NanoChat Language Model

This directory contains trained language model checkpoints for text generation.

## Available Models

### d12_base_1k
- **Architecture**: GPT with 12 layers, 768 hidden dimensions
- **Parameters**: ~100M
- **Training**: 1000 iterations on base training data
- **Validation BPB**: 1.0214
- **Model file**: `model_001000.pt` (612MB, tracked by Git LFS)

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Quick Test

```bash
python test_nanochat.py
```

### 3. Python API

```python
from src.dl_geoguesser.language.nanochat_model import NanoChatModel

# Load model
model = NanoChatModel(checkpoint_dir="models/d12_base_1k")

# Generate text
text = model.generate(
    prompt="The capital of France is",
    max_tokens=50,
    temperature=0.8,
    top_k=50
)
print(text)
```

### 4. CLI Interface

```bash
# Interactive mode
python -m src.dl_geoguesser.language.nanochat_model.main interactive \
    --checkpoint models/d12_base_1k

# Single generation
python -m src.dl_geoguesser.language.nanochat_model.main generate \
    --checkpoint models/d12_base_1k \
    --prompt "The capital of France is" \
    --max-tokens 50

# Model info
python -m src.dl_geoguesser.language.nanochat_model.main info \
    --checkpoint models/d12_base_1k
```

## Model Files

Each checkpoint directory contains:
- `model_XXXXXX.pt` - Model weights (tracked by Git LFS)
- `meta_XXXXXX.json` - Model configuration and training metadata
- `tokenizer/` - Tokenizer files (BPE with tiktoken)
  - `tokenizer.pkl` - Tokenizer encoding
  - `token_bytes.pt` - Token byte mappings (tracked by Git LFS)

## Git LFS

Model weights (`.pt` files) are stored using Git LFS. When you clone this repo:

```bash
# Make sure Git LFS is installed
git lfs install

# Clone will automatically download LFS files
git clone <repo-url>
```

## Integration with Vision Models

This language model complements the existing vision pipeline:
- **YOLO Detector** → Detects objects in images
- **OCR Pipeline** → Extracts text from detected objects
- **NanoChat LM** → Can process/generate text for downstream tasks
