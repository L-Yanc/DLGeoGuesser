"""
Checkpoint loading utilities for NanoChat models.
"""

import os
import json
import torch

from .gpt import GPT, GPTConfig
from .tokenizer import load_tokenizer


def load_checkpoint(checkpoint_dir, device="cpu"):
    """
    Load a NanoChat model checkpoint.

    Args:
        checkpoint_dir: Directory containing model checkpoint files
        device: Device to load model on ("cpu", "cuda", "mps")

    Returns:
        Tuple of (model, tokenizer, metadata)
        - model: Loaded GPT model
        - tokenizer: Tokenizer instance
        - metadata: Dict containing model config and training info
    """
    # Load metadata
    meta_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("meta_") and f.endswith(".json")]
    if not meta_files:
        raise FileNotFoundError(f"No metadata file found in {checkpoint_dir}")

    # Find the latest checkpoint
    meta_file = sorted(meta_files)[-1]
    meta_path = os.path.join(checkpoint_dir, meta_file)

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    # Extract step number from metadata filename
    step = metadata.get("step", int(meta_file.split("_")[1].split(".")[0]))

    # Load model weights
    model_file = f"model_{step:06d}.pt"
    model_path = os.path.join(checkpoint_dir, model_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from {model_path}")
    model_data = torch.load(model_path, map_location=device, weights_only=False)

    # Handle bfloat16 on CPU/MPS
    if device in ["cpu", "mps"]:
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }

    # Remove torch.compile prefix if present
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    # Build model from config
    model_config_kwargs = metadata["model_config"]
    print(f"Model config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)

    # Initialize model on meta device first (doesn't allocate memory)
    with torch.device("meta"):
        model = GPT(model_config)

    # Move to target device and load weights
    model.to_empty(device=device)
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()  # Set to evaluation mode

    # Initialize rotary embeddings (they're not saved in checkpoint)
    model.init_rotary_embeddings()

    # Load tokenizer - check checkpoint dir first, then parent dirs
    tokenizer_dir = os.path.join(checkpoint_dir, "tokenizer")

    if not os.path.exists(tokenizer_dir):
        # Try parent directory (for mid/sft checkpoints)
        parent_dir = os.path.dirname(os.path.dirname(checkpoint_dir))
        tokenizer_dir = os.path.join(parent_dir, "tokenizer")

        if not os.path.exists(tokenizer_dir):
            # Try grandparent (alternative structure)
            grandparent_dir = os.path.dirname(parent_dir)
            tokenizer_dir = os.path.join(grandparent_dir, "tokenizer")

            if not os.path.exists(tokenizer_dir):
                raise FileNotFoundError(
                    f"Tokenizer directory not found in checkpoint dir or parent dirs.\n"
                    f"Searched:\n"
                    f"  - {os.path.join(checkpoint_dir, 'tokenizer')}\n"
                    f"  - {os.path.join(parent_dir, 'tokenizer')}\n"
                    f"  - {os.path.join(grandparent_dir, 'tokenizer')}"
                )

    print(f"Using tokenizer from: {tokenizer_dir}")
    tokenizer = load_tokenizer(tokenizer_dir)

    # Verify compatibility
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], \
        f"Tokenizer vocab size ({tokenizer.get_vocab_size()}) != model vocab size ({model_config_kwargs['vocab_size']})"

    print(f"Successfully loaded model at step {step}")
    print(f"Validation bpb: {metadata.get('val_bpb', 'N/A')}")

    return model, tokenizer, metadata


def find_checkpoint_dir(base_dir, model_name=None):
    """
    Find checkpoint directory by name or use the most recent one.

    Args:
        base_dir: Base directory containing model checkpoints
        model_name: Optional model name (e.g., "d12")

    Returns:
        Path to checkpoint directory
    """
    if model_name is not None:
        checkpoint_dir = os.path.join(base_dir, model_name)
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        return checkpoint_dir

    # Find all checkpoint directories
    checkpoint_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {base_dir}")

    # Sort by modification time and return most recent
    checkpoint_dirs.sort(
        key=lambda d: os.path.getmtime(os.path.join(base_dir, d)),
        reverse=True
    )

    return os.path.join(base_dir, checkpoint_dirs[0])
