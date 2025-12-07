"""
NanoChat language model wrapper for text generation.

This module provides a high-level interface for loading and using
the trained NanoChat language model for text generation tasks.
"""

import torch
from contextlib import nullcontext

from .checkpoint import load_checkpoint, find_checkpoint_dir
from .engine import Engine


class NanoChatModel:
    """
    A wrapper class for the NanoChat language model.

    This class provides a structured interface for loading and running
    inference with a trained NanoChat GPT model. It standardizes
    interactions with the underlying model and tokenizer and is designed
    to be used as a component in a larger pipeline.

    Attributes:
        model: The GPT model instance
        tokenizer: The tokenizer instance
        metadata: Dict containing model config and training info
        device: Device the model is loaded on
        engine: Inference engine for efficient generation
    """

    def __init__(self, checkpoint_dir: str, device: str = None):
        """
        Initialize the NanoChatModel.

        Args:
            checkpoint_dir: Path to the checkpoint directory containing
                          model weights, metadata, and tokenizer
            device: Device to load model on. If None, auto-detects:
                   CUDA > MPS > CPU
        """
        if device is None:
            device = self._autodetect_device()

        print(f"Loading NanoChat model from: {checkpoint_dir}")
        print(f"Using device: {device}")

        # Load checkpoint
        self.model, self.tokenizer, self.metadata = load_checkpoint(
            checkpoint_dir, device=device
        )
        self.device = device

        # Create inference engine
        self.engine = Engine(self.model, self.tokenizer)

        # Set up autocast context for efficiency
        if device == "cuda":
            self.autocast_ctx = torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16
            )
        else:
            self.autocast_ctx = nullcontext()

        print("NanoChat model ready for inference")

    @staticmethod
    def _autodetect_device():
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @classmethod
    def from_name(cls, base_dir: str, model_name: str = None, device: str = None):
        """
        Load model by name from a base directory.

        Args:
            base_dir: Base directory containing model checkpoints
            model_name: Model name (e.g., "d12"). If None, uses most recent.
            device: Device to load on (None = auto-detect)

        Returns:
            NanoChatModel instance
        """
        checkpoint_dir = find_checkpoint_dir(base_dir, model_name)
        return cls(checkpoint_dir, device)

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.8,
                 top_k: int = 50, seed: int = 42, stream: bool = False):
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy, higher = more random)
            top_k: Top-k sampling parameter
            seed: Random seed for reproducibility
            stream: If True, yields tokens as they're generated.
                   If False, returns complete generated text.

        Returns:
            If stream=False: Complete generated text string
            If stream=True: Generator yielding text chunks
        """
        # Encode prompt
        bos = self.tokenizer.get_bos_token_id()
        tokens = self.tokenizer.encode(prompt, prepend=bos)

        if stream:
            # Streaming generation
            def token_stream():
                with self.autocast_ctx:
                    for token_column, _ in self.engine.generate(
                        tokens,
                        num_samples=1,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        seed=seed
                    ):
                        token = token_column[0]
                        text_chunk = self.tokenizer.decode([token])
                        yield text_chunk
            return token_stream()
        else:
            # Non-streaming generation
            generated_text = []
            with self.autocast_ctx:
                for token_column, _ in self.engine.generate(
                    tokens,
                    num_samples=1,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed
                ):
                    token = token_column[0]
                    text_chunk = self.tokenizer.decode([token])
                    generated_text.append(text_chunk)
            return ''.join(generated_text)

    def generate_batch(self, prompts: list, max_tokens: int = 100, temperature: float = 0.8,
                      top_k: int = 50, seed: int = 42):
        """
        Generate text for multiple prompts in parallel.

        Args:
            prompts: List of input text prompts
            max_tokens: Maximum number of tokens to generate per prompt
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            seed: Random seed

        Returns:
            List of generated text strings (one per prompt)
        """
        # For simplicity, we'll generate them one at a time
        # Could be optimized to use true batching later
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
                stream=False
            )
            results.append(result)
        return results

    def get_model_info(self):
        """
        Get information about the loaded model.

        Returns:
            Dict containing model configuration and training info
        """
        return {
            "model_config": self.metadata["model_config"],
            "training_step": self.metadata.get("step", "unknown"),
            "validation_bpb": self.metadata.get("val_bpb", "N/A"),
            "vocab_size": self.tokenizer.get_vocab_size(),
            "device": self.device,
        }

    def __repr__(self):
        """String representation of the model."""
        info = self.get_model_info()
        return (
            f"NanoChatModel(\n"
            f"  config={info['model_config']},\n"
            f"  step={info['training_step']},\n"
            f"  val_bpb={info['validation_bpb']},\n"
            f"  device={info['device']}\n"
            f")"
        )


if __name__ == '__main__':
    # Example usage
    print("NanoChat model wrapper created.")
    print("Example usage:")
    print("  model = NanoChatModel.from_name('models', 'd12')")
    print("  text = model.generate('The capital of France is', max_tokens=50)")
    print("  print(text)")
