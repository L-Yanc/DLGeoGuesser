#!/usr/bin/env python
"""
Interactive chat CLI for NanoChat language model.

Usage:
    python chat.py --stage base          # Use base pre-trained model
    python chat.py --stage mid           # Use mid-training checkpoint
    python chat.py --stage sft           # Use chat-finetuned model
    python chat.py --stage sft --stream  # Stream tokens as they're generated
"""

import argparse
import sys
import os

from dl_geoguesser.language.nanochat_model import NanoChatModel


# Map stage names to checkpoint paths
STAGE_PATHS = {
    "base": "models/d12_base_1k",
    "mid": "models/d12_base_1k/mid_checkpoints/d12",
    "sft": "models/d12_base_1k/chatsft_checkpoints/d12",
}


def main():
    parser = argparse.ArgumentParser(
        description="Chat with NanoChat language model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chat.py --stage base                          # Chat with base model
  python chat.py --stage sft --temperature 1.0         # More creative responses
  python chat.py --stage mid --max-tokens 200          # Longer responses
  python chat.py --stage sft --stream                  # Stream tokens live
        """
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=["base", "mid", "sft"],
        default="base",
        help="Model stage to use (default: base)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature, 0.0=greedy, higher=more random (default: 0.8)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens as they're generated"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt (non-interactive mode)"
    )

    args = parser.parse_args()

    # Get checkpoint path for selected stage
    checkpoint_dir = STAGE_PATHS[args.stage]

    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        print(f"Please ensure the {args.stage} model is available.")
        sys.exit(1)

    # Load model
    print("=" * 60)
    print(f"Loading NanoChat Model - Stage: {args.stage.upper()}")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_dir}")

    model = NanoChatModel(checkpoint_dir=checkpoint_dir, device=args.device)

    # Display model info
    info = model.get_model_info()
    print(f"\nModel Info:")
    print(f"  Training Step: {info['training_step']}")
    print(f"  Validation BPB: {info['validation_bpb']}")
    print(f"  Device: {info['device']}")
    print(f"\nGeneration Settings:")
    print(f"  Max Tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Streaming: {args.stream}")

    # Single prompt mode
    if args.prompt:
        print("\n" + "=" * 60)
        print(f"Prompt: {args.prompt}")
        print("-" * 60)

        if args.stream:
            print("Generated: ", end="", flush=True)
            for chunk in model.generate(
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                stream=True
            ):
                print(chunk, end="", flush=True)
            print()
        else:
            generated = model.generate(
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                stream=False
            )
            print(f"Generated: {generated}")

        print("=" * 60)
        return

    # Interactive chat mode
    print("\n" + "=" * 60)
    print("Interactive Chat Mode")
    print("=" * 60)
    print("Type your prompt and press Enter to generate text.")
    print("Commands:")
    print("  'quit' or 'exit' - Exit the chat")
    print("  'clear' - Clear screen")
    print("  'info' - Show model info")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        # Handle commands
        if prompt.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if prompt.lower() == "clear":
            os.system('clear' if os.name == 'posix' else 'cls')
            continue

        if prompt.lower() == "info":
            print(f"\nCurrent Settings:")
            print(f"  Stage: {args.stage}")
            print(f"  Max Tokens: {args.max_tokens}")
            print(f"  Temperature: {args.temperature}")
            print(f"  Top-K: {args.top_k}")
            print(f"  Device: {info['device']}")
            continue

        if not prompt:
            continue

        # Generate text
        print()
        if args.stream:
            for chunk in model.generate(
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                stream=True
            ):
                print(chunk, end="", flush=True)
            print()
        else:
            generated = model.generate(
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                stream=False
            )
            print(generated)


if __name__ == "__main__":
    main()
