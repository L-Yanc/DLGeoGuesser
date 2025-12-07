"""
CLI interface for the NanoChat language model.

Provides commands for text generation and model inspection.
"""

import argparse
import sys

from dl_geoguesser.language.nanochat_model.model import NanoChatModel


def main():
    """
    Main entrypoint for the NanoChat model module.
    Provides a CLI for text generation and model info.
    """
    parser = argparse.ArgumentParser(description="NanoChat Language Model CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Generate Command ---
    parser_generate = subparsers.add_parser("generate", help="Generate text from a prompt.")
    parser_generate.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint directory.")
    parser_generate.add_argument("--prompt", type=str, default="", help="Text prompt for generation. If empty, enters interactive mode.")
    parser_generate.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate.")
    parser_generate.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0.0 = greedy, higher = more random).")
    parser_generate.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter.")
    parser_generate.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser_generate.add_argument("--device", type=str, default=None, choices=['cpu', 'cuda', 'mps'], help="Device to use (default: auto-detect).")
    parser_generate.add_argument("--stream", action="store_true", help="Stream tokens as they're generated.")

    # --- Info Command ---
    parser_info = subparsers.add_parser("info", help="Display model information.")
    parser_info.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint directory.")
    parser_info.add_argument("--device", type=str, default=None, choices=['cpu', 'cuda', 'mps'], help="Device to use (default: auto-detect).")

    # --- Interactive Command ---
    parser_interactive = subparsers.add_parser("interactive", help="Start interactive text generation session.")
    parser_interactive.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint directory.")
    parser_interactive.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate per prompt.")
    parser_interactive.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser_interactive.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter.")
    parser_interactive.add_argument("--device", type=str, default=None, choices=['cpu', 'cuda', 'mps'], help="Device to use (default: auto-detect).")

    args = parser.parse_args()

    if args.command == "generate":
        # Load model
        print(f"Loading model from {args.checkpoint}...")
        model = NanoChatModel(checkpoint_dir=args.checkpoint, device=args.device)

        if args.prompt:
            # Single prompt generation
            print(f"\nPrompt: {args.prompt}")
            print("Generated text:")
            print("-" * 50)

            if args.stream:
                # Streaming generation
                print(args.prompt, end="", flush=True)
                for text_chunk in model.generate(
                    args.prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=args.seed,
                    stream=True
                ):
                    print(text_chunk, end="", flush=True)
                print()
            else:
                # Non-streaming generation
                generated = model.generate(
                    args.prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=args.seed,
                    stream=False
                )
                print(args.prompt + generated)
            print("-" * 50)
        else:
            # Interactive mode
            print("\nEntering interactive mode. Type 'quit' or 'exit' to end.")
            print("-" * 50)
            interactive_loop(model, args)

    elif args.command == "interactive":
        # Interactive mode
        print(f"Loading model from {args.checkpoint}...")
        model = NanoChatModel(checkpoint_dir=args.checkpoint, device=args.device)
        interactive_loop(model, args)

    elif args.command == "info":
        # Display model info
        print(f"Loading model from {args.checkpoint}...")
        model = NanoChatModel(checkpoint_dir=args.checkpoint, device=args.device)
        print("\nModel Information:")
        print("=" * 50)
        info = model.get_model_info()
        print(f"Model Config: {info['model_config']}")
        print(f"Training Step: {info['training_step']}")
        print(f"Validation BPB: {info['validation_bpb']}")
        print(f"Vocabulary Size: {info['vocab_size']}")
        print(f"Device: {info['device']}")
        print("=" * 50)


def interactive_loop(model, args):
    """
    Run an interactive text generation loop.

    Args:
        model: NanoChatModel instance
        args: Command-line arguments
    """
    print("\nInteractive Text Generation")
    print("-" * 50)
    print("Type a prompt and press Enter to generate.")
    print("Type 'quit' or 'exit' to end.")
    print("-" * 50)

    while True:
        try:
            prompt = input("\nPrompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if prompt.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if not prompt:
            continue

        # Generate text
        print("\nGenerated:", end=" ", flush=True)
        generated = model.generate(
            prompt,
            max_tokens=getattr(args, 'max_tokens', 100),
            temperature=getattr(args, 'temperature', 0.8),
            top_k=getattr(args, 'top_k', 50),
            stream=False
        )
        print(generated)


if __name__ == "__main__":
    main()
