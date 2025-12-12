import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dl_geoguesser.vision.clip_vibe.model import ClipVibe

# Define root directory relative to the script's location
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


def main(args):
    """Main prediction loop."""
    if not args.weights:
        print("Error: Please provide the path to the ClipVibe model weights using --weights.")
        return

    clip_vibe = ClipVibe(weights_path=args.weights, device=args.device)

    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = ROOT_DIR / source_path

    if not source_path.exists():
        print(f"Error: Source path does not exist: {source_path}")
        return

    output_dir = Path(args.output_dir)
    print(f"Output directory: {output_dir.resolve()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))) if source_path.is_dir() else [source_path]

    if not image_paths:
        print(f"No images found in {source_path}")
        return

    print(f"Loaded model: {args.weights}")
    print(f"Using device: {args.device if args.device else 'auto'}")
    print(f"Explanation method: {args.method}")
    if args.method == "integrated_gradients":
        print(f"Integration steps: {args.ig_steps}")
    print(f"Found {len(image_paths)} images to process.")
    print()

    for idx, image_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] Processing: {image_path.name}")

        try:
            pil_image = Image.open(image_path).convert("RGB")
        except IOError:
            print(f"  ⚠️  Could not read image, skipping.")
            continue

        # Get prediction
        class_scores = clip_vibe.predict(pil_image)
        top_class = max(class_scores, key=class_scores.get)
        top_score = class_scores[top_class]

        print(f"  Predicted: {top_class} (confidence: {top_score:.3f})")

        # Generate explanation
        print(f"  Generating {args.method} explanation...")
        try:
            heatmap_overlay = clip_vibe.explain(
                pil_image,
                top_class,
                method=args.method,
                ig_steps=args.ig_steps,
                alpha=args.alpha,
            )
        except Exception as e:
            print(f"  ⚠️  Failed to generate explanation: {e}")
            continue

        # Save output
        method_suffix = "ig" if args.method == "integrated_gradients" else "attention"
        output_path = output_dir / f"{image_path.stem}_{method_suffix}.jpg"
        heatmap_overlay.save(output_path)
        print(f"  ✓ Saved to: {output_path.name}")
        print()

    print(f"✓ Done! Processed {len(image_paths)} images.")
    print(f"  Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    default_source = str(ROOT_DIR / 'data' / 'processed' / 'clip_vibe')
    parser = argparse.ArgumentParser(
        description="Generate explanation visualizations for images using a trained ClipVibe model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate rigorous Integrated Gradients explanations (recommended)
  python predict_and_visualise.py --weights runs/model/best.pt --method integrated_gradients
  
  # Fast attention-based explanations for quick debugging
  python predict_and_visualise.py --weights runs/model/best.pt --method attention
  
  # Process a single image with custom settings
  python predict_and_visualise.py --weights runs/model/best.pt --source image.jpg --ig-steps 100 --alpha 0.6
        """
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to the trained ClipVibe model weights (.pt file)."
    )
    parser.add_argument(
        "--source", type=str, default=default_source,
        help=f"Path to the source image or directory of images. Default: {default_source}"
    )
    parser.add_argument(
        "--output_dir", type=str, default="test_images",
        help="Directory to save the explanation overlay images. Default: test_images"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (cpu, mps, cuda). Default: auto-detect"
    )
    parser.add_argument(
        "--method", type=str, default="integrated_gradients",
        choices=["integrated_gradients", "attention"],
        help=(
            "Explanation method. "
            "'integrated_gradients' (default): Rigorous, shows actual pixel importance (slower). "
            "'attention': Fast, shows CLIP attention patterns (less rigorous)."
        )
    )
    parser.add_argument(
        "--ig-steps", type=int, default=50,
        help="Number of integration steps for integrated_gradients method. More = more accurate but slower. Default: 50"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Overlay transparency (0.0 = original image, 1.0 = full heatmap). Default: 0.5"
    )
    args = parser.parse_args()
    main(args)
