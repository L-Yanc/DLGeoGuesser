
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

    clip_vibe = ClipVibe(weights_path=args.weights)

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
    print(f"Found {len(image_paths)} images to predict.")

    for image_path in image_paths:
        try:
            pil_image = Image.open(image_path).convert("RGB")
        except IOError:
            print(f"Warning: Could not read image {image_path}, skipping.")
            continue

        class_scores = clip_vibe.predict(pil_image)
        top_class = max(class_scores, key=class_scores.get)
        
        print(f"Generating heatmap for top class: {top_class} for image {image_path.name}")
        heatmap_overlay = clip_vibe.generate_gradcam(pil_image, top_class)
        
        output_path = output_dir / f"{image_path.stem}_heatmap.jpg"
        print(f"Saving heatmap to {output_path.resolve()}")
        heatmap_overlay.save(output_path)
        print(f"Saved heatmap successfully.")


if __name__ == "__main__":
    default_source = str(ROOT_DIR / 'data' / 'processed' / 'clip_vibe')
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for a directory of images using a trained ClipVibe model.")
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
        help="Directory to save the heatmap overlay images."
    )
    args = parser.parse_args()
    main(args)
