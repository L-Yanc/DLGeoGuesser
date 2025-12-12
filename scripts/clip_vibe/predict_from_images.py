
import argparse
from pathlib import Path

import torch
from PIL import Image

from dl_geoguesser.vision.clip_vibe.model import ClipVibe


def predict_from_directory(weights_path: str, images_dir: str, top_k: int, device: str):
    """
    Runs the ClipVibe model on all images in a directory and prints the top-k predictions.
    """
    images_path = Path(images_dir)
    if not images_path.exists() or not images_path.is_dir():
        print(f"Error: The specified image directory does not exist or is not a directory: {images_dir}")
        return

    try:
        clip_vibe_classifier = ClipVibe(weights_path=weights_path, device=device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    image_files = list(images_path.glob('*.*'))
    if not image_files:
        print(f"No images found in {images_dir}")
        return

    for image_path in image_files:
        try:
            image = Image.open(image_path).convert("RGB")
        except IOError:
            print(f"Could not open or read image file: {image_path}")
            continue

        class_scores = clip_vibe_classifier.predict(image)

        if class_scores:
            sorted_scores = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)

            print(f"--- Top {top_k} Predictions for {image_path.name} ---")
            for class_name, score in sorted_scores[:top_k]:
                print(f"- {class_name}: {score:.3f}")
            print("-" * (20 + len(str(top_k)) + len(image_path.name)))
        else:
            print(f"Could not make a prediction for {image_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Run ClipVibe on a directory of images.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the trained ClipVibe model weights (.pt file).")
    parser.add_argument("--images_dir", type=str, default="test_images", help="Directory containing the images to predict.")
    parser.add_argument("--top_k", type=int, default=5, help="The number of top predictions to display for each image.")
    parser.add_argument("--device", type=str, default="mps", help="The device to run the model on (e.g., 'cpu', 'cuda', 'mps').")

    args = parser.parse_args()

    predict_from_directory(args.weights, args.images_dir, args.top_k, args.device)


if __name__ == "__main__":
    main()
