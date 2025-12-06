import argparse
import pprint
from pathlib import Path

import cv2
import numpy as np

from dl_geoguesser.vision.yolo_detector.model import YOLOv8Detector

# Define root directory relative to the script's location
ROOT_DIR = Path(__file__).parent.parent.resolve()


def find_latest_run_weights(project_dir: Path) -> str | None:
    """
    Finds the 'last.pt' file from the most recent YOLOv8 training run.
    """
    if not project_dir.exists():
        return None

    # Find all directories that seem to be training runs
    train_dirs = [d for d in project_dir.iterdir() if d.is_dir() and 'run' in d.name]
    if not train_dirs:
        return None

    # Get the most recently modified directory
    latest_dir = max(train_dirs, key=lambda p: p.stat().st_mtime)
    weights_path = latest_dir / 'weights' / 'last.pt'

    if weights_path.exists():
        print(f"Found latest weights at: {weights_path}")
        return str(weights_path)
    return None


def draw_predictions(image: np.ndarray, predictions: dict) -> np.ndarray:
    """
    Draws predicted bounding boxes and labels on an image.
    """
    class_names = sorted(predictions.keys())
    colors = {
        name: tuple(int(c) for c in color_val)
        for name, color_val in zip(class_names, (np.linspace(0, 255, len(class_names) * 2)[-len(class_names):, np.newaxis] * np.array([[1.8, 1.2, 1.5]])) % 255)
    }

    for class_name, instances in predictions.items():
        for instance in instances:
            x1, y1, x2, y2 = instance['bbox_crop']
            confidence = instance['confidence']
            color = colors.get(class_name, (0, 255, 0))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image


def main(args):
    """Main visualization loop for predictions."""
    weights_path = args.weights or find_latest_run_weights(ROOT_DIR / 'runs' / 'detect')
    if not weights_path:
        print("Error: Could not find any training runs or weights file specified.")
        print("Please train a model first or provide a path using --weights.")
        return

    detector = YOLOv8Detector(model_path=weights_path)

    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = ROOT_DIR / source_path

    if not source_path.exists():
        print(f"Error: Source path does not exist: {source_path}")
        return

    image_paths = sorted(list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))) if source_path.is_dir() else [source_path]

    if not image_paths:
        print(f"No images found in {source_path}")
        return

    print("\n--- YOLO Prediction Visualizer ---")
    print(f"Loaded model: {weights_path}")
    print(f"Found {len(image_paths)} images to predict.")
    print("\nControls:")
    print("  -> Right Arrow Key: Next Image")
    print("  <- Left Arrow Key:  Previous Image")
    print("  'q' or ESC: Quit")
    print("------------------------------------")

    current_index, window_name = 0, "YOLO Prediction Visualizer"
    while True:
        image_path = image_paths[current_index]
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not read image {image_path}, skipping.")
            current_index = (current_index + 1) % len(image_paths)
            continue

        predictions = detector.predict(image, conf=args.conf)
        vis_image = draw_predictions(image, predictions)

        filename_text = f"[{current_index + 1}/{len(image_paths)}] {image_path.name}"
        cv2.putText(vis_image, filename_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
        cv2.putText(vis_image, filename_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(window_name, vis_image)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key in [3, 83]:
            current_index = (current_index + 1) % len(image_paths)
        elif key in [2, 81]:
            current_index = (current_index - 1 + len(image_paths)) % len(image_paths)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    default_source = str(ROOT_DIR / 'data' / 'processed' / 'vistas_yolo' / 'val' / 'images')
    parser = argparse.ArgumentParser(description="Visualize YOLOv8 model predictions.")
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to model weights (.pt file). If not provided, will use the last checkpoint from the latest run."
    )
    parser.add_argument(
        "--source", type=str, default=default_source,
        help=f"Path to the source image or directory of images. Default: {default_source}"
    )
    parser.add_argument(
        "--conf", type=float, default=0.4,
        help="Confidence threshold for predictions."
    )
    args = parser.parse_args()
    main(args)
