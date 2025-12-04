
import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse

# --- CONFIGURATION ---

# The script is in scripts/, so the project root is one level up.
ROOT_DIR = Path(__file__).parent.parent.resolve()
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed' / 'vistas_yolo'
CLASS_TRANSLATION_PATH = ROOT_DIR / 'data' / 'raw' / 'class_translation.json'

# --- CORE FUNCTIONS ---

def get_class_names():
    """Loads class translation to get an ordered list of new class names."""
    if not CLASS_TRANSLATION_PATH.exists():
        print(f"Error: Class translation file not found at {CLASS_TRANSLATION_PATH}")
        print("Please ensure the file was moved to 'data/raw/'.")
        exit()

    with open(CLASS_TRANSLATION_PATH, 'r') as f:
        original_to_details = json.load(f)

    # Get a sorted, unique list of the new class names to match the class IDs
    new_class_names = sorted(list(set(details['name'] for details in original_to_details.values())))
    return new_class_names


def draw_yolo_bboxes(image_path, class_names):
    """
    Loads an image and its corresponding YOLO label file, then draws the
    bounding boxes and labels on the image.
    """
    label_path = image_path.parent.parent / 'labels' / f"{image_path.stem}.txt"
    
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return None
    h, w, _ = image.shape

    if not label_path.exists():
        return image

    # Generate a unique color for each class
    colors = [
        (int(c[0]), int(c[1]), int(c[2]))
        for c in (np.linspace(0, 255, len(class_names) * 2)[-len(class_names):, np.newaxis] * np.array([[1.5, 0.8, 1.2]])) % 255
    ]

    with open(label_path, 'r') as f:
        for line in f:
            try:
                class_id, x_center, y_center, norm_w, norm_h = map(float, line.strip().split())
                class_id = int(class_id)
            except ValueError:
                print(f"Warning: Skipping malformed line in {label_path}: '{line.strip()}'")
                continue

            box_w, box_h = norm_w * w, norm_h * h
            x1, y1 = int((x_center * w) - (box_w / 2)), int((y_center * h) - (box_h / 2))
            x2, y2 = int(x1 + box_w), int(y1 + box_h)

            class_name = class_names[class_id]
            color = colors[class_id]

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
    return image


def main(args):
    """Main visualization loop."""
    split = args.split
    
    if not PROCESSED_DATA_DIR.exists():
        print(f"Error: Processed data root directory not found at '{PROCESSED_DATA_DIR}'")
        print("Please run the 'prepare_vistas_yolo.py' script first.")
        return

    class_names = get_class_names()
    images_dir = PROCESSED_DATA_DIR / split / 'images'
    
    if not images_dir.exists() or not any(images_dir.iterdir()):
        print(f"Error: Processed images directory for split '{split}' is empty or not found.")
        print(f"Looked in: {images_dir}")
        print("Please ensure the preparation script was run correctly for this split.")
        return

    image_paths = sorted(list(images_dir.glob('*.jpg')))
    if not image_paths:
        print(f"No .jpg images found in {images_dir}")
        return

    print("--- YOLO Bounding Box Visualizer ---")
    print(f"Visualizing split: '{split}' ({len(image_paths)} images).")
    print("  -> Right Arrow Key: Next Image")
    print("  <- Left Arrow Key:  Previous Image")
    print("  'q' or ESC: Quit")

    current_index, window_name = 0, "YOLO BBox Visualizer"
    while True:
        image_path = image_paths[current_index]
        vis_image = draw_yolo_bboxes(image_path, class_names)
        
        filename_text = f"[{current_index + 1}/{len(image_paths)}] {image_path.name}"
        cv2.putText(vis_image, filename_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
        cv2.putText(vis_image, filename_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(window_name, vis_image)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q') or key == 27: break
        elif key in [3, 83]: current_index = (current_index + 1) % len(image_paths)
        elif key in [2, 81]: current_index = (current_index - 1 + len(image_paths)) % len(image_paths)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize processed YOLO annotations.")
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help="The processed dataset split to visualize (e.g., 'train', 'val', 'test')."
    )
    args = parser.parse_args()
    
    if 'DISPLAY' not in os.environ and 'WAYLAND_DISPLAY' not in os.environ:
         print("Warning: No display environment found. The script might fail if not run in a graphical session.")
         print("This is normal in some remote environments (like SSH without -X).")

    main(args)
