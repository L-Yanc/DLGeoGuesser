import argparse
import json
import os
import random
import shutil
from pathlib import Path

from tqdm import tqdm

# --- CONFIGURATION ---

# The script is in scripts/, so the project root is one level up.
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw' / 'vistas2'
PROCESSED_DATA_DIR_BASE = ROOT_DIR / 'data' / 'processed'
CLASS_TRANSLATION_PATH = ROOT_DIR / 'data' / 'raw' / 'class_translation.json'

# Specific output directory for this dataset, inside 'processed'
PROCESSED_OUTPUT_DIR = PROCESSED_DATA_DIR_BASE / 'vistas_yolo'

# --- CORE FUNCTIONS ---


def load_class_translation(translation_path):
    """Loads the class translation file and creates mappings from the new format."""
    if not translation_path.exists():
        print(f"Error: Class translation file not found at {translation_path}")
        exit()
    with open(translation_path, 'r') as f:
        original_to_details = json.load(f)
    reverse_mapping = {
        original_class: details['name']
        for original_class, details in original_to_details.items()
    }
    new_class_names = sorted(list(set(reverse_mapping.values())))
    new_class_to_id = {name: i for i, name in enumerate(new_class_names)}
    class_mapping = {name: [] for name in new_class_names}
    for original_class, new_class_name in reverse_mapping.items():
        class_mapping[new_class_name].append(original_class)
    return class_mapping, reverse_mapping, new_class_to_id


def scan_and_cache_annotations(split, reverse_mapping):
    """
    Scans the dataset for relevant annotations and saves them to a cache file.
    If the cache file already exists, it loads from the cache instead.
    Cache is stored in the base processed dir.
    """
    cache_path = PROCESSED_DATA_DIR_BASE / f"{split}_vistas_cache.json"  # More specific cache name
    if cache_path.exists():
        print(f"Found existing cache file for '{split}' split. Loading from cache...")
        with open(cache_path, 'r') as f:
            return json.load(f)

    print(f"No cache found for '{split}' split. Scanning dataset...")
    panoptic_json_path = RAW_DATA_DIR / split / 'v2.0' / 'panoptic' / 'panoptic_2020.json'
    if not panoptic_json_path.exists():
        print(f"Error: Panoptic annotation file not found at {panoptic_json_path}")
        return None

    with open(panoptic_json_path) as f:
        panoptic_data = json.load(f)
    categories = {cat['id']: cat for cat in panoptic_data['categories']}
    images = {img['id']: img for img in panoptic_data['images']}
    image_data_cache = {}
    for ann in tqdm(panoptic_data['annotations'], desc=f"Scanning {split} annotations"):
        image_id = ann['image_id']
        image_info = images.get(image_id)
        if not image_info:
            continue
        img_filename, img_width, img_height = image_info['file_name'], image_info['width'], image_info['height']
        found_objects = []
        for segment in ann['segments_info']:
            category_id = segment['category_id']
            original_class_name = categories[category_id]['name']
            if original_class_name in reverse_mapping:
                new_class_name = reverse_mapping[original_class_name]
                x_min, y_min, w, h = segment['bbox']
                if w > 0 and h > 0:
                    found_objects.append({"new_class": new_class_name, "bbox_abs": [x_min, y_min, w, h]})
        if found_objects:
            image_data_cache[img_filename] = {"width": img_width, "height": img_height, "objects": found_objects}
    print(f"Scanning complete. Found {len(image_data_cache)} images with relevant objects.")
    PROCESSED_DATA_DIR_BASE.mkdir(exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(image_data_cache, f, indent=2)
    print(f"Saved cache to {cache_path}")
    return image_data_cache


def _process_and_copy_single_image(img_filename, data, source_split, dest_split_name, new_class_to_id):
    """Helper to process one image: create label file and copy image."""
    output_images_dir = PROCESSED_OUTPUT_DIR / dest_split_name / 'images'
    output_labels_dir = PROCESSED_OUTPUT_DIR / dest_split_name / 'labels'
    img_width, img_height = data['width'], data['height']
    yolo_label_path = output_labels_dir / f"{Path(img_filename).stem}.txt"
    with open(yolo_label_path, 'w') as f:
        for obj in data['objects']:
            class_id = new_class_to_id[obj['new_class']]
            x_min, y_min, w, h = obj['bbox_abs']
            x_center = (x_min + w / 2) / img_width
            y_center = (y_min + h / 2) / img_height
            norm_w = w / img_width
            norm_h = h / img_height
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
    original_image_path = RAW_DATA_DIR / source_split / 'images' / img_filename
    if original_image_path.exists():
        shutil.copy(original_image_path, output_images_dir / img_filename)
    else:
        print(f"Warning: Could not find original image {original_image_path}")


def create_yolo_files_full(source_split, image_data_cache, new_class_to_id):
    """Processes an entire split into a single corresponding output folder."""
    output_images_dir = PROCESSED_OUTPUT_DIR / source_split / 'images'
    if output_images_dir.exists() and any(output_images_dir.iterdir()):
        print(f"Output directory '{output_images_dir}' is populated. Skipping.")
        return
    print(f"Creating YOLO files for '{source_split}' split...")
    (PROCESSED_OUTPUT_DIR / source_split / 'images').mkdir(parents=True, exist_ok=True)
    (PROCESSED_OUTPUT_DIR / source_split / 'labels').mkdir(parents=True, exist_ok=True)
    for img_filename, data in tqdm(image_data_cache.items(), desc=f"Creating {source_split} files"):
        _process_and_copy_single_image(img_filename, data, source_split, source_split, new_class_to_id)
    print("YOLO file creation complete.")


def create_yolo_split_files(source_split, image_data_cache, new_class_to_id, args):
    """Deterministically splits the data into train, val, and test sets."""
    print("Splitting cached data into new train/val/test sets...")
    all_images = list(image_data_cache.keys())
    random.seed(args.seed)
    random.shuffle(all_images)
    total_requested = args.train_size + args.val_size + args.test_size
    if total_requested > len(all_images):
        print(f"Warning: Requested {total_requested} images, but only {len(all_images)} are available.")
    splits = {}
    start = 0
    if args.train_size > 0:
        end = start + args.train_size
        splits['train'] = all_images[start:end]
        start = end
    if args.val_size > 0:
        end = start + args.val_size
        splits['val'] = all_images[start:end]
        start = end
    if args.test_size > 0:
        end = start + args.test_size
        splits['test'] = all_images[start:end]
    for split_name, image_list in splits.items():
        print(f"Processing {len(image_list)} images for new '{split_name}' set...")
        output_images_dir = PROCESSED_OUTPUT_DIR / split_name / 'images'
        if output_images_dir.exists() and any(output_images_dir.iterdir()):
            print(f"Output directory '{output_images_dir}' is populated. Skipping this split.")
            continue
        (PROCESSED_OUTPUT_DIR / split_name / 'images').mkdir(parents=True, exist_ok=True)
        (PROCESSED_OUTPUT_DIR / split_name / 'labels').mkdir(parents=True, exist_ok=True)
        for img_filename in tqdm(image_list, desc=f"Creating {split_name} files"):
            data = image_data_cache[img_filename]
            _process_and_copy_single_image(img_filename, data, source_split, split_name, new_class_to_id)
    print("Sub-split processing complete.")


def main(args):
    """Main function to orchestrate the dataset preparation."""
    source_split = args.split
    use_sub_splitting = args.train_size is not None
    print("--- Starting YOLO Dataset Preparation ---")
    print(f"Processing source split: {source_split}")
    if use_sub_splitting:
        print(f"Mode: Sub-splitting into Train ({args.train_size}), Val ({args.val_size}), Test ({args.test_size})")
    else:
        print(f"Mode: Full split processing")
    _, reverse_mapping, new_class_to_id = load_class_translation(CLASS_TRANSLATION_PATH)
    image_data_cache = scan_and_cache_annotations(source_split, reverse_mapping)
    if not image_data_cache:
        print(f"Could not load or build cache for '{source_split}' split. Exiting.")
        return
    if use_sub_splitting:
        create_yolo_split_files(source_split, image_data_cache, new_class_to_id, args)
    else:
        create_yolo_files_full(source_split, image_data_cache, new_class_to_id)
    print(f"--- Dataset preparation for '{source_split}' split finished. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Mapillary Vistas dataset for YOLO training.")
    parser.add_argument(
        '--split', type=str, default='training', choices=['training', 'validation'],
        help='The source dataset split to process (default: training).'
    )
    parser.add_argument(
        '--train-size', type=int, default=None, help='Number of images for the new training set. Activates sub-splitting.'
    )
    parser.add_argument(
        '--val-size', type=int, default=0, help='Number of images for the new validation set.'
    )
    parser.add_argument(
        '--test-size', type=int, default=0, help='Number of images for the new test set.'
    )
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for deterministic shuffling.'
    )
    args = parser.parse_args()
    main(args)
