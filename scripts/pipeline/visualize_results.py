#!/usr/bin/env python3
"""
Standalone script to visualize pipeline results.

This script can be used to generate visualizations from already-processed images
or to visualize specific detections.

Usage:
    # Visualize with full pipeline
    python scripts/pipeline/visualize_results.py --image photo.jpg --clip_weights runs/.../best.pt
    
    # Just show bounding boxes (no CLIP needed)
    python scripts/pipeline/visualize_results.py --image photo.jpg --yolo_weights runs/.../best.pt --bbox_only
"""

import argparse
from pathlib import Path

from PIL import Image

from dl_geoguesser.vision.clip_vibe.model import ClipVibe
from dl_geoguesser.vision.yolo_detector.model import YOLOv8Detector
from dl_geoguesser.vision.pipeline.visualize import (
    draw_bounding_boxes,
    generate_clip_heatmap,
    create_result_grid,
    save_visualizations,
)

# Default model weights
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

DEFAULT_WEIGHTS = {
    "clip": ROOT_DIR / "runs" / "clip_vibe" / "clip_vibe_precomputed_run" / "best.pt",
    "yolo": ROOT_DIR / "runs" / "yolo" / "M1Pro_run" / "weights" / "best.pt",
}


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GeoLocation Pipeline Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full visualization with heatmap
    python scripts/pipeline/visualize_results.py --image photo.jpg
    
    # Just bounding boxes (faster, no CLIP loading)
    python scripts/pipeline/visualize_results.py --image photo.jpg --bbox_only
    
    # Custom output directory
    python scripts/pipeline/visualize_results.py --image photo.jpg --output_dir my_viz
    
    # Process directory of images
    python scripts/pipeline/visualize_results.py --images_dir test_images/ --bbox_only
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", type=str,
        help="Path to a single image file"
    )
    input_group.add_argument(
        "--images_dir", type=str,
        help="Path to a directory of images"
    )
    
    # Model weights
    parser.add_argument(
        "--clip_weights", type=str, default=str(DEFAULT_WEIGHTS["clip"]),
        help=f"Path to CLIP model weights. Default: {DEFAULT_WEIGHTS['clip']}"
    )
    parser.add_argument(
        "--yolo_weights", type=str, default=str(DEFAULT_WEIGHTS["yolo"]),
        help=f"Path to YOLO model weights. Default: {DEFAULT_WEIGHTS['yolo']}"
    )
    
    # Visualization options
    parser.add_argument(
        "--bbox_only", action="store_true",
        help="Only draw bounding boxes (skip CLIP heatmap generation)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="visualizations",
        help="Directory to save visualizations. Default: visualizations"
    )
    parser.add_argument(
        "--yolo_conf", type=float, default=0.4,
        help="YOLO confidence threshold. Default: 0.4"
    )
    parser.add_argument(
        "--heatmap_alpha", type=float, default=0.5,
        help="Heatmap overlay transparency (0.0-1.0). Default: 0.5"
    )
    parser.add_argument(
        "--device", type=str, default="mps",
        help="Device to run models on (cpu, cuda, mps). Default: mps"
    )
    
    args = parser.parse_args()
    
    # Collect images to process
    if args.image:
        image_paths = [Path(args.image)]
    else:
        images_dir = Path(args.images_dir)
        image_paths = sorted(
            list(images_dir.glob("*.jpg")) + 
            list(images_dir.glob("*.png")) +
            list(images_dir.glob("*.jpeg"))
        )
    
    if not image_paths:
        print("No images found to process.")
        return
    
    print("\n🎨 Initializing Visualization Pipeline")
    print("-" * 50)
    
    # Load YOLO
    print("Loading YOLO detector...")
    yolo = YOLOv8Detector(model_path=args.yolo_weights)
    
    # Load CLIP if needed
    clip_model = None
    if not args.bbox_only:
        print("Loading CLIP model...")
        clip_model = ClipVibe(weights_path=args.clip_weights, device=args.device)
    
    print(f"\n📷 Processing {len(image_paths)} image(s)...")
    print()
    
    output_dir = Path(args.output_dir)
    
    for idx, image_path in enumerate(image_paths, 1):
        if not image_path.exists():
            print(f"[{idx}/{len(image_paths)}] ⚠️  Image not found: {image_path}")
            continue
        
        print(f"[{idx}/{len(image_paths)}] Processing: {image_path.name}")
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Run YOLO detection
            detections = yolo.predict(image, conf=args.yolo_conf)
            num_detections = sum(len(v) for v in detections.values())
            print(f"  Detected {num_detections} objects")
            
            # Get top class for heatmap
            top_class = None
            if clip_model:
                vibe_scores = clip_model.predict(image)
                top_class = max(vibe_scores, key=vibe_scores.get) if vibe_scores else None
                if top_class:
                    print(f"  Top vibe: {top_class} ({vibe_scores[top_class]:.3f})")
            
            # Save visualizations
            saved_files = save_visualizations(
                image=image,
                detections=detections,
                output_dir=output_dir,
                image_name=image_path.name,
                clip_model=clip_model,
                top_class=top_class,
                create_grid=not args.bbox_only,
            )
            
            for viz_type, file_path in saved_files.items():
                print(f"  ✓ Saved {viz_type}: {file_path.name}")
            
            print()
            
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            print()
    
    print(f"✅ Visualization complete!")
    print(f"   Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
