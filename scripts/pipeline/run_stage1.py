#!/usr/bin/env python3
"""
Run Stage 1 of the GeoLocation Pipeline.

This script demonstrates the parallel CLIP + YOLO → DINO pipeline.

Usage:
    python scripts/pipeline/run_stage1.py --image path/to/image.jpg
    python scripts/pipeline/run_stage1.py --images_dir path/to/images/
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from dl_geoguesser.vision.pipeline import (
    GeoLocationPipeline,
    PipelineConfig,
    save_visualizations,
)

# Default model weights (relative to project root)
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

DEFAULT_WEIGHTS = {
    "clip": ROOT_DIR / "runs" / "clip_vibe" / "clip_vibe_precomputed_run" / "best.pt",
    "yolo": ROOT_DIR / "runs" / "yolo" / "M1Pro_run" / "weights" / "best.pt",
    "dino": ROOT_DIR / "runs" / "dino" / "dino_precomputed" / "best.pt",
}


def print_result(result, top_k: int = 5):
    """Pretty print pipeline results."""
    print("\n" + "=" * 60)
    if result.image_path:
        print(f"Image: {Path(result.image_path).name}")
    print("=" * 60)
    
    # CLIP Vibe Results
    print("\n🎨 VIBE CLASSIFICATION (CLIP)")
    print("-" * 40)
    sorted_vibes = sorted(result.vibe_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (vibe, score) in enumerate(sorted_vibes[:top_k], 1):
        bar = "█" * int(score * 20)
        print(f"  {i}. {vibe:20s} {score:.3f} {bar}")
    
    # YOLO Detection Results
    print(f"\n🔍 OBJECT DETECTIONS (YOLO) - {result.num_detections} objects")
    print("-" * 40)
    if result.detections:
        for class_name, instances in result.detections.items():
            print(f"  • {class_name}: {len(instances)} detected")
            for inst in instances[:3]:  # Show max 3 per class
                conf = inst['confidence']
                bbox = inst['bbox_crop']
                print(f"      conf={conf:.2f}, bbox={bbox}")
    else:
        print("  No objects detected")
    
    # DINO Country Results
    print("\n🌍 COUNTRY PREDICTION (DINO)")
    print("-" * 40)
    sorted_countries = sorted(result.country_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (country, score) in enumerate(sorted_countries[:top_k], 1):
        bar = "█" * int(score * 20)
        print(f"  {i}. {country:20s} {score:.3f} {bar}")
    
    # OCR Results (if available)
    if result.ocr_results or result.extracted_text:
        print("\n📝 TEXT EXTRACTION (OCR)")
        print("-" * 40)
        if result.extracted_text:
            if result.detected_languages:
                print(f"  Languages: {', '.join(result.detected_languages)}")
            print(f"  Text: {result.extracted_text[:200]}")
            if len(result.extracted_text) > 200:
                print(f"        ... ({len(result.extracted_text)} chars total)")
        else:
            print("  No text detected in image")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run Stage 1 GeoLocation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a single image
    python scripts/pipeline/run_stage1.py --image test_images/photo.jpg
    
    # Process a directory of images
    python scripts/pipeline/run_stage1.py --images_dir test_images/
    
    # Use custom model weights
    python scripts/pipeline/run_stage1.py --image photo.jpg \\
        --clip_weights runs/clip_vibe/custom/best.pt \\
        --yolo_weights runs/yolo/custom/weights/best.pt \\
        --dino_weights runs/dino/custom/best.pt
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
    parser.add_argument(
        "--dino_weights", type=str, default=str(DEFAULT_WEIGHTS["dino"]),
        help=f"Path to DINO model weights. Default: {DEFAULT_WEIGHTS['dino']}"
    )
    
    # Pipeline settings
    parser.add_argument(
        "--device", type=str, default="mps",
        help="Device to run models on (cpu, cuda, mps). Default: mps"
    )
    parser.add_argument(
        "--yolo_conf", type=float, default=0.4,
        help="YOLO confidence threshold. Default: 0.4"
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of top predictions to show. Default: 5"
    )
    parser.add_argument(
        "--disable_ocr", action="store_true",
        help="Disable OCR text extraction"
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Optional: Save results to JSON file"
    )
    
    # Visualization options
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate and save visualizations (bounding boxes + heatmaps)"
    )
    parser.add_argument(
        "--viz_output_dir", type=str, default="visualizations",
        help="Directory to save visualizations. Default: visualizations"
    )
    
    args = parser.parse_args()
    
    # Create pipeline config
    config = PipelineConfig(
        clip_weights=args.clip_weights,
        yolo_weights=args.yolo_weights,
        dino_weights=args.dino_weights,
        device=args.device,
        yolo_conf_threshold=args.yolo_conf,
        top_k_results=args.top_k,
        enable_ocr=not args.disable_ocr,
    )
    
    # Initialize pipeline
    print("\n🚀 Initializing GeoLocation Pipeline (Stage 1)")
    print("-" * 50)
    pipeline = GeoLocationPipeline(config)
    pipeline.load_models()
    
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
    
    print(f"\n📷 Processing {len(image_paths)} image(s)...")
    
    # Process images
    all_results = []
    for image_path in image_paths:
        if not image_path.exists():
            print(f"⚠️  Image not found: {image_path}")
            continue
        
        try:
            result = pipeline.predict_from_path(str(image_path))
            print_result(result, top_k=args.top_k)
            all_results.append(result.to_dict())
            
            # Generate visualizations if requested
            if args.visualize:
                print(f"\n📊 Generating visualizations...")
                image = Image.open(image_path).convert("RGB")
                
                saved_files = save_visualizations(
                    image=image,
                    detections=result.detections,
                    output_dir=Path(args.viz_output_dir),
                    image_name=image_path.name,
                    clip_model=pipeline._clip,
                    top_class=result.top_vibe,
                    create_grid=True,
                )
                
                for viz_type, file_path in saved_files.items():
                    print(f"  ✓ Saved {viz_type}: {file_path}")
                
        except Exception as e:
            print(f"⚠️  Error processing {image_path}: {e}")
    
    # Save to JSON if requested
    if args.output_json and all_results:
        output_path = Path(args.output_json)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
    
    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
