"""
Run the GeoLocation pipeline with VLA fine-tuned model for explanations.

This script demonstrates how to integrate the vision pipeline with the
VLA (Vision-Language Assistant) fine-tuned LLaMA model for generating
natural language explanations of location predictions.

Usage:
    # Set VLA server URL (if not localhost:8000)
    export VLA_SERVER_URL="https://your-runpod-url.proxy.runpod.net"
    
    # Run on a single image
    python scripts/pipeline/run_with_vla.py path/to/image.jpg
    
    # Run on multiple images
    python scripts/pipeline/run_with_vla.py path/to/image1.jpg path/to/image2.jpg
"""

import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from PIL import Image
from src.dl_geoguesser.vision.pipeline import GeoLocationPipeline, PipelineConfig
from src.dl_geoguesser.language.vla_client import VLAInferenceClient


def format_vision_data_for_vla(result):
    """
    Convert PipelineResult to the format expected by VLA.
    
    Args:
        result: PipelineResult from vision pipeline
        
    Returns:
        Dict formatted for VLA inference
    """
    # Get top 5 countries and vibes
    top_countries = sorted(
        result.country_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    top_vibes = sorted(
        result.vibe_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    vision_data = {
        "country": result.top_country,
        "country_confidence": result.top_country_confidence,
        "vibe_top": result.top_vibe,
        "vibe_distribution": {name: score for name, score in top_vibes},
        "evidence": {
            "top_contents": list(result.detections.keys()),
            "detected_objects_count": result.num_detections,
        }
    }
    
    # Add OCR data if available
    if result.extracted_text:
        vision_data["evidence"]["detected_text"] = result.extracted_text
        vision_data["evidence"]["detected_languages"] = result.detected_languages
    
    # Add alternative country predictions
    if len(top_countries) > 1:
        vision_data["alternative_countries"] = [
            {"name": name, "confidence": score}
            for name, score in top_countries[1:4]
        ]
    
    return vision_data


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_with_vla.py <image_path> [image_path2 ...]")
        print("\nExample:")
        print("  python scripts/pipeline/run_with_vla.py test_images/sample.jpg")
        sys.exit(1)
    
    image_paths = sys.argv[1:]
    
    # Get VLA server URL from environment
    vla_server_url = os.environ.get("VLA_SERVER_URL", "http://localhost:8000")
    
    print("=" * 70)
    print("GeoLocation Pipeline with VLA Fine-tuned Model")
    print("=" * 70)
    print(f"\nVLA Server: {vla_server_url}")
    print(f"Images to process: {len(image_paths)}\n")
    
    # Initialize VLA client
    print("🌐 Connecting to VLA server...")
    vla_client = VLAInferenceClient(vla_server_url)
    
    try:
        health = vla_client.health_check()
        print(f"✅ VLA server connected: {health}\n")
    except Exception as e:
        print(f"❌ Failed to connect to VLA server: {e}")
        print("\nMake sure the VLA server is running:")
        print("  - Locally: python -m src.dl_geoguesser.language.vla_server.server")
        print("  - Or set VLA_SERVER_URL to your remote server")
        sys.exit(1)
    
    # Initialize vision pipeline
    print("🔧 Initializing vision pipeline...")
    config = PipelineConfig(
        clip_weights=str(ROOT_DIR / "runs" / "clip_vibe" / "clip_vibe_precomputed_run" / "best.pt"),
        yolo_weights=str(ROOT_DIR / "runs" / "yolo" / "M1Pro_run" / "weights" / "best.pt"),
        dino_weights=str(ROOT_DIR / "runs" / "dino" / "dino_precomputed" / "best.pt"),
        device="cpu",
        yolo_conf_threshold=0.4,
        enable_ocr=True,
    )
    pipeline = GeoLocationPipeline(config)
    pipeline.load_models()
    print("✅ Vision pipeline ready\n")
    
    # Process each image
    for i, image_path in enumerate(image_paths, 1):
        print("=" * 70)
        print(f"Processing image {i}/{len(image_paths)}: {image_path}")
        print("=" * 70)
        
        # Check if image exists
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}\n")
            continue
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        print(f"📸 Image size: {image.size}")
        
        # Run vision pipeline
        print("\n🔍 Running vision analysis...")
        result = pipeline.predict(image, image_path=image_path)
        
        # Display vision results
        print("\n📊 Vision Analysis Results:")
        print(f"  🌍 Country: {result.top_country} ({result.top_country_confidence:.1%})")
        print(f"  🏞️  Vibe: {result.top_vibe} ({result.top_vibe_confidence:.1%})")
        print(f"  🔍 Detections: {result.num_detections} objects")
        if result.detections:
            print(f"     Objects: {', '.join(list(result.detections.keys())[:5])}")
        if result.extracted_text:
            print(f"  📝 Text: {result.extracted_text[:100]}...")
            if result.detected_languages:
                print(f"     Languages: {', '.join(result.detected_languages)}")
        
        # Format data for VLA
        vision_data = format_vision_data_for_vla(result)
        
        # Generate explanation with VLA
        print("\n🤖 Generating explanation with VLA fine-tuned model...")
        try:
            explanation = vla_client.generate_explanation(
                vision_data=vision_data,
                max_tokens=150,
                temperature=0.7,
                top_p=0.9
            )
            
            print("\n" + "=" * 70)
            print("🎯 VLA EXPLANATION:")
            print("=" * 70)
            print(explanation)
            print("=" * 70)
            
        except Exception as e:
            print(f"❌ Failed to generate explanation: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print("\n✅ All images processed!")


if __name__ == "__main__":
    main()
