"""
Test script for VLA client connection and inference.

This script tests the VLA client's ability to connect to the server
and generate GeoGuesser explanations.

Usage:
    # Test with default localhost
    python scripts/test_vla_client.py
    
    # Test with custom server URL
    VLA_SERVER_URL="https://your-server.com" python scripts/test_vla_client.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.dl_geoguesser.language.vla_client import VLAInferenceClient


def test_health_check(client):
    """Test server health check."""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    try:
        health = client.health_check()
        print("✅ Server is healthy!")
        print(f"   Device: {health.get('device', 'unknown')}")
        print(f"   Model: {health.get('model', 'unknown')}")
        print(f"   LoRA: {health.get('lora_checkpoint', 'not loaded')}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_simple_generation(client):
    """Test simple text generation."""
    print("\n" + "="*60)
    print("TEST 2: Simple Text Generation")
    print("="*60)
    
    try:
        # Simple prompt
        vision_data = {"prompt": "Hello, how are you?"}
        response = client.generate_explanation(
            vision_data=vision_data,
            max_tokens=50,
            temperature=0.7,
            custom_prompt="Hello, how are you?"
        )
        
        print("✅ Generation successful!")
        print(f"   Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_geoguesser_explanation(client):
    """Test GeoGuesser explanation generation."""
    print("\n" + "="*60)
    print("TEST 3: GeoGuesser Explanation")
    print("="*60)
    
    # Example vision data
    vision_data = {
        "country": "france",
        "country_confidence": 0.85,
        "vibe_top": "suburban residential area",
        "vibe_distribution": {
            "suburban residential area": 0.45,
            "urban city center": 0.25,
            "rural farmland": 0.15,
        },
        "evidence": {
            "top_contents": ["road_sign", "architecture", "vegetation"],
            "detected_text": "Rue de la Paix",
            "detected_languages": ["fr"],
        }
    }
    
    print("\nInput vision data:")
    print(f"  Country: {vision_data['country']} ({vision_data['country_confidence']:.0%})")
    print(f"  Vibe: {vision_data['vibe_top']}")
    print(f"  Objects: {', '.join(vision_data['evidence']['top_contents'])}")
    print(f"  Text: {vision_data['evidence']['detected_text']}")
    
    try:
        response = client.generate_explanation(
            vision_data=vision_data,
            max_tokens=128,
            temperature=0.7,
            top_p=0.9
        )
        
        print("\n✅ Explanation generated!")
        print("\n" + "-"*60)
        print(response)
        print("-"*60)
        return True
    except Exception as e:
        print(f"❌ Explanation generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Get server URL from environment
    server_url = os.environ.get("VLA_SERVER_URL", "http://localhost:8000")
    
    print("="*60)
    print("VLA Client Test Suite")
    print("="*60)
    print(f"\nServer URL: {server_url}")
    print("\nThis script will test:")
    print("  1. Server health check")
    print("  2. Simple text generation")
    print("  3. GeoGuesser explanation generation")
    
    # Initialize client
    print("\n🌐 Initializing VLA client...")
    client = VLAInferenceClient(server_url)
    
    # Run tests
    results = []
    results.append(("Health Check", test_health_check(client)))
    
    if results[0][1]:  # Only continue if health check passed
        results.append(("Simple Generation", test_simple_generation(client)))
        results.append(("GeoGuesser Explanation", test_geoguesser_explanation(client)))
    else:
        print("\n⚠️  Skipping remaining tests due to health check failure")
        print("\nTroubleshooting:")
        print("  1. Make sure VLA server is running:")
        print("     python -m src.dl_geoguesser.language.vla_server.server")
        print("  2. Check server URL is correct:")
        print(f"     Current: {server_url}")
        print("  3. For remote server, set VLA_SERVER_URL environment variable")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! VLA client is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
