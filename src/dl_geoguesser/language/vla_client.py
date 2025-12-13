"""
Client for interfacing with the VLA-PEFT inference server.
"""
import requests
from typing import Optional, Dict, Any


class VLAInferenceClient:
    """Client for making inference requests to the VLA-PEFT server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the VLA inference client.
        
        Args:
            base_url: Base URL of the inference server (e.g., "http://localhost:8000" or 
                     "https://t5uuas4ux32flb-8000.proxy.runpod.net")
        """
        self.base_url = base_url.rstrip('/')
        
    def generate_explanation(
        self,
        vision_data: Dict[str, Any],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a GeoGuessr explanation based on vision data.
        
        Args:
            vision_data: Dictionary containing vision model outputs with structure:
                {
                    "country": str,
                    "country_confidence": float,
                    "driving_side": str,
                    "driving_side_confidence": float,
                    "vibe_top": str,
                    "vibe_distribution": dict,
                    "evidence": {
                        "top_sign_countries": list[str],
                        "top_contents": list[str],
                        "gradcam_examples": list (optional)
                    }
                }
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter
            custom_prompt: Optional custom prompt to override default
            
        Returns:
            Generated explanation text
            
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        # Create the prompt from vision data
        import json
        vision_str = json.dumps(vision_data, ensure_ascii=False, indent=2)
        
        if custom_prompt:
            # Use custom prompt with vision data appended
            prompt = f"{custom_prompt}\n\nVision Data:\n{vision_str}\n\nResponse:"
        else:
            # Default GeoGuessr prompt
            prompt = (
                "You are an expert GeoGuessr player.\n"
                "You are given structured evidence extracted from a street-view image.\n"
                "Reason step by step about where this location could be, and explain why.\n\n"
                "EVIDENCE (JSON):\n"
                f"{vision_str}\n\n"
                "TASK:\n"
                "Write a short paragraph explaining:\n"
                "- Which country you think this is (best guess),\n"
                "- Any plausible alternative countries,\n"
                "- Why the driving side, architecture, vegetation, signs, and landmarks support your guess.\n"
                "Do NOT output JSON; answer in natural language.\n\n"
                "ANSWER:\n"
            )
        
        # Make request to the server
        response = requests.post(
            f"{self.base_url}/chat",
            json={
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            },
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        return result["assistant"]
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the server is healthy and ready.
        
        Returns:
            Server health status dictionary
        """
        response = requests.get(f"{self.base_url}/health", timeout=5)
        response.raise_for_status()
        return response.json()


# Example usage
if __name__ == "__main__":
    # For RunPod (replace with your actual URL)
    client = VLAInferenceClient("https://t5uuas4ux32flb-8000.proxy.runpod.net")
    
    # For local testing (if server is running locally)
    # client = VLAInferenceClient("http://localhost:8000")
    
    # Example vision data
    vision_data = {
        "country": "france",
        "country_confidence": 0.85,
        "driving_side": "right",
        "driving_side_confidence": 0.9,
        "vibe_top": "suburban residential area",
        "vibe_distribution": {
            "suburban residential area": 0.45,
            "urban city center": 0.25,
            "rural farmland": 0.15,
            "coastal town": 0.10,
            "mountain area": 0.05
        },
        "evidence": {
            "top_sign_countries": ["france", "belgium", "france"],
            "top_contents": ["architecture", "road signs"],
            "gradcam_examples": []
        }
    }
    
    # Check health
    try:
        health = client.health_check()
        print("Server health:", health)
    except Exception as e:
        print(f"Server not available: {e}")
        exit(1)
    
    # Generate explanation
    try:
        explanation = client.generate_explanation(vision_data, max_tokens=128)
        print("\nGenerated explanation:")
        print(explanation)
    except Exception as e:
        print(f"Error generating explanation: {e}")
