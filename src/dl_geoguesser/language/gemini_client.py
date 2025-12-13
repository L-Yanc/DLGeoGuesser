"""
Gemini API client for GeoGuesser explanations.

This module provides integration with Google's Gemini API for generating
high-quality GeoGuesser explanations using state-of-the-art models.
"""

import os
import json
from typing import Dict, Any, Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiClient:
    """
    Client for Google Gemini API.
    
    Provides a simple interface for generating GeoGuesser explanations
    using Gemini models (gemini-pro, gemini-1.5-pro, etc.).
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3-pro-preview"
    ):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google API key. If None, reads from GEMINI_API_KEY env var.
            model_name: Gemini model to use (default: gemini-3-pro-preview)
                       Options: gemini-3-pro-preview, gemini-2.5-flash
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )
        
        # Get API key
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. "
                "Set GEMINI_API_KEY environment variable or pass api_key parameter."
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        
        print(f"✅ Gemini client initialized with model: {model_name}")
    
    def generate_explanation(
        self,
        vision_data: Dict[str, Any],
        max_tokens: int = 10000,
        temperature: float = 0.7,
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a GeoGuessr explanation based on vision data.
        
        Args:
            vision_data: Dictionary containing vision model outputs
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            custom_prompt: Optional custom prompt to override default
            
        Returns:
            Generated explanation text
        """
        # Create the prompt from vision data
        vision_str = json.dumps(vision_data, ensure_ascii=False, indent=2)
        
        if custom_prompt:
            prompt = f"{custom_prompt}\n\nVision Data:\n{vision_str}\n\nResponse:"
        else:
            # Default GeoGuessr prompt
            prompt = (
                "You are an expert GeoGuessr player analyzing a street-view image.\n\n"
                "EVIDENCE FROM COMPUTER VISION MODELS:\n"
                f"{vision_str}\n\n"
                "TASK:\n"
                "Based on this evidence, write a concise paragraph (2-3 sentences) explaining:\n"
                "1. Which country you think this is (your best guess)\n"
                "2. Key evidence supporting your guess (architecture, signs, vegetation, etc.)\n"
                "3. Any alternative possibilities\n\n"
                "Be specific and confident. Focus on the most distinctive clues.\n\n"
                "EXPLANATION:"
            )
        
        # Generate with Gemini
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Configure safety settings to be less restrictive
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Handle blocked responses
        if not response.candidates:
            return "Unable to generate response (blocked by safety filters)"
        
        candidate = response.candidates[0]
        
        # Extract text - try multiple methods
        text = ""
        try:
            # Method 1: Direct response.text (works for normal responses)
            text = response.text
        except:
            try:
                # Method 2: From candidate parts
                if candidate.content and candidate.content.parts:
                    text = candidate.content.parts[0].text
            except Exception as e:
                print(f"⚠️  Failed to extract text: {e}")
                print(f"   Finish reason: {candidate.finish_reason}")
                print(f"   Has content: {candidate.content is not None}")
        
        # Check finish reason
        if candidate.finish_reason != 1:  # 1 = STOP (normal completion)
            # finish_reason: 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
            if candidate.finish_reason == 2:
                # Response was cut off at max tokens - return what we have anyway
                if text:
                    print(f"⚠️  Response truncated at max_tokens, returning {len(text)} chars")
                    return text.strip()
                else:
                    return "Response truncated (no content generated)"
            elif candidate.finish_reason == 3:
                return "Unable to generate response (blocked by safety filters)"
            else:
                return f"Unable to generate response (finish_reason: {candidate.finish_reason})"
        
        # Normal response
        return text.strip()
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 10000,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text from a raw prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Configure safety settings to be less restrictive
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Handle blocked responses
        if not response.candidates:
            return "Unable to generate response (blocked by safety filters)"
        
        candidate = response.candidates[0]
        
        # Extract text - try multiple methods
        text = ""
        try:
            # Method 1: Direct response.text (works for normal responses)
            text = response.text
        except:
            try:
                # Method 2: From candidate parts
                if candidate.content and candidate.content.parts:
                    text = candidate.content.parts[0].text
            except Exception as e:
                print(f"⚠️  Failed to extract text: {e}")
                print(f"   Finish reason: {candidate.finish_reason}")
                print(f"   Has content: {candidate.content is not None}")
        
        # Check finish reason
        if candidate.finish_reason != 1:  # 1 = STOP (normal completion)
            if candidate.finish_reason == 2:
                # Response was cut off at max tokens - return what we have anyway
                if text:
                    print(f"⚠️  Response truncated at max_tokens, returning {len(text)} chars")
                    return text.strip()
                else:
                    return "Response truncated (no content generated)"
            elif candidate.finish_reason == 3:
                return "Unable to generate response (blocked by safety filters)"
            else:
                return f"Unable to generate response (finish_reason: {candidate.finish_reason})"
        
        # Normal response
        return text.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model": self.model_name,
            "provider": "Google Gemini",
            "api_key_set": bool(self.api_key),
        }
    
    def __repr__(self):
        """String representation."""
        return f"GeminiClient(model={self.model_name})"


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Check if API key is set
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        print("\nSet it with:")
        print("  export GEMINI_API_KEY='your_api_key_here'")
        print("\nGet your API key from:")
        print("  https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    # Initialize client
    client = GeminiClient()
    
    # Example vision data
    vision_data = {
        "country": "france",
        "country_confidence": 0.85,
        "vibe_top": "suburban residential area",
        "vibe_distribution": {
            "suburban residential area": 0.45,
            "urban city center": 0.25,
        },
        "evidence": {
            "top_contents": ["road_sign", "architecture"],
            "detected_text": "Rue de la Paix",
        }
    }
    
    # Generate explanation
    print("\n" + "="*60)
    print("Generating GeoGuesser explanation with Gemini...")
    print("="*60)
    explanation = client.generate_explanation(vision_data, max_tokens=10000)
    print(explanation)
