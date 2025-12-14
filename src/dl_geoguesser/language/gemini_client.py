"""
Gemini API client for GeoGuesser explanations.

This module provides integration with Google's Gemini API for generating
high-quality GeoGuesser explanations using state-of-the-art models.

Supports both the new google.genai SDK (for Gemini 3) and legacy google.generativeai.
"""

import os
import json
import re
from typing import Dict, Any, Optional

# Try new SDK first (for Gemini 3), fall back to legacy
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
    USE_NEW_SDK = True
except ImportError:
    try:
        import google.generativeai as genai_legacy
        GEMINI_AVAILABLE = True
        USE_NEW_SDK = False
    except ImportError:
        GEMINI_AVAILABLE = False
        USE_NEW_SDK = False


def format_markdown_to_html(text: str) -> str:
    """
    Convert markdown formatting to HTML.
    
    Handles:
    - **bold** -> <strong>bold</strong>
    - *italic* -> <em>italic</em>
    - `code` -> <code>code</code>
    """
    if not text:
        return text
    
    # Bold: **text** -> <strong>text</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    
    # Italic: *text* (but not already part of bold) -> <em>text</em>
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    
    # Inline code: `text` -> <code>text</code>
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    
    return text


class GeminiClient:
    """
    Client for Google Gemini API.
    
    Supports Gemini 3 Pro (new SDK) and Gemini 1.5 models.
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
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Gemini SDK not installed. "
                "Install with: pip install google-genai"
            )
        
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. "
                "Set GEMINI_API_KEY environment variable or pass api_key parameter."
            )
        
        self.model_name = model_name
        self.use_new_sdk = USE_NEW_SDK
        
        if self.use_new_sdk:
            # New SDK for Gemini 3
            self.client = genai.Client(api_key=self.api_key)
        else:
            # Legacy SDK
            genai_legacy.configure(api_key=self.api_key)
            self.model = genai_legacy.GenerativeModel(model_name)
        
        print(f"✅ Gemini client initialized with model: {model_name} (new_sdk={self.use_new_sdk})")
    
    def generate_explanation(
        self,
        vision_data: Dict[str, Any],
        max_tokens: int = None,
        temperature: float = 1.0,
        custom_prompt: Optional[str] = None,
        format_html: bool = True
    ) -> str:
        """
        Generate a GeoGuessr explanation based on vision data.
        
        Args:
            vision_data: Dictionary containing vision model outputs
            max_tokens: Maximum tokens to generate (None = no limit)
            temperature: Sampling temperature (1.0 recommended for Gemini 3)
            custom_prompt: Optional custom prompt to override default
            format_html: Convert markdown formatting to HTML (default: True)
        """
        vision_str = json.dumps(vision_data, ensure_ascii=False, indent=2)
        
        if custom_prompt:
            prompt = f"{custom_prompt}\n\nVision Data:\n{vision_str}\n\nResponse:"
        else:
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
        
        response = self._generate(prompt, max_tokens, temperature)
        if format_html and response:
            response = format_markdown_to_html(response)
        return response
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = 1.0,
        format_html: bool = True
    ) -> str:
        """
        Generate text from a raw prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate (None = no limit)
            temperature: Sampling temperature (1.0 recommended for Gemini 3)
            format_html: Convert markdown formatting to HTML (default: True)
        """
        response = self._generate(prompt, max_tokens, temperature)
        if format_html and response:
            response = format_markdown_to_html(response)
        return response
    
    def _generate(self, prompt: str, max_tokens: int = None, temperature: float = 1.0) -> str:
        """Internal generation method."""
        if self.use_new_sdk:
            return self._generate_new_sdk(prompt, max_tokens, temperature)
        else:
            return self._generate_legacy_sdk(prompt, max_tokens, temperature)

    def _generate_new_sdk(self, prompt: str, max_tokens: int = None, temperature: float = 1.0) -> str:
        """Generate using new google.genai SDK (Gemini 3)."""
        try:
            # Build config following Gemini 3 recommendations
            config_kwargs = {}
            
            # Temperature - Gemini 3 recommends 1.0 default
            if temperature != 1.0:
                config_kwargs["temperature"] = temperature
            
            # Max tokens - Gemini 3 Pro supports up to 64k output tokens
            # Set to a high default if None to avoid truncation
            if max_tokens is None:
                config_kwargs["max_output_tokens"] = 8024 # Reasonable default (64k max available)
            elif max_tokens > 0:
                config_kwargs["max_output_tokens"] = min(max_tokens, 65536)  # Cap at 64k
            # If max_tokens is 0 or negative, don't set it (use API default)
            
            print(f"🔍 Config kwargs: {config_kwargs}")
            print(f"🔍 Prompt length: {len(prompt)} chars")
            
            # Create config
            config = types.GenerateContentConfig(**config_kwargs)
            
            # Generate content
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            print(f"🔍 Response received, has candidates: {hasattr(response, 'candidates') and bool(response.candidates)}")
            
            # Extract text from response
            # According to Gemini 3 docs, response.text should work directly
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
            
            # Fallback: Try response.parts
            if hasattr(response, 'parts') and response.parts:
                text_parts = []
                for part in response.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                if text_parts:
                    return ' '.join(text_parts).strip()
            
            # Fallback: Extract from candidates using model_dump()
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    try:
                        content_dict = candidate.content.model_dump() if hasattr(candidate.content, 'model_dump') else candidate.content.dict()
                        print(f"🔍 Content dict: {content_dict}")
                        
                        if content_dict and 'parts' in content_dict:
                            parts_list = content_dict['parts']
                            print(f"🔍 Parts list: {parts_list}, type: {type(parts_list)}")
                            
                            if parts_list:
                                text_parts = []
                                for i, part in enumerate(parts_list):
                                    print(f"🔍 Part {i}: {part}, type: {type(part)}")
                                    if isinstance(part, dict) and 'text' in part and part['text']:
                                        text_parts.append(part['text'])
                                    elif hasattr(part, 'text') and part.text:
                                        text_parts.append(part.text)
                                
                                if text_parts:
                                    print(f"✅ Extracted {len(text_parts)} text parts")
                                    return ' '.join(text_parts).strip()
                                else:
                                    print(f"⚠️  No text found in {len(parts_list)} parts")
                    except Exception as e:
                        print(f"⚠️  Error extracting from content dict: {e}")
                        import traceback
                        traceback.print_exc()
            
            # If we get here, no text was found
            finish_reason = "unknown"
            if hasattr(response, 'candidates') and response.candidates:
                finish_reason = getattr(response.candidates[0], 'finish_reason', 'unknown')
            
            print(f"⚠️  No text found in response (finish_reason: {finish_reason})")
            return "No response generated"
            
        except Exception as e:
            print(f"⚠️  Gemini generation error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def _generate_legacy_sdk(self, prompt: str, max_tokens: int = None, temperature: float = 1.0) -> str:
        """Generate using legacy google.generativeai SDK."""
        try:
            config_params = {"temperature": temperature}
            if max_tokens is not None:
                config_params["max_output_tokens"] = max_tokens
            generation_config = genai_legacy.types.GenerationConfig(**config_params)
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            if response.text:
                return response.text.strip()
            
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    return candidate.content.parts[0].text.strip()
            
            return "No response generated"
            
        except Exception as e:
            print(f"⚠️  Gemini generation error: {e}")
            return f"Error: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model": self.model_name,
            "provider": "Google Gemini",
            "api_key_set": bool(self.api_key),
            "new_sdk": self.use_new_sdk,
        }
    
    def __repr__(self):
        return f"GeminiClient(model={self.model_name})"


if __name__ == "__main__":
    import sys
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        sys.exit(1)
    
    client = GeminiClient()
    
    # Test simple generation
    print("\n" + "="*60)
    print("Testing Gemini 3 Pro...")
    print("="*60)
    response = client.generate("What is 2+2? Answer briefly.")
    print(f"Response: {response}")
