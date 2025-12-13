"""
Stage 1 of the GeoLocation Pipeline.

This module implements the first stage of image processing:
1. CLIP (ClipVibe) - analyzes full image for "vibe" classification
2. YOLO (YOLOv8Detector) - detects objects/regions of interest
3. DINO (DinoGeoguesser) - analyzes YOLO-detected regions for country prediction

CLIP and YOLO run in parallel, then YOLO results feed into DINO.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from PIL import Image

from dl_geoguesser.vision.clip_vibe.model import ClipVibe
from dl_geoguesser.vision.dino_geoguesser.model import DinoGeoguesser
from dl_geoguesser.vision.yolo_detector.model import YOLOv8Detector

# Optional OCR support
try:
    from dl_geoguesser.vision.ocr_pipeline.model import MultiLangOCR
    from dl_geoguesser.vision.ocr_pipeline.main import process_yolo_predictions
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    MultiLangOCR = None
    process_yolo_predictions = None


@dataclass
class PipelineConfig:
    """Configuration for the GeoLocation pipeline."""
    clip_weights: str
    yolo_weights: str
    dino_weights: str
    device: str = "mps"
    yolo_conf_threshold: float = 0.35
    top_k_results: int = 5
    enable_ocr: bool = True  # Enable OCR text extraction


@dataclass
class PipelineResult:
    """Results from the Stage 1 pipeline."""
    # CLIP vibe classification scores
    vibe_scores: Dict[str, float] = field(default_factory=dict)
    top_vibe: str = ""
    top_vibe_confidence: float = 0.0
    
    # YOLO detections
    detections: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    num_detections: int = 0
    
    # DINO country predictions (from YOLO regions)
    country_scores: Dict[str, float] = field(default_factory=dict)
    top_country: str = ""
    top_country_confidence: float = 0.0
    
    # OCR text extraction (optional)
    ocr_results: Dict[str, Any] = field(default_factory=dict)
    detected_languages: List[str] = field(default_factory=list)
    extracted_text: str = ""
    
    # Metadata
    image_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary."""
        result = {
            "vibe": {
                "scores": self.vibe_scores,
                "top": self.top_vibe,
                "confidence": self.top_vibe_confidence,
            },
            "detections": {
                "objects": self.detections,
                "count": self.num_detections,
            },
            "country": {
                "scores": self.country_scores,
                "top": self.top_country,
                "confidence": self.top_country_confidence,
            },
            "image_path": self.image_path,
        }
        
        if self.ocr_results:
            result["ocr"] = {
                "detected_languages": self.detected_languages,
                "extracted_text": self.extracted_text,
                "raw_results": self.ocr_results,
            }
        
        return result


class GeoLocationPipeline:
    """
    Stage 1 GeoLocation Pipeline.
    
    Processes images through CLIP, YOLO, DINO, and optionally OCR models 
    to predict location-related information.
    
    Architecture:
        Image ─┬─> [CLIP]  ─────────────────────────> vibe_scores
               │
               ├─> [YOLO] ─┬─> detections ─> [DINO] ─> country_scores
               │           │
               │           └─> text regions ─> [OCR] ─> extracted_text (optional)
    
    CLIP and YOLO run in parallel for efficiency.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline with model weights.
        
        Args:
            config: PipelineConfig with paths to model weights and settings.
        """
        self.config = config
        self._models_loaded = False
        
        # Models will be lazily loaded
        self._clip: Optional[ClipVibe] = None
        self._yolo: Optional[YOLOv8Detector] = None
        self._dino: Optional[DinoGeoguesser] = None
        self._ocr: Optional[MultiLangOCR] = None
    
    def load_models(self) -> None:
        """Load all models into memory."""
        if self._models_loaded:
            return
            
        print("Loading pipeline models...")
        
        # Load models (could parallelize this too if needed)
        print("  Loading CLIP (ClipVibe)...")
        self._clip = ClipVibe(
            weights_path=self.config.clip_weights,
            device=self.config.device
        )
        
        print("  Loading YOLO...")
        self._yolo = YOLOv8Detector(model_path=self.config.yolo_weights)
        
        print("  Loading DINO (DinoGeoguesser)...")
        self._dino = DinoGeoguesser(
            weights_path=self.config.dino_weights,
            device=self.config.device
        )
        
        # Load OCR if enabled
        if self.config.enable_ocr:
            if not OCR_AVAILABLE:
                print("  ⚠️  OCR requested but dependencies not available (easyocr, langdetect)")
                print("     Install with: pip install easyocr langdetect")
            else:
                print("  Loading OCR (MultiLangOCR) - this may take a while...")
                try:
                    self._ocr = MultiLangOCR()
                    # Check if any readers were successfully loaded
                    if not self._ocr.readers:
                        print("  ⚠️  OCR failed to load any language models")
                        self._ocr = None
                    else:
                        print(f"  OCR loaded successfully with {len(self._ocr.readers)} language families")
                except Exception as e:
                    print(f"  ⚠️  OCR initialization failed: {e}")
                    self._ocr = None
        
        self._models_loaded = True
        print("All models loaded successfully.")
    
    def _run_clip(self, image: Image.Image) -> Dict[str, float]:
        """Run CLIP model on the image."""
        return self._clip.predict(image)
    
    def _run_yolo(self, image: Image.Image) -> Dict[str, List[Dict]]:
        """Run YOLO model on the image."""
        return self._yolo.predict(image, conf=self.config.yolo_conf_threshold)
    
    def _run_dino(self, image: Image.Image, detections: Dict) -> Dict[str, float]:
        """Run DINO model on YOLO-detected regions."""
        if not detections:
            # If no detections, use full image as fallback
            detections = {
                "full_image": [{
                    "bbox_crop": (0, 0, image.width, image.height),
                    "confidence": 1.0
                }]
            }
        return self._dino.predict_from_crops(image, detections)
    
    def _run_ocr(self, image: Image.Image, detections: Dict) -> Dict[str, Any]:
        """Run OCR on text-containing regions detected by YOLO."""
        if not self._ocr or not process_yolo_predictions:
            print("  OCR: No OCR model available")
            return {}
        
        if not detections:
            print("  OCR: No detections provided")
            return {}
        
        # Convert PIL to numpy
        image_np = np.array(image)
        
        # Use the proper process_yolo_predictions function from ocr_pipeline
        print(f"  OCR: Processing YOLO detections with process_yolo_predictions")
        ocr_results = process_yolo_predictions(detections, image_np)
        
        print(f"  OCR: Found {len(ocr_results)} text regions with content")
        
        # Extract text and languages from results
        # ocr_results structure: {(x1,y1,x2,y2): {"text": str, "languages": [{"lang": str, "confidence": float}]}}
        extracted_texts = []
        detected_langs = set()
        
        for bbox, result in ocr_results.items():
            text = result.get("text", "")
            if text.strip():
                extracted_texts.append(text)
                print(f"  OCR: Found text: '{text[:50]}...'")
            
            # Get language codes
            for lang_info in result.get("languages", []):
                lang_code = lang_info.get("lang")
                if lang_code:
                    detected_langs.add(lang_code)
        
        return {
            "raw_results": ocr_results,
            "extracted_text": " ".join(extracted_texts),
            "detected_languages": list(detected_langs),
        }
    
    def predict(self, image: Image.Image, image_path: Optional[str] = None) -> PipelineResult:
        """
        Run the full Stage 1 pipeline on an image.
        
        CLIP and YOLO run in parallel, then DINO and OCR process YOLO results.
        
        Args:
            image: PIL Image to process.
            image_path: Optional path for metadata.
            
        Returns:
            PipelineResult with all predictions.
        """
        if not self._models_loaded:
            self.load_models()
        
        result = PipelineResult(image_path=image_path)
        
        # Run CLIP and YOLO in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            clip_future = executor.submit(self._run_clip, image)
            yolo_future = executor.submit(self._run_yolo, image)
            
            # Get results
            vibe_scores = clip_future.result()
            detections = yolo_future.result()
        
        # Process CLIP results
        result.vibe_scores = vibe_scores
        if vibe_scores:
            top_vibe = max(vibe_scores, key=vibe_scores.get)
            result.top_vibe = top_vibe
            result.top_vibe_confidence = vibe_scores[top_vibe]
        
        # Process YOLO results
        result.detections = detections
        result.num_detections = sum(len(v) for v in detections.values())
        
        # Run DINO and OCR in parallel (both depend on YOLO)
        if self.config.enable_ocr and self._ocr:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                dino_future = executor.submit(self._run_dino, image, detections)
                ocr_future = executor.submit(self._run_ocr, image, detections)
                
                country_scores = dino_future.result()
                ocr_data = ocr_future.result()
        else:
            country_scores = self._run_dino(image, detections)
            ocr_data = {}
        
        # Process DINO results
        result.country_scores = country_scores
        if country_scores:
            top_country = max(country_scores, key=country_scores.get)
            result.top_country = top_country
            result.top_country_confidence = country_scores[top_country]
        
        # Process OCR results
        if ocr_data:
            result.ocr_results = ocr_data.get("raw_results", {})
            result.extracted_text = ocr_data.get("extracted_text", "")
            result.detected_languages = ocr_data.get("detected_languages", [])
        
        return result
    
    def predict_from_path(self, image_path: str) -> PipelineResult:
        """
        Convenience method to predict from an image file path.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            PipelineResult with all predictions.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(path).convert("RGB")
        return self.predict(image, image_path=str(path))
    
    def predict_batch(
        self, 
        images: List[Image.Image], 
        image_paths: Optional[List[str]] = None
    ) -> List[PipelineResult]:
        """
        Process multiple images.
        
        Args:
            images: List of PIL Images.
            image_paths: Optional list of paths for metadata.
            
        Returns:
            List of PipelineResult objects.
        """
        if image_paths is None:
            image_paths = [None] * len(images)
        
        return [
            self.predict(img, path) 
            for img, path in zip(images, image_paths)
        ]
    
    def get_top_predictions(
        self, 
        result: PipelineResult, 
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get top-k predictions from a result.
        
        Args:
            result: PipelineResult to extract from.
            k: Number of top results (defaults to config.top_k_results).
            
        Returns:
            Dictionary with top vibes and countries.
        """
        k = k or self.config.top_k_results
        
        top_vibes = sorted(
            result.vibe_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:k]
        
        top_countries = sorted(
            result.country_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:k]
        
        return {
            "top_vibes": top_vibes,
            "top_countries": top_countries,
            "detections": result.detections,
        }
