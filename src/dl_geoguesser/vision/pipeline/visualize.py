"""
Visualization utilities for the GeoLocation Pipeline.

Provides functions to visualize:
- CLIP attention heatmaps
- YOLO bounding boxes
- Combined visualizations
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dl_geoguesser.vision.clip_vibe.model import ClipVibe


def draw_bounding_boxes(
    image: Image.Image,
    detections: Dict[str, List[Dict[str, Any]]],
    show_labels: bool = True,
    show_confidence: bool = True,
    line_width: int = 3,
) -> Image.Image:
    """
    Draw YOLO bounding boxes on an image.
    
    Args:
        image: PIL Image to draw on.
        detections: YOLO detections dict with class names and instances.
        show_labels: Whether to show class labels.
        show_confidence: Whether to show confidence scores.
        line_width: Width of bounding box lines.
        
    Returns:
        PIL Image with bounding boxes drawn.
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Create a copy to draw on
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Generate colors for each class
    class_names = sorted(detections.keys())
    colors = _generate_colors(len(class_names))
    class_colors = {name: colors[i] for i, name in enumerate(class_names)}
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()
    
    # Draw each detection
    for class_name, instances in detections.items():
        color = class_colors[class_name]
        
        for instance in instances:
            x1, y1, x2, y2 = instance['bbox_crop']
            confidence = instance['confidence']
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            
            # Draw label
            if show_labels:
                label = class_name
                if show_confidence:
                    label += f" {confidence:.2f}"
                
                # Get text size
                bbox = draw.textbbox((x1, y1), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Draw label background
                draw.rectangle(
                    [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                    fill=color
                )
                
                # Draw label text
                draw.text((x1 + 2, y1 - text_height - 2), label, fill=(255, 255, 255), font=font)
    
    return img_draw


def generate_clip_heatmap(
    clip_model: ClipVibe,
    image: Image.Image,
    target_class: str,
    alpha: float = 0.5,
) -> Image.Image:
    """
    Generate CLIP attention heatmap for a specific class.
    
    Args:
        clip_model: Loaded ClipVibe model.
        image: PIL Image to analyze.
        target_class: Class name to generate heatmap for.
        alpha: Overlay transparency (0.0 = original, 1.0 = full heatmap).
        
    Returns:
        PIL Image with attention heatmap overlay.
    """
    return clip_model.explain(
        image,
        target_class,
        method="attention",
        alpha=alpha,
    )


def create_combined_visualization(
    image: Image.Image,
    detections: Dict[str, List[Dict[str, Any]]],
    clip_model: Optional[ClipVibe] = None,
    top_class: Optional[str] = None,
    heatmap_alpha: float = 0.4,
) -> Tuple[Image.Image, Optional[Image.Image]]:
    """
    Create combined visualization with bounding boxes and optional heatmap.
    
    Args:
        image: Original PIL Image.
        detections: YOLO detections.
        clip_model: Optional ClipVibe model for heatmap generation.
        top_class: Top predicted class for heatmap.
        heatmap_alpha: Heatmap overlay transparency.
        
    Returns:
        Tuple of (bbox_image, heatmap_image). heatmap_image is None if no model provided.
    """
    # Draw bounding boxes
    bbox_image = draw_bounding_boxes(image, detections)
    
    # Generate heatmap if model provided
    heatmap_image = None
    if clip_model and top_class:
        try:
            heatmap_image = generate_clip_heatmap(
                clip_model,
                image,
                top_class,
                alpha=heatmap_alpha,
            )
        except Exception as e:
            print(f"Warning: Failed to generate heatmap: {e}")
    
    return bbox_image, heatmap_image


def create_result_grid(
    original: Image.Image,
    bbox_image: Image.Image,
    heatmap_image: Optional[Image.Image] = None,
    result_text: Optional[str] = None,
) -> Image.Image:
    """
    Create a grid visualization showing original, bboxes, and heatmap.
    
    Args:
        original: Original image.
        bbox_image: Image with bounding boxes.
        heatmap_image: Optional image with heatmap.
        result_text: Optional text to overlay on images.
        
    Returns:
        Combined grid image.
    """
    images = [original, bbox_image]
    titles = ["Original", "Detections"]
    
    if heatmap_image:
        images.append(heatmap_image)
        titles.append("Attention Heatmap")
    
    # Resize all to same height
    target_height = 400
    resized = []
    for img in images:
        aspect = img.width / img.height
        new_width = int(target_height * aspect)
        resized.append(img.resize((new_width, target_height), Image.Resampling.LANCZOS))
    
    # Calculate grid dimensions
    total_width = sum(img.width for img in resized) + 20 * (len(resized) - 1)
    total_height = target_height + 60  # Extra space for titles
    
    # Create grid
    grid = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    
    # Try to load font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(grid)
    
    # Paste images with titles
    x_offset = 0
    for img, title in zip(resized, titles):
        # Draw title
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x_offset + (img.width - text_width) // 2
        draw.text((text_x, 10), title, fill=(0, 0, 0), font=font)
        
        # Paste image
        grid.paste(img, (x_offset, 40))
        x_offset += img.width + 20
    
    return grid


def save_visualizations(
    image: Image.Image,
    detections: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    image_name: str,
    clip_model: Optional[ClipVibe] = None,
    top_class: Optional[str] = None,
    create_grid: bool = True,
) -> Dict[str, Path]:
    """
    Save all visualizations to disk.
    
    Args:
        image: Original PIL Image.
        detections: YOLO detections.
        output_dir: Directory to save visualizations.
        image_name: Base name for output files.
        clip_model: Optional ClipVibe model for heatmap.
        top_class: Top predicted class for heatmap.
        create_grid: Whether to create a combined grid visualization.
        
    Returns:
        Dict mapping visualization type to saved file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    base_name = Path(image_name).stem
    
    # Draw bounding boxes
    bbox_image = draw_bounding_boxes(image, detections)
    bbox_path = output_dir / f"{base_name}_detections.jpg"
    bbox_image.save(bbox_path, quality=95)
    saved_files['detections'] = bbox_path
    
    # Generate heatmap if model provided
    heatmap_image = None
    if clip_model and top_class:
        try:
            heatmap_image = generate_clip_heatmap(clip_model, image, top_class)
            heatmap_path = output_dir / f"{base_name}_heatmap.jpg"
            heatmap_image.save(heatmap_path, quality=95)
            saved_files['heatmap'] = heatmap_path
        except Exception as e:
            print(f"Warning: Failed to save heatmap: {e}")
    
    # Create grid visualization
    if create_grid:
        grid_image = create_result_grid(image, bbox_image, heatmap_image)
        grid_path = output_dir / f"{base_name}_grid.jpg"
        grid_image.save(grid_path, quality=95)
        saved_files['grid'] = grid_path
    
    return saved_files


def _generate_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        # Convert HSV to RGB
        h = hue * 6
        c = 1.0
        x = 1.0 - abs((h % 2) - 1.0)
        
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    
    return colors
