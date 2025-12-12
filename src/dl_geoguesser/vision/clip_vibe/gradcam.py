from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


def compute_gradcam(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    target_class: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Very simple GradCAM-like saliency:

    - Treat input image as requires_grad
    - Pick a target logit
    - Backprop d(logit)/d(image)
    - Aggregate gradient magnitude over channels to get a heatmap

    Args:
        model: ClipVibeModel
        images: [B, 3, H, W] tensor (already normalised)
        device: torch.device
        target_class:
            If provided: The class index to explain.
            If None: use argmax prediction.

    Returns:
        heatmaps: [B, H, W] tensor (0..1)
        preds:    [B] tensor of predicted class indices
    """
    model.eval()
    images = images.to(device)
    images = images.clone().detach().requires_grad_(True)

    # Forward pass
    out = model(images)
    logits = out["logits"]  # [B, num_classes]

    # Choose target class per sample
    with torch.no_grad():
        preds = torch.argmax(logits, dim=-1)  # [B]

    if target_class is None:
        target_indices = preds
    else:
        target_indices = torch.tensor([target_class] * images.size(0), device=device)

    # Build scalar objective: sum of selected logits
    batch_idx = torch.arange(logits.size(0), device=device)
    selected_logits = logits[batch_idx, target_indices]  # [B]
    objective = selected_logits.sum()

    # Backward
    model.zero_grad(set_to_none=True)
    if images.grad is not None:
        images.grad.zero_()

    objective.backward()

    # images.grad: [B, 3, H, W]
    grads = images.grad  # [B, 3, H, W]
    # Aggregate channels → [B, H, W]
    heatmaps = grads.abs().mean(dim=1)

    # Normalise per-image to [0, 1]
    B = heatmaps.size(0)
    flat = heatmaps.view(B, -1)
    min_vals, _ = flat.min(dim=1, keepdim=True)
    max_vals, _ = flat.max(dim=1, keepdim=True)
    denom = (max_vals - min_vals).clamp(min=1e-8)
    flat_norm = (flat - min_vals) / denom
    heatmaps_norm = flat_norm.view_as(heatmaps)

    return heatmaps_norm.detach().cpu(), preds.detach().cpu()