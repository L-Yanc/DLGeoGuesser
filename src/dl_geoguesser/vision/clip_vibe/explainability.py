from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def compute_attention_saliency(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    target_class: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes gradient-weighted attention saliency for a Vision Transformer model.

    **What this shows:**
    - Regions that CLIP's pre-trained attention mechanism focused on
    - Weighted by gradient signal from the classification head

    **Important limitations:**
    - Does NOT show which pixels caused the prediction (if backbone is frozen)
    - Attention weights were never optimized by your classifier
    - Best used for qualitative debugging, not causal explanation

    **For rigorous explanations:** Use compute_integrated_gradients() instead,
    or fine-tune the backbone (set freeze_backbone: false).

    Args:
        model: The ViT-based model (e.g., ClipVibeModel).
        images: [B, 3, H, W] tensor of normalised images.
        device: The torch device to use.
        target_class: If provided, the class index to explain. If None,
                      the predicted class with the highest score is used.

    Returns:
        A tuple containing:
        - heatmaps: [B, H_patch, W_patch] tensor of saliency maps.
        - preds: [B] tensor of predicted class indices.
    """
    model.eval()
    images = images.to(device)

    # Forward pass to get logits and all attention maps
    out = model(images)
    logits = out["logits"]
    all_attentions: Tuple[torch.Tensor, ...] = out["attentions"]

    # Use last layer attention
    last_layer_attention = all_attentions[-1]
    last_layer_attention.retain_grad()

    # Choose target class per sample
    with torch.no_grad():
        preds = torch.argmax(logits, dim=-1)

    if target_class is None:
        target_indices = preds
    else:
        target_indices = torch.tensor([target_class] * images.size(0), device=device)

    # Build scalar objective: sum of selected logits
    batch_idx = torch.arange(logits.size(0), device=device)
    selected_logits = logits[batch_idx, target_indices]
    objective = selected_logits.sum()

    # Backward pass
    model.zero_grad(set_to_none=True)
    objective.backward()

    attention_grads = last_layer_attention.grad

    if attention_grads is None:
        raise RuntimeError(
            "Could not get gradients for attention maps. "
            "Ensure that the attention computation chain has requires_grad=True "
            "and model.backbone is not frozen (or temporarily unfrozen)."
        )

    # Average gradients and attention across heads
    averaged_attention_grads_cls_to_patches = attention_grads.mean(dim=1)[:, 0, 1:]
    averaged_attention_cls_to_patches = last_layer_attention.mean(dim=1)[:, 0, 1:]

    # Gradient-weighted attention
    cam = averaged_attention_grads_cls_to_patches * averaged_attention_cls_to_patches
    cam = F.relu(cam)

    # Reshape to 2D grid
    num_patches = cam.shape[1]
    num_patches_side = int(math.sqrt(num_patches))

    if num_patches_side * num_patches_side != num_patches:
        raise ValueError(f"Number of patches ({num_patches}) is not a perfect square.")

    cam = cam.view(-1, num_patches_side, num_patches_side)

    # Normalize per-image to [0, 1]
    B = cam.size(0)
    flat = cam.view(B, -1)

    min_vals = flat.min(dim=1, keepdim=True)[0]
    max_vals = flat.max(dim=1, keepdim=True)[0]

    denom = (max_vals - min_vals).clamp(min=1e-8)
    heatmaps_norm = (flat - min_vals) / denom
    heatmaps_norm = heatmaps_norm.view_as(cam)

    return heatmaps_norm.detach().cpu(), preds.detach().cpu()


def compute_integrated_gradients(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    target_class: Optional[int] = None,
    steps: int = 50,
    baseline: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Integrated Gradients for pixel-level attribution.

    This method provides rigorous, causal explanations by measuring how
    each pixel contributes to the model's prediction. Works correctly
    even with frozen backbones.

    **What this shows:**
    - The actual importance of each pixel for the classification decision
    - Satisfies axioms: Sensitivity and Implementation Invariance

    **How it works:**
    1. Create a path from a baseline (black image) to your input image
    2. Compute gradients at multiple points along this path
    3. Integrate these gradients to get pixel attributions

    Reference: "Axiomatic Attribution for Deep Networks" (Sundararajan et al., 2017)
    https://arxiv.org/abs/1703.01365

    Args:
        model: The classification model.
        images: [B, 3, H, W] tensor of normalised images.
        device: The torch device to use.
        target_class: If provided, the class index to explain. If None,
                      uses the predicted class.
        steps: Number of interpolation steps (more = more accurate but slower).
        baseline: Optional baseline image. If None, uses zeros (black image).

    Returns:
        A tuple containing:
        - attributions: [B, H, W] tensor of pixel attributions.
        - preds: [B] tensor of predicted class indices.
    """
    model.eval()
    images = images.to(device)

    if baseline is None:
        baseline = torch.zeros_like(images)
    else:
        baseline = baseline.to(device)

    # Get predictions
    with torch.no_grad():
        out = model(images)
        logits = out["logits"]
        preds = torch.argmax(logits, dim=-1)

    if target_class is None:
        target_indices = preds
    else:
        target_indices = torch.tensor([target_class] * images.size(0), device=device)

    # Create interpolation path
    alphas = torch.linspace(0, 1, steps, device=device)

    # Store gradients for each step
    all_gradients = []

    for alpha in alphas:
        # Interpolate between baseline and image
        interpolated = baseline + alpha * (images - baseline)
        interpolated.requires_grad_(True)

        # Forward pass
        out = model(interpolated)
        logits = out["logits"]

        # Get score for target class
        batch_idx = torch.arange(logits.size(0), device=device)
        scores = logits[batch_idx, target_indices]

        # Compute gradient
        model.zero_grad()
        scores.sum().backward()

        gradients = interpolated.grad.detach()
        all_gradients.append(gradients)

    # Average gradients across all steps
    avg_gradients = torch.stack(all_gradients).mean(dim=0)

    # Integrated gradients = (image - baseline) * average_gradients
    integrated_grads = (images - baseline) * avg_gradients

    # Sum across color channels to get per-pixel attribution
    attributions = integrated_grads.sum(dim=1).abs()  # [B, H, W]

    # Normalize per-image to [0, 1]
    B, H, W = attributions.shape
    flat = attributions.view(B, -1)

    min_vals = flat.min(dim=1, keepdim=True)[0]
    max_vals = flat.max(dim=1, keepdim=True)[0]

    denom = (max_vals - min_vals).clamp(min=1e-8)
    attributions_norm = (flat - min_vals) / denom
    attributions_norm = attributions_norm.view(B, H, W)

    return attributions_norm.detach().cpu(), preds.detach().cpu()

