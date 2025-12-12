from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
import torch.nn as nn # Not strictly needed with this approach, but good to keep if further hook usage is considered


def compute_gradcam(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    target_class: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Attention-based Grad-CAM for a Vision Transformer (ViT) model.
    This method leverages attention maps and their gradients to create heatmaps,
    addressing the limitation of direct gradient-based methods when classification
    relies solely on the CLS token.

    Args:
        model: The ViT-based model (e.g., ClipVibeModel).
        images: [B, 3, H, W] tensor of normalised images.
        device: The torch device to use.
        target_class: If provided, the class index to explain. If None,
                      the predicted class with the highest score is used.

    Returns:
        A tuple containing:
        - heatmaps: [B, H_patch, W_patch] tensor of class activation maps.
        - preds: [B] tensor of predicted class indices.
    """
    model.eval()
    images = images.to(device)

    # Forward pass to get logits and all attention maps
    out = model(images)
    logits = out["logits"]
    all_attentions: Tuple[torch.Tensor, ...] = out["attentions"] # attentions are a tuple of tensors for each layer

    # We are interested in the attention maps of the last layer
    # These attentions have shape [B, num_heads, N, N] where N = num_patches + 1 (for CLS token)
    last_layer_attention = all_attentions[-1]

    # Ensure gradients can be computed for the attention maps
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

    # Backward pass to get gradients of the objective with respect to the attention maps
    model.zero_grad(set_to_none=True)
    objective.backward()

    attention_grads = last_layer_attention.grad # Shape: [B, num_heads, N, N]

    if attention_grads is None:
         raise RuntimeError(
            "Could not get gradients for attention maps. "
            "Ensure that the attention computation chain has requires_grad=True "
            "and model.backbone is not frozen."
        )

    # Average gradients across attention heads
    # We are interested in the gradients of the CLS token's attention (row 0) to other patches (columns 1:)
    # Shape after mean(dim=1) for attention_grads: [B, N, N]
    averaged_attention_grads_cls_to_patches = attention_grads.mean(dim=1)[:, 0, 1:] # Shape: [B, N-1]

    # Get the actual attention weights from the CLS token to other patches from the last layer.
    # Shape after mean(dim=1) for last_layer_attention: [B, N, N]
    averaged_attention_cls_to_patches = last_layer_attention.mean(dim=1)[:, 0, 1:] # Shape: [B, N-1]

    # Compute the CAM: Element-wise multiply gradients by attentions and apply ReLU.
    # This is a common practice in attention-based Grad-CAM for ViTs.
    cam = averaged_attention_grads_cls_to_patches * averaged_attention_cls_to_patches
    cam = F.relu(cam) # Shape: [B, N-1]

    # Reshape the CAM to a 2D grid.
    num_patches = cam.shape[1]
    num_patches_side = int(math.sqrt(num_patches))
    
    if num_patches_side * num_patches_side != num_patches:
        raise ValueError(f"Number of patches ({num_patches}) is not a perfect square for reshaping.")

    cam = cam.view(-1, num_patches_side, num_patches_side)  # [B, H_patch, W_patch]

    # Normalise per-image to [0, 1] for visualisation
    B = cam.size(0)
    flat = cam.view(B, -1)
    
    min_vals = flat.min(dim=1, keepdim=True)[0]
    max_vals = flat.max(dim=1, keepdim=True)[0]
    
    denom = (max_vals - min_vals).clamp(min=1e-8)
    heatmaps_norm = (flat - min_vals) / denom
    heatmaps_norm = heatmaps_norm.view_as(cam)

    return heatmaps_norm.detach().cpu(), preds.detach().cpu()