from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


def _to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if t.device != device:
        return t.to(device, non_blocking=True)
    return t


def compute_input_gradcam(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    target_type: str = "country",  # "country" or "content"
    target_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Very simple GradCAM-like saliency:

    - Treat input image as requires_grad
    - Pick a target logit (country or content)
    - Backprop d(logit)/d(image)
    - Aggregate gradient magnitude over channels to get a heatmap

    Args:
        model: ClipLandscapeModel
        images: [B, 3, H, W] tensor (already normalised)
        device: torch.device
        target_type: "country" or "content"
        target_indices:
            If provided: [B] tensor of class indices to explain.
            If None: use argmax prediction per sample.

    Returns:
        heatmaps: [B, H, W] tensor (0..1)
        preds:    [B] tensor of predicted class indices for that target_type
    """
    model.eval()
    images = _to_device(images, device)
    images = images.clone().detach().requires_grad_(True)

    # Forward pass
    out = model(images)
    if target_type == "country":
        logits = out["country_logits"]  # [B, num_countries]
    elif target_type == "content":
        logits = out["content_logits"]  # [B, num_contents]
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    # Choose target class per sample
    with torch.no_grad():
        preds = torch.argmax(logits, dim=-1)  # [B]

    if target_indices is None:
        target_indices = preds
    else:
        target_indices = _to_device(target_indices, device)

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


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    Convert a normalised image tensor [3, H, W] back to a visualisable PIL image.

    NOTE: This assumes CLIP-style normalisation; we roughly undo it.
    It is only for qualitative GradCAM overlays, not for exact colour fidelity.
    """
    # CLIP mean/std
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    img = image_tensor.detach().cpu() * std + mean
    img = img.clamp(0.0, 1.0)
    img = (img * 255).byte().permute(1, 2, 0).numpy()

    return Image.fromarray(img)


def overlay_heatmap_on_image(
    image: Image.Image,
    heatmap: torch.Tensor,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """
    Overlay a heatmap [H, W] on top of a PIL RGB image.
    """
    heatmap_np = heatmap.detach().cpu().numpy().astype(np.float32)
    heatmap_np = np.clip(heatmap_np, 0.0, 1.0)

    # Resize heatmap to image size if needed
    H_img, W_img = image.size[1], image.size[0]
    H_hm, W_hm = heatmap_np.shape
    if (H_hm, W_hm) != (H_img, W_img):
        heatmap_np = np.array(
            Image.fromarray((heatmap_np * 255).astype(np.uint8)).resize(
                (W_img, H_img), resample=Image.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0

    # Apply colormap (simple jet-like mapping)
    # shape: [H, W, 3]
    cmap = np.zeros((H_img, W_img, 3), dtype=np.float32)
    cmap[..., 0] = np.clip(2.0 * heatmap_np - 0.5, 0.0, 1.0)  # red
    cmap[..., 1] = np.clip(2.0 * heatmap_np, 0.0, 1.0)        # green
    cmap[..., 2] = np.clip(2.0 * (1.0 - heatmap_np), 0.0, 1.0)  # blue

    cmap_img = Image.fromarray((cmap * 255).astype(np.uint8)).convert("RGBA")
    base_rgba = image.convert("RGBA")

    blended = Image.blend(base_rgba, cmap_img, alpha=alpha)
    return blended.convert("RGB")


def save_gradcam_examples(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    out_dir: Path,
    target_type: str = "country",
    max_images: int = 8,
) -> List[Path]:
    """
    Convenience helper:
      - takes a batch dict from DataLoader
      - computes GradCAM heatmaps
      - overlays them and saves under out_dir

    Returns list of saved image paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = batch["image"]  # [B, 3, H, W]
    filenames = batch.get("filename", None)

    B = images.size(0)
    if max_images is not None:
        B = min(B, max_images)
        images = images[:B]
        if filenames is not None:
            filenames = filenames[:B]

    heatmaps, preds = compute_input_gradcam(
        model=model,
        images=images,
        device=device,
        target_type=target_type,
        target_indices=None,  # use argmax
    )

    saved_paths: List[Path] = []

    for i in range(B):
        img_tensor = images[i]
        hm = heatmaps[i]

        pil_img = tensor_to_pil(img_tensor)
        overlay = overlay_heatmap_on_image(pil_img, hm, alpha=0.5)

        if filenames is not None:
            base_name = Path(str(filenames[i])).stem
        else:
            base_name = f"sample_{i}"

        fname = f"{base_name}_gradcam_{target_type}.jpg"
        out_path = out_dir / fname
        overlay.save(out_path)
        saved_paths.append(out_path)

    return saved_paths
