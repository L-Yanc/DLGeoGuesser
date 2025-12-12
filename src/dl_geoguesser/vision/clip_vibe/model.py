from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel


@dataclass
class ModelConfig:
    backbone: str
    pretrained_name: str
    freeze_backbone: bool
    hidden_dims: List[int]
    dropout: float
    device: str


def build_model_config(cfg: Dict) -> ModelConfig:
    """
    Extract model-related fields from the full YAML config.
    """
    m = cfg["model"]
    train = cfg.get("training", {})

    return ModelConfig(
        backbone=m.get("backbone", "clip"),
        pretrained_name=m.get("pretrained_name", "openai/clip-vit-base-patch32"),
        freeze_backbone=bool(m.get("freeze_backbone", True)),
        hidden_dims=m.get("hidden_dims", [512]),
        dropout=float(m.get("dropout", 0.1)),
        device=train.get("device", "mps"),
    )


class ClipImageBackbone(nn.Module):
    """
    Wrapper around a HuggingFace CLIP model to expose image embeddings.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.backbone != "clip":
            raise ValueError(f"Unsupported backbone: {cfg.backbone}")

        self.clip = CLIPVisionModel.from_pretrained(cfg.pretrained_name)
        self.embed_dim = self.clip.config.hidden_size

        if cfg.freeze_backbone:
            for p in self.clip.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, H, W] already normalised.
        Returns: [B, embed_dim] image embeddings (pooler_output).
        """
        outputs = self.clip(pixel_values=images)
        return outputs.pooler_output


class ClassificationHead(nn.Module):
    """
    MLP head on top of CLIP embeddings for classification.
    The architecture is dynamically built based on the `hidden_dims` list.
    """

    def __init__(self, embed_dim: int, num_classes: int, cfg: ModelConfig):
        super().__init__()

        layers = []
        in_dim = embed_dim
        for h_dim in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(cfg.dropout))
            in_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        feats: [B, embed_dim]
        Returns dict with:
          - 'logits': [B, num_classes]
          - 'head_features': [B, last_hidden_dim]
        """
        h = self.mlp(feats)
        logits = self.classifier(h)

        return {
            "logits": logits,
            "head_features": h,
        }


class ClipVibeModel(nn.Module):
    """
    Full model: CLIP image backbone + head for classification.
    """

    def __init__(self, num_classes: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = ClipImageBackbone(cfg)
        self.head = ClassificationHead(
            embed_dim=self.backbone.embed_dim,
            num_classes=num_classes,
            cfg=cfg,
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        images: [B, 3, H, W]
        Returns dict with:
          - 'logits'
          - 'backbone_features'
          - 'head_features'
        """
        feats = self.backbone(images)
        head_out = self.head(feats)

        return {
            "logits": head_out["logits"],
            "backbone_features": feats,
            "head_features": head_out["head_features"],
        }


def build_clip_vibe_model(
    cfg: Dict,
    num_classes: int,
) -> Tuple[ClipVibeModel, ModelConfig]:
    """
    Convenience function to build the model from config + label counts.
    """
    model_cfg = build_model_config(cfg)
    model = ClipVibeModel(
        num_classes=num_classes,
        cfg=model_cfg,
    )

    # The device will be handled by the main training script or wrapper class
    return model, model_cfg


def get_device(cfg: Dict, cli_device: Optional[str] = None) -> torch.device:
    """
    Determines the correct torch device to use, prioritizing the command-line
    argument, then the config file, and falling back to CPU.
    """
    if cli_device:
        desired = cli_device
    else:
        desired = cfg.get("training", {}).get("device", "cpu")

    if desired in ["cuda", "mps"] and getattr(torch, desired).is_available():
        return torch.device(desired)
    return torch.device("cpu")


import numpy as np

from . import gradcam


class ClipVibe:
    """
    A wrapper for the CLIP-based classifier.
    This class handles loading a trained model and running predictions on images.
    """

    def __init__(self, weights_path: str, device: Optional[str] = None):
        self.device = get_device(self._load_checkpoint(weights_path), cli_device=device)
        self._initialize_model()
        self._initialize_processor()
        print(f"ClipVibe initialized on device: {self.device}")

    def _load_checkpoint(self, weights_path: str) -> Dict:
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"ClipVibe weights not found at {weights_path}")
        self.ckpt = torch.load(weights_path, map_location="cpu")
        self.cfg = self.ckpt['config']
        self.class2id = self.ckpt['class2id']
        self.id2class = {v: k for k, v in self.class2id.items()}
        return self.cfg

    def _initialize_model(self):
        self.model, self.model_cfg = build_clip_vibe_model(self.cfg, num_classes=len(self.class2id))
        self.model.load_state_dict(self.ckpt['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def _initialize_processor(self):
        self.processor = CLIPImageProcessor.from_pretrained(self.model_cfg.pretrained_name)

    def predict(self, image: Image.Image) -> Dict[str, float]:
        """
        Takes an image and returns a dictionary of class scores.
        """
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            out = self.model(inputs['pixel_values'])
            probs = torch.softmax(out['logits'], dim=-1)
            mean_probs = probs.mean(dim=0)

        scores = {self.id2class[i]: mean_probs[i].item() for i in range(len(mean_probs))}
        return scores
    
    def _tensor_to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        """
        Convert a normalised image tensor [3, H, W] back to a visualisable PIL image.
        """
        mean = torch.tensor(self.processor.image_mean).view(3, 1, 1)
        std = torch.tensor(self.processor.image_std).view(3, 1, 1)

        img = image_tensor.detach().cpu() * std + mean
        img = img.clamp(0.0, 1.0)
        img = (img * 255).byte().permute(1, 2, 0).numpy()

        return Image.fromarray(img)

    def _overlay_heatmap_on_image(
        self,
        image: Image.Image,
        heatmap: torch.Tensor,
        alpha: float = 0.5,
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

    def generate_gradcam(self, image: Image.Image, target_class: str) -> Image.Image:
        """
        Generates a Grad-CAM overlay for a given image and target class.
        """
        if target_class not in self.class2id:
            raise ValueError(f"Unknown class: {target_class}")
        target_class_id = self.class2id[target_class]
        
        inputs = self.processor(images=image, return_tensors="pt")
        
        heatmaps, _ = gradcam.compute_gradcam(
            model=self.model,
            images=inputs["pixel_values"],
            device=self.device,
            target_class=target_class_id,
        )
        
        heatmap = heatmaps[0]
        overlay = self._overlay_heatmap_on_image(image, heatmap)
        return overlay
