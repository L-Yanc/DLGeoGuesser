from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel

from .data import get_dino_transforms


@dataclass
class ModelConfig:
    backbone: str
    pretrained_name: str
    freeze_backbone: bool
    hidden_dim: int
    dropout: float
    device: str


def build_model_config(cfg: Dict) -> ModelConfig:
    """
    Extract model-related fields from the full YAML config.
    """
    m = cfg["model"]
    train = cfg.get("training", {})

    return ModelConfig(
        backbone=m.get("backbone", "dino"),
        pretrained_name=m.get("pretrained_name", "facebook/dinov2-base"),
        freeze_backbone=bool(m.get("freeze_backbone", True)),
        hidden_dim=int(m.get("hidden_dim", 512)),
        dropout=float(m.get("dropout", 0.1)),
        device=train.get("device", "mps"),
    )


class DinoImageBackbone(nn.Module):
    """
    Wrapper around a HuggingFace DINOv2 model to expose image embeddings.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.backbone != "dino":
            raise ValueError(f"Unsupported backbone: {cfg.backbone}")

        self.dino = AutoModel.from_pretrained(cfg.pretrained_name)
        self.embed_dim = self.dino.config.hidden_size

        if cfg.freeze_backbone:
            for p in self.dino.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, H, W] already normalised.
        Returns: [B, embed_dim] image embeddings (pooler_output).
        """
        outputs = self.dino(pixel_values=images)
        return outputs.pooler_output


class CountryHead(nn.Module):
    """
    MLP head on top of DINO embeddings for country classification.
    """

    def __init__(self, embed_dim: int, num_countries: int, cfg: ModelConfig):
        super().__init__()
        hidden_dim = cfg.hidden_dim
        dropout = cfg.dropout

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.country_classifier = nn.Linear(hidden_dim, num_countries)

    def forward(self, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        feats: [B, embed_dim]
        Returns dict with:
          - 'country_logits': [B, num_countries]
          - 'head_features': [B, hidden_dim]
        """
        h = self.mlp(feats)
        country_logits = self.country_classifier(h)

        return {
            "country_logits": country_logits,
            "head_features": h,
        }


class DinoGeoguesserModel(nn.Module):
    """
    Full model: DINOv2 image backbone + head for country classification.
    """

    def __init__(self, num_countries: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = DinoImageBackbone(cfg)
        self.head = CountryHead(
            embed_dim=self.backbone.embed_dim,
            num_countries=num_countries,
            cfg=cfg,
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        images: [B, 3, H, W]
        Returns dict with:
          - 'country_logits'
          - 'backbone_features'
          - 'head_features'
        """
        feats = self.backbone(images)
        head_out = self.head(feats)

        return {
            "country_logits": head_out["country_logits"],
            "backbone_features": feats,
            "head_features": head_out["head_features"],
        }


def build_dino_geoguesser_model(
    cfg: Dict,
    num_countries: int,
) -> Tuple[DinoGeoguesserModel, ModelConfig]:
    """
    Convenience function to build the model from config + label counts.
    """
    model_cfg = build_model_config(cfg)
    model = DinoGeoguesserModel(
        num_countries=num_countries,
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


class DinoGeoguesser:
    """
    A wrapper for the DINO-based country classifier.
    This class handles loading a trained model and running predictions on
    image crops based on YOLO outputs.
    """

    def __init__(self, weights_path: str, device: Optional[str] = None):
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"DINO weights not found at {weights_path}")

        self.ckpt = torch.load(weights_path, map_location="cpu")
        self.cfg = self.ckpt['config']
        self.country2id = self.ckpt['country2id']
        self.id2country = {v: k for k, v in self.country2id.items()}
        self.device = get_device(self.cfg, cli_device=device)

        self.model, _ = build_dino_geoguesser_model(self.cfg, num_countries=len(self.country2id))
        self.model.load_state_dict(self.ckpt['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        _, self.transform = get_dino_transforms(self.cfg["data"]["image_size"])
        print(f"DinoGeoguesser initialized on device: {self.device}")

    def predict_from_crops(self, image: Image.Image, detections: Dict) -> Dict[str, float]:
        """
        Takes an image and YOLO detections, and returns aggregated country scores.
        """
        if not detections:
            return {}

        all_probs = []
        for instances in detections.values():
            for instance in instances:
                bbox = instance['bbox_crop']
                crop = image.crop(bbox)

                crop_tensor = self.transform(crop).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    out = self.model(crop_tensor)
                    probs = torch.softmax(out['country_logits'], dim=-1)
                    all_probs.append(probs.cpu())

        if not all_probs:
            return {}

        mean_probs = torch.cat(all_probs, dim=0).mean(dim=0)

        country_scores = {self.id2country[i]: mean_probs[i].item() for i in range(len(mean_probs))}
        return country_scores