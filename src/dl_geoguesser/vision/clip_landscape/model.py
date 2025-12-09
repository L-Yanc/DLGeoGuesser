from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import CLIPModel


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
        backbone=m.get("backbone", "clip"),
        pretrained_name=m.get("pretrained_name", "openai/clip-vit-base-patch32"),
        freeze_backbone=bool(m.get("freeze_backbone", True)),
        hidden_dim=int(m.get("hidden_dim", 512)),
        dropout=float(m.get("dropout", 0.1)),
        device=train.get("device", "cuda"),
    )


class ClipImageBackbone(nn.Module):
    """
    Thin wrapper around HuggingFace CLIPModel to expose image embeddings.

    We assume inputs are already preprocessed with CLIP-style transforms
    (normalised, resized, etc.).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.backbone != "clip":
            raise ValueError(f"Unsupported backbone: {cfg.backbone}")

        self.clip = CLIPModel.from_pretrained(cfg.pretrained_name)
        # projection_dim is the size of the final image/text embeddings
        self.embed_dim = self.clip.config.projection_dim

        if cfg.freeze_backbone:
            for p in self.clip.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, H, W] already normalised with CLIP mean/std.
        Returns: [B, embed_dim] image embeddings.
        """
        # get_image_features handles the vision encoder + projection
        feats = self.clip.get_image_features(pixel_values=images)
        # CLIP usually returns L2-normalised embeddings; we keep as-is.
        return feats


class CountryContentHead(nn.Module):
    """
    Small MLP head on top of CLIP embeddings with two classifiers:
    - country logits
    - content / vibe logits
    """

    def __init__(self, embed_dim: int, num_countries: int, num_contents: int, cfg: ModelConfig):
        super().__init__()
        hidden_dim = cfg.hidden_dim
        dropout = cfg.dropout

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.country_classifier = nn.Linear(hidden_dim, num_countries)
        self.content_classifier = nn.Linear(hidden_dim, num_contents)

    def forward(self, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        feats: [B, embed_dim]
        Returns dict with:
          - 'country_logits': [B, num_countries]
          - 'content_logits': [B, num_contents]
          - 'head_features': [B, hidden_dim]
        """
        h = self.mlp(feats)
        country_logits = self.country_classifier(h)
        content_logits = self.content_classifier(h)

        return {
            "country_logits": country_logits,
            "content_logits": content_logits,
            "head_features": h,
        }


class ClipLandscapeModel(nn.Module):
    """
    Full model: CLIP image backbone + dual head for country + content.
    """

    def __init__(self, num_countries: int, num_contents: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = ClipImageBackbone(cfg)

        embed_dim = self.backbone.embed_dim
        self.head = CountryContentHead(
            embed_dim=embed_dim,
            num_countries=num_countries,
            num_contents=num_contents,
            cfg=cfg,
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        images: [B, 3, H, W]
        Returns dict with:
          - 'country_logits'
          - 'content_logits'
          - 'backbone_features'
          - 'head_features'
        """
        feats = self.backbone(images)
        head_out = self.head(feats)

        return {
            "country_logits": head_out["country_logits"],
            "content_logits": head_out["content_logits"],
            "backbone_features": feats,
            "head_features": head_out["head_features"],
        }


def build_clip_landscape_model(
    cfg: Dict,
    num_countries: int,
    num_contents: int,
) -> Tuple[ClipLandscapeModel, ModelConfig]:
    """
    Convenience function to build the model from config + label counts.
    """
    model_cfg = build_model_config(cfg)
    model = ClipLandscapeModel(
        num_countries=num_countries,
        num_contents=num_contents,
        cfg=model_cfg,
    )

    device = torch.device(model_cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, model_cfg
