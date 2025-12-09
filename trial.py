import torch

from src.dl_geoguesser.vision.clip_landscape.data import load_config, create_dataloaders
from src.dl_geoguesser.vision.clip_landscape.model import build_clip_landscape_model

cfg = load_config("configs/clip_landscape.yaml")
train_loader, val_loader, test_loader, country2id, content2id = create_dataloaders(cfg)

num_countries = len(country2id)
num_contents = len(content2id)

model, model_cfg = build_clip_landscape_model(cfg, num_countries, num_contents)

batch = next(iter(train_loader))
images = batch["image"]

with torch.no_grad():
    out = model(images)
    print(out["country_logits"].shape)   # [B, num_countries]
    print(out["content_logits"].shape)   # [B, num_contents]
