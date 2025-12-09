from pathlib import Path

import torch
from tqdm import tqdm

from src.dl_geoguesser.vision.clip_landscape.data import load_config, create_dataloaders
from src.dl_geoguesser.vision.clip_landscape.model import build_clip_landscape_model
from src.dl_geoguesser.vision.clip_landscape.training import get_device


def compute_split_embeddings(model, data_loader, device):
    """
    Run the CLIP backbone over one split and collect:
      - embeddings  [N, D]
      - country_ids [N]
      - content_ids [N]
      - filenames   list of strings, length N
    """
    model.eval()

    all_embeds = []
    all_country_ids = []
    all_content_ids = []
    all_filenames = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Encoding", leave=True):
            images = batch["image"].to(device, non_blocking=True)

            out = model(images)
            feats = out["backbone_features"]  # [B, D]

            all_embeds.append(feats.cpu())
            all_country_ids.append(batch["country_id"].clone())
            all_content_ids.append(batch["content_id"].clone())
            all_filenames.extend(batch["filename"])

    embeddings = torch.cat(all_embeds, dim=0)
    country_ids = torch.cat(all_country_ids, dim=0)
    content_ids = torch.cat(all_content_ids, dim=0)

    return embeddings, country_ids, content_ids, all_filenames


def main(config_path: str = "configs/clip_landscape.yaml"):
    # 1) Load config and pick device
    cfg = load_config(config_path)
    device = get_device(cfg)

    # For Windows dev to avoid multiprocessing issues; on GPU boxes you can bump this back up
    cfg["data"]["num_workers"] = 0

    # 2) Build dataloaders (no shuffling, we want a stable ordering)
    train_loader, val_loader, test_loader, country2id, content2id = create_dataloaders(
        cfg, shuffle_train=False
    )

    num_countries = len(country2id)
    num_contents = len(content2id)

    # 3) Build model (CLIP backbone + head; we only use backbone_features)
    model, _ = build_clip_landscape_model(
        cfg,
        num_countries=num_countries,
        num_contents=num_contents,
    )
    model.to(device)

    # 4) Output dir for kNN baseline embeddings
    base_dir = Path(cfg["eval"].get("save_dir", "models/clip_landscape"))
    out_dir = base_dir / "knn_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]

    for split_name, loader in splits:
        print(f"\n[run_clip_embeddings] Processing split: {split_name}")
        embeddings, country_ids, content_ids, filenames = compute_split_embeddings(
            model, loader, device
        )

        payload = {
            "embeddings": embeddings,     # [N, D] float32
            "country_ids": country_ids,   # [N] long
            "content_ids": content_ids,   # [N] long
            "filenames": filenames,       # list[str]
            "country2id": country2id,
            "content2id": content2id,
        }

        out_path = out_dir / f"{split_name}_embeddings.pt"
        torch.save(payload, out_path)

        print(
            f"[run_clip_embeddings] Saved {split_name} embeddings to {out_path} "
            f"with shape {tuple(embeddings.shape)}"
        )


if __name__ == "__main__":
    main()
