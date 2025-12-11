from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import create_dataloaders, load_config
from .model import DinoGeoguesserModel, build_dino_geoguesser_model, get_device


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def train_one_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, dev: torch.device, epoch: int, num_epochs: int, use_embeddings: bool, augment_embedding_noise: float) -> Dict[str, float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, steps = 0.0, 0.0, 0
    loop = tqdm(loader, desc=f"Train Epoch {epoch+1}/{num_epochs}", leave=False)

    for batch in loop:
        optim.zero_grad()
        targets = batch["country_id"].to(dev)

        if use_embeddings:
            features = batch["embedding"].to(dev)
        else:
            images = batch["image"].to(dev)
            # Get features from the backbone
            with torch.no_grad():
                features = model.backbone(images)

        # Apply embedding augmentation if specified
        if augment_embedding_noise > 0:
            features += torch.randn_like(features) * augment_embedding_noise

        logits = model.head(features)["country_logits"]
        loss = loss_fn(logits, targets)
        loss.backward()
        optim.step()

        acc = accuracy_from_logits(logits, targets)
        total_loss += loss.item()
        total_acc += acc
        steps += 1
        loop.set_postfix(loss=loss.item(), acc=acc)

    return {"loss": total_loss / steps, "acc": total_acc / steps}


def evaluate(model: nn.Module, loader: DataLoader, dev: torch.device, use_embeddings: bool) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, steps = 0.0, 0.0, 0
    loop = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in loop:
            targets = batch["country_id"].to(dev)
            if use_embeddings:
                features = batch["embedding"].to(dev)
            else:
                features = model.backbone(batch["image"].to(dev))

            logits = model.head(features)["country_logits"]
            loss = loss_fn(logits, targets)

            acc = accuracy_from_logits(logits, targets)
            total_loss += loss.item()
            total_acc += acc
            steps += 1
            loop.set_postfix(loss=loss.item(), acc=acc)
    return {"loss": total_loss / steps, "acc": total_acc / steps}


def precompute_embeddings(model: DinoGeoguesserModel, cfg: Dict, device: torch.device, save_path: Path):
    print(f"Pre-computing embeddings and saving to {save_path}...")
    model.eval()

    # Create dataloaders with no image augmentation for pre-computation
    train_loader, val_loader, test_loader, country2id = create_dataloaders(cfg, augmentations='none')

    splits = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    output = {'country2id': country2id}

    with torch.no_grad():
        for split_name, loader in splits.items():
            all_embeds, all_labels = [], []
            for batch in tqdm(loader, desc=f"Computing {split_name} embeddings"):
                images = batch["image"].to(device)
                labels = batch["country_id"]
                embeds = model.backbone(images)
                all_embeds.append(embeds.cpu())
                all_labels.append(labels.cpu())

            output[f'{split_name}_embeds'] = torch.cat(all_embeds, dim=0)
            output[f'{split_name}_labels'] = torch.cat(all_labels, dim=0)
            print(f"Computed {len(output[f'{split_name}_embeds'])} embeddings for {split_name} split.")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, save_path)
    print("Pre-computation complete.")


def save_checkpoint(model, optimizer, epoch, best_val_acc, cfg, country2id, out_dir, name="best.pt"):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch, "best_val_acc": best_val_acc, "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "config": cfg, "country2id": country2id
    }, out_dir / name)
    print(f"[Training] Saved checkpoint to {out_dir / name}")


def train(cfg_path: str, name: str, device_str: Optional[str], weights: Optional[str], resume: bool, precompute: bool, augmentations: str):
    # --- Validation ---
    if precompute and ('image' in augmentations or 'both' in augmentations):
        raise ValueError("Image augmentation cannot be used with pre-computed embeddings.")

    # --- Setup ---
    cfg = load_config(cfg_path)
    device = get_device(cfg, cli_device=device_str)
    save_dir = Path(cfg["eval"]["save_dir"]) / name
    use_embed_aug = 'embedding' in augmentations or 'both' in augmentations

    # --- Model and Optimizer ---
    # country2id will be loaded later, so we init with a placeholder
    model, _ = build_dino_geoguesser_model(cfg, num_countries=10)
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg["training"]["lr_head"]))

    # --- Checkpoint/Resuming Logic ---
    start_epoch, best_val_acc = 0, -1.0
    if resume:
        ckpt_path = save_dir / "last.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found at {ckpt_path}")
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch, best_val_acc = ckpt['epoch'] + 1, ckpt['best_val_acc']
    elif weights:
        print(f"Loading initial weights from {weights}")
        ckpt = torch.load(weights, map_location=device)
        model.load_state_dict(ckpt.get('state_dict', ckpt))

    # --- Data Loading ---
    embeddings_path = save_dir / "embeddings.pt"
    if precompute:
        if not embeddings_path.exists():
            print("Embeddings not found. Starting pre-computation process...")
            precompute_embeddings(model, cfg, device, embeddings_path)
        train_loader, val_loader, test_loader, country2id = create_dataloaders(cfg, augmentations=augmentations, precomputed_path=embeddings_path)
    else:
        print("Using on-the-fly image loading and processing.")
        train_loader, val_loader, test_loader, country2id = create_dataloaders(cfg, augmentations=augmentations)

    # Re-initialize model if the number of countries from data is different
    if model.head.country_classifier.out_features != len(country2id):
        raise ValueError(f"Detected {len(country2id)} countries, but model head has {model.head.country_classifier.out_features}")
        # print("Re-initializing model head with correct number of classes.")
        # model, _ = build_dino_geoguesser_model(cfg, num_countries=len(country2id))
        # model.to(device)

    # --- Training Loop ---
    print(f"Starting training on device: {device}")
    num_epochs = cfg["training"]["num_epochs"]
    noise = float(cfg["training"]["embedding_noise_level"]) if use_embed_aug else 0.0

    for epoch in range(start_epoch, num_epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, device, epoch, num_epochs, precompute, noise)
        val_stats = evaluate(model, val_loader, device, precompute)
        print(f"[Epoch {epoch+1}/{num_epochs}] train_loss={train_stats['loss']:.4f}, acc={train_stats['acc']:.3f} | val_loss={val_stats['loss']:.4f}, acc={val_stats['acc']:.3f}")

        save_checkpoint(model, optimizer, epoch, best_val_acc, cfg, country2id, save_dir, name="last.pt")
        if val_stats['acc'] > best_val_acc:
            best_val_acc = val_stats['acc']
            print(f"  -> New best val acc: {best_val_acc:.3f}")
            save_checkpoint(model, optimizer, epoch, best_val_acc, cfg, country2id, save_dir, name="best.pt")

    print("\nTraining complete. Final test set evaluation...")
    test_stats = evaluate(model, test_loader, device, precompute)
    print(f"[Test] loss={test_stats['loss']:.4f}, acc={test_stats['acc']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="DINO Geoguesser Model CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    parser_train = subparsers.add_parser("train", help="Train the model")
    parser_train.add_argument("--config", type=str, default="configs/dino_geoguesser.yaml")
    parser_train.add_argument("--name", type=str, default="dino_run")
    parser_train.add_argument("--device", type=str, default=None)
    parser_train.add_argument("--weights", type=str, default=None, help="Load initial model weights for fine-tuning.")
    parser_train.add_argument("--resume", action="store_true", help="Resume from last.pt in run directory.")
    parser_train.add_argument("--precompute-embeddings", action="store_true", help="Pre-compute and use embeddings for faster training.")
    parser_train.add_argument("--augmentations", type=str, default="none", choices=['none', 'image', 'embedding', 'both'])

    args = parser.parse_args()
    if args.command == "train":
        if args.resume and args.weights:
            raise ValueError("--resume and --weights are mutually exclusive.")
        train(args.config, args.name, args.device, args.weights, args.resume, args.precompute_embeddings, args.augmentations)


if __name__ == "__main__":
    main()
