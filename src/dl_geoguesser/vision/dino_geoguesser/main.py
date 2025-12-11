from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import shutil
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import create_dataloaders, load_config
from .model import DinoGeoguesserModel, build_dino_geoguesser_model, get_device

@dataclass
class TrainingState:
    epoch: int
    best_val_acc: float

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
            with torch.no_grad():
                features = model.backbone(images)

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
    train_loader, val_loader, test_loader, country2id = create_dataloaders(cfg, augmentations='none')
    splits = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    output = {'country2id': country2id}

    with torch.no_grad():
        for split_name, loader in splits.items():
            all_embeds, all_labels = [], []
            for batch in tqdm(loader, desc=f"Computing {split_name} embeddings"):
                embeds = model.backbone(batch["image"].to(device))
                all_embeds.append(embeds.cpu())
                all_labels.append(batch["country_id"].cpu())
            output[f'{split_name}_embeds'] = torch.cat(all_embeds, dim=0)
            output[f'{split_name}_labels'] = torch.cat(all_labels, dim=0)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, save_path)
    print("Pre-computation complete.")

def save_checkpoint(model, optimizer, epoch, best_val_acc, cfg, country2id, out_dir, name="best.pt"):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"epoch": epoch, "best_val_acc": best_val_acc, "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": cfg, "country2id": country2id}
    torch.save(payload, out_dir / name)
    print(f"[Training] Saved checkpoint to {out_dir / name}")

def train(cfg_path: str, name: str, device_str: Optional[str], weights: Optional[str], resume: bool, precompute: bool, augmentations: str):
    # --- Phase 1: Config, Device, and Directory Setup ---
    if precompute and ('image' in augmentations or 'both' in augmentations):
        raise ValueError("Image augmentation is not compatible with pre-computed embeddings.")

    if resume:
        temp_cfg = load_config(cfg_path)
        save_dir = Path(temp_cfg["eval"]["save_dir"]) / name
        ckpt_path = save_dir / "last.pt"
        if not ckpt_path.exists(): raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt['config']
        country2id = ckpt['country2id']
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt['best_val_acc']
        print(f"Resuming from epoch {start_epoch}. Loaded config from checkpoint.")
    else:
        cfg = load_config(cfg_path)
        save_dir = Path(cfg["eval"]["save_dir"]) / name
        # For a new run, determine country mapping first
        _, _, _, country2id = create_dataloaders(cfg, augmentations='none')
        start_epoch = 0
        best_val_acc = -1.0
        # Save config copy
        save_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfg_path, save_dir / "config.yaml")
        print(f"Saved configuration to {save_dir / 'config.yaml'}")

    # --- Phase 2: Build Model and Optimizer ---
    device = get_device(cfg, cli_device=device_str)
    model, _ = build_dino_geoguesser_model(cfg, num_countries=len(country2id))
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg["training"]["lr_head"]))

    # --- Phase 3: Load State (if applicable) ---
    if resume:
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    elif weights:
        print(f"Loading initial weights from {weights} for new run.")
        w_ckpt = torch.load(weights, map_location=device)
        model.load_state_dict(w_ckpt.get('state_dict', w_ckpt), strict=False)

    # --- Phase 4: Pre-computation and Final Dataloaders ---
    embeddings_path = save_dir / "embeddings.pt"
    if precompute:
        if not embeddings_path.exists():
            precompute_embeddings(model, cfg, device, embeddings_path)
        train_loader, val_loader, test_loader, _ = create_dataloaders(cfg, augmentations=augmentations, precomputed_path=embeddings_path)
    else:
        train_loader, val_loader, test_loader, _ = create_dataloaders(cfg, augmentations=augmentations)
        
    # --- Phase 5: Training Loop ---
    print(f"Starting training on device: {device}")
    num_epochs = cfg["training"]["num_epochs"]
    use_embed_aug = 'embedding' in augmentations or 'both' in augmentations
    noise = float(cfg["training"]["embedding_noise_level"]) if use_embed_aug else 0.0

    for epoch in range(start_epoch, num_epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, device, epoch, num_epochs, precompute, noise)
        val_stats = evaluate(model, val_loader, device, precompute)
        print(f"[Epoch {epoch+1}/{num_epochs}] train_loss={train_stats['loss']:.4f}, acc={train_stats['acc']:.3f} | val_loss={val_stats['loss']:.4f}, acc={val_stats['acc']:.3f}")

        if val_stats['acc'] > best_val_acc:
            best_val_acc = val_stats['acc']
            print(f"  -> New best val acc: {best_val_acc:.3f}")
            save_checkpoint(model, optimizer, epoch, best_val_acc, cfg, country2id, save_dir, name="best.pt")
        save_checkpoint(model, optimizer, epoch, best_val_acc, cfg, country2id, save_dir, name="last.pt")

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
        if args.resume and args.weights: raise ValueError("--resume and --weights are mutually exclusive.")
        train(args.config, args.name, args.device, args.weights, args.resume, args.precompute_embeddings, args.augmentations)

if __name__ == "__main__":
    main()
