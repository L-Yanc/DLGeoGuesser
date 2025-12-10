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
from .model import build_dino_geoguesser_model, get_device


@dataclass
class TrainingState:
    epoch: int
    best_val_acc: float

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def train_one_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, dev: torch.device, epoch: int, num_epochs: int) -> Dict[str, float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, steps = 0.0, 0.0, 0
    loop = tqdm(loader, desc=f"Train Epoch {epoch+1}/{num_epochs}", leave=False)
    for batch in loop:
        images = batch["image"].to(dev)
        targets = batch["country_id"].to(dev)

        optim.zero_grad()
        logits = model(images)["country_logits"]
        loss = loss_fn(logits, targets)
        loss.backward()
        optim.step()

        total_loss += loss.item()
        acc = accuracy_from_logits(logits, targets)
        total_acc += acc
        steps += 1
        loop.set_postfix(loss=loss.item(), acc=acc)
    return {"loss": total_loss / steps, "acc": total_acc / steps}


def evaluate(model: nn.Module, loader: DataLoader, dev: torch.device) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, steps = 0.0, 0.0, 0
    loop = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in loop:
            images = batch["image"].to(dev)
            targets = batch["country_id"].to(dev)

            logits = model(images)["country_logits"]
            loss = loss_fn(logits, targets)

            acc = accuracy_from_logits(logits, targets)
            total_loss += loss.item()
            total_acc += acc
            steps += 1
            loop.set_postfix(loss=loss.item(), acc=acc)
    return {"loss": total_loss / steps, "acc": total_acc / steps}


def save_checkpoint(model, optimizer, epoch, best_val_acc, cfg, country2id, out_dir, name="best.pt"):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "country2id": country2id,
    }
    torch.save(payload, out_dir / name)
    print(f"[Training] Saved checkpoint to {out_dir / name}")


def train(
    config_path: str, 
    name: str, 
    device_str: Optional[str] = None, 
    weights: Optional[str] = None, 
    resume: bool = False
):
    cfg = load_config(config_path)
    device = get_device(cfg, cli_device=device_str)
    print(f"Starting training on device: {device}")

    # --- Data Loading ---
    train_loader, val_loader, test_loader, country2id = create_dataloaders(cfg)
    
    # --- Model and Optimizer Initialization ---
    model, _ = build_dino_geoguesser_model(cfg, num_countries=len(country2id))
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["training"]["lr_head"]),
        weight_decay=float(cfg["training"]["weight_decay"])
    )

    # --- State Initialization and Checkpoint Loading ---
    start_epoch = 0
    best_val_acc = -1.0
    save_dir = Path(cfg["eval"]["save_dir"]) / name
    
    if resume:
        ckpt_path = save_dir / "last.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume flag was provided, but checkpoint not found at {ckpt_path}")
        
        print(f"Resuming training from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt['best_val_acc']

    elif weights:
        if not Path(weights).exists():
            raise FileNotFoundError(f"Weights file not found at {weights}")
        
        print(f"Loading initial model weights from: {weights}")
        ckpt = torch.load(weights, map_location=device)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else: # Handle raw state_dict checkpoints
            model.load_state_dict(ckpt)

    # --- Training Loop ---
    num_epochs = cfg["training"]["num_epochs"]
    for epoch in range(start_epoch, num_epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, device, epoch, num_epochs)
        val_stats = evaluate(model, val_loader, device)

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['acc']:.3f} | "
              f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['acc']:.3f}")

        # Save last checkpoint
        save_checkpoint(model, optimizer, epoch, best_val_acc, cfg, country2id, save_dir, name="last.pt")

        if val_stats['acc'] > best_val_acc:
            best_val_acc = val_stats['acc']
            print(f"  -> New best validation accuracy: {best_val_acc:.3f}")
            save_checkpoint(model, optimizer, epoch, best_val_acc, cfg, country2id, save_dir, name="best.pt")

    print("\nTraining complete. Running final evaluation on test set...")
    test_stats = evaluate(model, test_loader, device)
    print(f"[Test] loss={test_stats['loss']:.4f} acc={test_stats['acc']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="DINO Geoguesser Model CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_train = subparsers.add_parser("train", help="Train the model")
    parser_train.add_argument("--config", type=str, default="configs/dino_geoguesser.yaml", help="Path to the config file.")
    parser_train.add_argument("--name", type=str, default="dino_run", help="Name of the training run.")
    parser_train.add_argument("--device", type=str, default=None, help="Device to use for training (e.g., 'cpu', 'mps', 'cuda'). Overrides config.")
    parser_train.add_argument("--weights", type=str, default=None, help="Path to a .pt file to load initial model weights for fine-tuning.")
    parser_train.add_argument("--resume", action="store_true", help="Resume training from the 'last.pt' checkpoint in the run directory.")

    args = parser.parse_args()

    if args.command == "train":
        if args.resume and args.weights:
            raise ValueError("--resume and --weights cannot be used at the same time.")
        train(args.config, args.name, args.device, args.weights, args.resume)


if __name__ == "__main__":
    main()
