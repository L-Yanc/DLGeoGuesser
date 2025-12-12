from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from .data import create_dataloaders
from .model import ClipVibe, ClipVibeModel, build_clip_vibe_model, get_device


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
        targets = batch["class_id"].to(dev)

        if use_embeddings:
            features = batch["embedding"].to(dev)
        else:
            images = batch["image"].to(dev)
            with torch.no_grad():
                features = model.backbone(images)

        if augment_embedding_noise > 0:
            features += torch.randn_like(features) * augment_embedding_noise

        logits = model.head(features)["logits"]
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
            targets = batch["class_id"].to(dev)
            if use_embeddings:
                features = batch["embedding"].to(dev)
            else:
                features = model.backbone(batch["image"].to(dev))

            logits = model.head(features)["logits"]
            loss = loss_fn(logits, targets)
            acc = accuracy_from_logits(logits, targets)
            total_loss += loss.item()
            total_acc += acc
            steps += 1
            loop.set_postfix(loss=loss.item(), acc=acc)
    return {"loss": total_loss / steps, "acc": total_acc / steps}


def precompute_embeddings(model: ClipVibeModel, cfg: Dict, device: torch.device, save_path: Path):
    print(f"Pre-computing embeddings and saving to {save_path}...")
    model.eval()
    train_loader, val_loader, test_loader, class2id = create_dataloaders(cfg, augmentations='none')
    splits = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    output = {'class2id': class2id}

    with torch.no_grad():
        for split_name, loader in splits.items():
            all_embeds, all_labels, all_contents = [], [], []
            for batch in tqdm(loader, desc=f"Computing {split_name} embeddings"):
                embeds = model.backbone(batch["image"].to(device))
                all_embeds.append(embeds.cpu())
                all_labels.append(batch["class_id"].cpu())
                all_contents.extend(batch["content"])

            output[f'{split_name}_embeds'] = torch.cat(all_embeds, dim=0)
            output[f'{split_name}_labels'] = torch.cat(all_labels, dim=0)
            output[f'{split_name}_contents'] = all_contents
            print(f"Computed {len(output[f'{split_name}_embeds'])} embeddings for {split_name} split.")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, save_path)
    print("Pre-computation complete.")


def save_checkpoint(model, optimizer, epoch, best_val_acc, cfg, class2id, out_dir, name="best.pt"):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"epoch": epoch, "best_val_acc": best_val_acc, "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": cfg, "class2id": class2id}
    torch.save(payload, out_dir / name)
    print(f"[Training] Saved checkpoint to {out_dir / name}")


def train(cfg_path: str, name: str, device_str: Optional[str], weights: Optional[str], resume: bool, precompute: bool, augmentations: str):
    # --- Phase 1: Config, Device, and Directory Setup ---
    if precompute and ('image' in augmentations or 'both' in augmentations):
        raise ValueError("Image augmentation is not compatible with pre-computed embeddings.")
    
    cfg = yaml.safe_load(Path(cfg_path).read_text())

    if resume:
        save_dir = Path(cfg["eval"]["save_dir"]) / name
        ckpt_path = save_dir / "last.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt['config']
        class2id = ckpt['class2id']
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt['best_val_acc']
        print(f"Resuming from epoch {start_epoch}. Loaded config from checkpoint.")
    else:
        save_dir = Path(cfg["eval"]["save_dir"]) / name
        _, _, _, class2id = create_dataloaders(cfg, augmentations='none')
        start_epoch = 0
        best_val_acc = -1.0
        save_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfg_path, save_dir / "config.yaml")
        print(f"Saved configuration to {save_dir / 'config.yaml'}")

    # --- Phase 2: Build Model and Optimizer ---
    device = get_device(cfg, cli_device=device_str)
    model, _ = build_clip_vibe_model(cfg, num_classes=len(class2id))
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg["training"]["lr_head"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["training"]["lr_scheduler_step_size"], gamma=cfg["training"]["lr_scheduler_gamma"])

    # --- Phase 3: Load State ---
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
            save_checkpoint(model, optimizer, epoch, best_val_acc, cfg, class2id, save_dir, name="best.pt")
        save_checkpoint(model, optimizer, epoch, best_val_acc, cfg, class2id, save_dir, name="last.pt")
        
        scheduler.step()

    print("\nTraining complete. Final test set evaluation...")
    test_stats = evaluate(model, test_loader, device, precompute)
    print(f"[Test] loss={test_stats['loss']:.4f}, acc={test_stats['acc']:.3f}")


def evaluate_detailed(weights_path: str, split: str, device_str: Optional[str]):
    # --- Load Model and Config ---
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    ckpt = torch.load(weights_path, map_location="cpu")
    cfg = ckpt['config']
    class2id = ckpt['class2id']
    id2class = {v: k for k, v in class2id.items()}

    device = get_device(cfg, cli_device=device_str)
    model, _ = build_clip_vibe_model(cfg, num_classes=len(class2id))
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded model from {weights_path} on device {device}")

    # --- Dataloaders ---
    run_dir = Path(weights_path).parent
    embeddings_path = run_dir / "embeddings.pt"
    use_embeddings = embeddings_path.exists()

    if use_embeddings:
        print("Found pre-computed embeddings for this run. Using them for evaluation.")
        _, val_loader, test_loader, _ = create_dataloaders(cfg, precomputed_path=embeddings_path)
    else:
        print("No pre-computed embeddings found. Using on-the-fly image evaluation.")
        _, val_loader, test_loader, _ = create_dataloaders(cfg)

    loader = val_loader if split == 'val' else test_loader

    # --- Inference Loop ---
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating on '{split}' split"):
            targets = batch["class_id"]

            if use_embeddings:
                features = batch["embedding"].to(device)
            else:
                features = model.backbone(batch["image"].to(device))

            logits = model.head(features)["logits"]
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # --- Metrics and Reporting ---
    print("\n" + "="*80)
    print(f"Overall Classification Report for '{split}' split")
    print("="*80)

    all_class_indices = sorted(id2class.keys())
    all_class_names = [id2class[i] for i in all_class_indices]
    report = classification_report(all_targets, all_preds, labels=all_class_indices, target_names=all_class_names, zero_division=0)
    print(report)


def generate_gradcam_image(weights_path: str, image_path: str, class_name: str, output_path: str, device_str: Optional[str]):
    """
    Generates and saves a Grad-CAM overlay for a given image and class.
    """
    # --- Load Model ---
    clip_vibe = ClipVibe(weights_path=weights_path, device=device_str)

    # --- Generate Grad-CAM ---
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return

    try:
        overlay = clip_vibe.generate_gradcam(image, class_name)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # --- Save Overlay ---
    try:
        overlay.save(output_path)
        print(f"Saved Grad-CAM overlay to {output_path}")
    except IOError:
        print(f"Error: Could not save image to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CLIP Vibe Model CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Train Command ---
    parser_train = subparsers.add_parser("train", help="Train the model")
    parser_train.add_argument("--config", type=str, default="configs/clip_vibe.yaml")
    parser_train.add_argument("--name", type=str, default="clip_vibe_run")
    parser_train.add_argument("--device", type=str, default=None)
    parser_train.add_argument("--weights", type=str, default=None, help="Load initial model weights for fine-tuning.")
    parser_train.add_argument("--resume", action="store_true", help="Resume from last.pt in run directory.")
    parser_train.add_argument("--precompute-embeddings", action="store_true", help="Pre-compute and use embeddings for faster training.")
    parser_train.add_argument("--augmentations", type=str, default="none", choices=['none', 'image', 'embedding', 'both'])

    # --- Evaluate Command ---
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate a trained model with detailed metrics.")
    parser_eval.add_argument("--weights", type=str, required=True, help="Path to the trained model weights (.pt file).")
    parser_eval.add_argument("--split", type=str, default="val", choices=['val', 'test'], help="Dataset split to evaluate on.")
    parser_eval.add_argument("--device", type=str, default=None, help="Device to use for evaluation.")

    # --- Grad-CAM Command ---
    parser_gradcam = subparsers.add_parser("gradcam", help="Generate a Grad-CAM heatmap for a given image and class.")
    parser_gradcam.add_argument("--weights", type=str, required=True, help="Path to the trained model weights (.pt file).")
    parser_gradcam.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser_gradcam.add_argument("--class_name", type=str, required=True, help="The target class for which to generate the heatmap.")
    parser_gradcam.add_argument("--output", type=str, default="gradcam.jpg", help="Path to save the Grad-CAM overlay image.")
    parser_gradcam.add_argument("--device", type=str, default=None, help="Device to use for Grad-CAM generation.")

    args = parser.parse_args()
    if args.command == "train":
        if args.resume and args.weights:
            raise ValueError("--resume and --weights are mutually exclusive.")
        train(args.config, args.name, args.device, args.weights, args.resume, args.precompute_embeddings, args.augmentations)
    elif args.command == "evaluate":
        evaluate_detailed(args.weights, args.split, args.device)
    elif args.command == "gradcam":
        generate_gradcam_image(args.weights, args.image, args.class_name, args.output, args.device)


if __name__ == "__main__":
    main()
