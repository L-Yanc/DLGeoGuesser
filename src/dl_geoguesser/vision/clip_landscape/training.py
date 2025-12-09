from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import load_config, create_dataloaders
from .model import build_clip_landscape_model, ModelConfig


@dataclass
class TrainingState:
    epoch: int
    best_val_score: float
    global_step: int


def get_device(cfg: Dict) -> torch.device:
    """
    Pick device based on cfg and CUDA availability.
    """
    desired = cfg.get("training", {}).get("device", "cuda")
    if desired == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_losses(
    logits_country: torch.Tensor,
    logits_content: torch.Tensor,
    targets_country: torch.Tensor,
    targets_content: torch.Tensor,
    alpha_country: float = 1.0,
    alpha_content: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Cross-entropy loss for country + content, with simple weights.
    """
    ce = nn.CrossEntropyLoss()
    loss_country = ce(logits_country, targets_country)
    loss_content = ce(logits_content, targets_content)

    loss = alpha_country * loss_country + alpha_content * loss_content

    stats = {
        "loss_total": float(loss.item()),
        "loss_country": float(loss_country.item()),
        "loss_content": float(loss_content.item()),
    }
    return loss, stats


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: Tuple[float, float] = (1.0, 1.0),
    max_steps: int | None = None,
) -> Dict[str, float]:
    """
    Single training epoch over train_loader.
    Returns average losses and accuracies.
    """
    model.train()
    alpha_country, alpha_content = loss_weights

    total_loss = 0.0
    total_country_acc = 0.0
    total_content_acc = 0.0
    n_steps = 0

    for step, batch in enumerate(train_loader):
        images = batch["image"].to(device, non_blocking=True)
        country_ids = batch["country_id"].to(device, non_blocking=True)
        content_ids = batch["content_id"].to(device, non_blocking=True)

        optimizer.zero_grad()

        out = model(images)
        logits_country = out["country_logits"]
        logits_content = out["content_logits"]

        loss, _ = compute_losses(
            logits_country,
            logits_content,
            country_ids,
            content_ids,
            alpha_country=alpha_country,
            alpha_content=alpha_content,
        )

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            country_acc = accuracy_from_logits(logits_country, country_ids)
            content_acc = accuracy_from_logits(logits_content, content_ids)

        total_loss += float(loss.item())
        total_country_acc += country_acc
        total_content_acc += content_acc
        n_steps += 1

        if max_steps is not None and n_steps >= max_steps:
            break

    if n_steps == 0:
        return {
            "loss": 0.0,
            "country_acc": 0.0,
            "content_acc": 0.0,
        }

    return {
        "loss": total_loss / n_steps,
        "country_acc": total_country_acc / n_steps,
        "content_acc": total_content_acc / n_steps,
    }


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_steps: int | None = None,
) -> Dict[str, float]:
    """
    Evaluation loop computing average loss + accuracies.
    """
    model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_country_acc = 0.0
    total_content_acc = 0.0
    n_steps = 0

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            images = batch["image"].to(device, non_blocking=True)
            country_ids = batch["country_id"].to(device, non_blocking=True)
            content_ids = batch["content_id"].to(device, non_blocking=True)

            out = model(images)
            logits_country = out["country_logits"]
            logits_content = out["content_logits"]

            loss_country = ce(logits_country, country_ids)
            loss_content = ce(logits_content, content_ids)
            loss = loss_country + loss_content

            country_acc = accuracy_from_logits(logits_country, country_ids)
            content_acc = accuracy_from_logits(logits_content, content_ids)

            total_loss += float(loss.item())
            total_country_acc += country_acc
            total_content_acc += content_acc
            n_steps += 1

            if max_steps is not None and n_steps >= max_steps:
                break

    if n_steps == 0:
        return {
            "loss": 0.0,
            "country_acc": 0.0,
            "content_acc": 0.0,
        }

    return {
        "loss": total_loss / n_steps,
        "country_acc": total_country_acc / n_steps,
        "content_acc": total_content_acc / n_steps,
    }


def save_checkpoint(
    model: nn.Module,
    cfg: Dict,
    country2id: Dict[str, int],
    content2id: Dict[str, int],
    out_dir: Path,
    name: str = "best.pt",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / name

    payload = {
        "state_dict": model.state_dict(),
        "config": cfg,
        "country2id": country2id,
        "content2id": content2id,
    }
    torch.save(payload, ckpt_path)
    print(f"[training] Saved checkpoint to {ckpt_path}")


def train_clip_head(
    config_path: str | Path = "configs/clip_landscape.yaml",
    max_steps_per_epoch: int | None = None,
) -> Dict[str, float]:
    """
    High-level training entrypoint for the CLIP landscape model.

    - loads config
    - builds dataloaders
    - builds model + optimizer
    - trains only the head (backbone is frozen)
    - tracks best val country+content accuracy
    - saves best checkpoint into models/clip_landscape/classifier_head/best.pt
    """
    cfg = load_config(config_path)
    device = get_device(cfg)

    # You can override num_workers for Windows development if needed:
    # cfg["data"]["num_workers"] = 0

    train_loader, val_loader, test_loader, country2id, content2id = create_dataloaders(cfg)

    num_countries = len(country2id)
    num_contents = len(content2id)

    model, model_cfg = build_clip_landscape_model(
        cfg,
        num_countries=num_countries,
        num_contents=num_contents,
    )

    # Make sure model is on the same device we chose
    model.to(device)

    # Only optimise unfrozen parameters (the head, since backbone is frozen)
    params = [p for p in model.parameters() if p.requires_grad]
    lr_head = cfg["training"].get("lr_head", 1e-3)
    weight_decay = cfg["training"].get("weight_decay", 1e-4)

    optimizer = torch.optim.AdamW(params, lr=lr_head, weight_decay=weight_decay)

    num_epochs = int(cfg["training"].get("num_epochs", 5))
    alpha_country = 1.0
    alpha_content = 1.0

    state = TrainingState(epoch=0, best_val_score=-1.0, global_step=0)

    save_dir = Path(cfg["eval"].get("save_dir", "models/clip_landscape")) / "classifier_head"

    for epoch in range(num_epochs):
        state.epoch = epoch

        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_weights=(alpha_country, alpha_content),
            max_steps=max_steps_per_epoch,
        )

        val_stats = evaluate(
            model,
            val_loader,
            device,
            max_steps=max_steps_per_epoch,
        )

        # Combined score: average of country + content accuracy on val
        val_score = 0.5 * (val_stats["country_acc"] + val_stats["content_acc"])

        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_country_acc={train_stats['country_acc']:.3f} "
            f"train_content_acc={train_stats['content_acc']:.3f} | "
            f"val_loss={val_stats['loss']:.4f} "
            f"val_country_acc={val_stats['country_acc']:.3f} "
            f"val_content_acc={val_stats['content_acc']:.3f}"
        )

        if val_score > state.best_val_score:
            state.best_val_score = val_score
            save_checkpoint(
                model=model,
                cfg=cfg,
                country2id=country2id,
                content2id=content2id,
                out_dir=save_dir,
                name="best.pt",
            )

    # Final evaluation on test set using the final weights (you can also reload best.pt if you want)
    test_stats = evaluate(model, test_loader, device, max_steps=None)
    print(
        f"[Test] loss={test_stats['loss']:.4f} "
        f"country_acc={test_stats['country_acc']:.3f} "
        f"content_acc={test_stats['content_acc']:.3f}"
    )

    return test_stats
