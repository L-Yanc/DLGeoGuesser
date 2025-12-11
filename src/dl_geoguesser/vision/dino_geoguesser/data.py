from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import yaml
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torchvision import transforms


# ImageNet normalisation stats for DINOv2
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


@dataclass
class DataConfig:
    root: Path
    metadata_csv: Path
    image_column: str
    country_column: str
    image_size: int
    num_workers: int
    batch_size: int
    val_split_size: float
    test_split_size: float


def load_config(config_path: str | Path) -> Dict:
    with open(Path(config_path), "r") as f:
        return yaml.safe_load(f)


def build_data_config(cfg: Dict) -> DataConfig:
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    return DataConfig(
        root=Path(data_cfg["root"]),
        metadata_csv=Path(data_cfg["metadata_csv"]),
        image_column=data_cfg.get("image_column", "filename"),
        country_column=data_cfg.get("country_column", "country"),
        image_size=int(data_cfg.get("image_size", 224)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        batch_size=int(train_cfg.get("batch_size", 32)),
        val_split_size=float(data_cfg.get("val_split_size", 0.15)),
        test_split_size=float(data_cfg.get("test_split_size", 0.15)),
    )


class GeoHintsDataset(Dataset):
    """ Loads images and labels from a Pandas DataFrame for on-the-fly processing. """
    def __init__(self, df: pd.DataFrame, image_root: Path, country2id: Dict[str, int], image_column: str, country_column: str, transform: Optional[transforms.Compose] = None):
        self.df = df
        self.image_root = image_root
        self.country2id = country2id
        self.image_column = image_column
        self.country_column = country_column
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_root / row[self.image_column]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        country_id = self.country2id[row[self.country_column]]
        return {"image": img, "country_id": torch.tensor(country_id, dtype=torch.long)}

class EmbeddingDataset(Dataset):
    """ A dataset for loading pre-computed embeddings and their labels. """
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor, augment: bool = False, noise_level: float = 0.01):
        self.embeddings = embeddings
        self.labels = labels
        self.augment = augment
        self.noise_level = noise_level

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int):
        embedding = self.embeddings[idx]
        if self.augment:
            noise = torch.randn_like(embedding) * self.noise_level
            embedding += noise
        return {"embedding": embedding, "country_id": self.labels[idx]}

def build_label_mapping(df: pd.DataFrame, country_col: str) -> Dict[str, int]:
    countries = sorted(df[country_col].unique())
    return {c: i for i, c in enumerate(countries)}

def get_dino_transforms(image_size: int, augment: bool) -> transforms.Compose:
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=DINO_MEAN, std=DINO_STD),
        ])
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=DINO_MEAN, std=DINO_STD),
    ])

def create_image_datasets(cfg: Dict, use_image_aug: bool) -> Tuple[Dataset, Dataset, Dataset, Dict[str, int]]:
    data_cfg = build_data_config(cfg)
    df = pd.read_csv(data_cfg.metadata_csv)
    
    def _exists(fname: str) -> bool:
        return (data_cfg.root / fname).exists()
    df = df[df[data_cfg.image_column].apply(_exists)].reset_index(drop=True)

    country2id = build_label_mapping(df, data_cfg.country_column)
    
    # Create one dataset with augmentations and one without
    train_transform = get_dino_transforms(data_cfg.image_size, augment=use_image_aug)
    eval_transform = get_dino_transforms(data_cfg.image_size, augment=False)
    
    full_dataset_train = GeoHintsDataset(df, data_cfg.root, country2id, data_cfg.image_column, data_cfg.country_column, train_transform)
    full_dataset_eval = GeoHintsDataset(df, data_cfg.root, country2id, data_cfg.image_column, data_cfg.country_column, eval_transform)

    # Split indices
    total_size = len(df)
    val_size = int(total_size * data_cfg.val_split_size)
    test_size = int(total_size * data_cfg.test_split_size)
    train_size = total_size - val_size - test_size

    generator = torch.Generator().manual_seed(cfg["training"]["seed"])
    indices = torch.randperm(total_size, generator=generator).tolist()
    
    # Use the appropriate dataset for each split
    train_ds = torch.utils.data.Subset(full_dataset_train, indices[:train_size])
    val_ds = torch.utils.data.Subset(full_dataset_eval, indices[train_size : train_size + val_size])
    test_ds = torch.utils.data.Subset(full_dataset_eval, indices[train_size + val_size :])
    
    return train_ds, val_ds, test_ds, country2id

def create_dataloaders(
    cfg: Dict,
    augmentations: str = 'none',
    precomputed_path: Optional[Path] = None,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    data_cfg = build_data_config(cfg)
    
    if precomputed_path and precomputed_path.exists():
        print(f"Loading pre-computed embeddings from {precomputed_path}")
        data = torch.load(precomputed_path)
        country2id = data['country2id']
        
        use_embed_aug = 'embedding' in augmentations or 'both' in augmentations
        noise_level = float(cfg.get("training", {}).get("embedding_noise_level", 0.01))
        
        train_ds = EmbeddingDataset(data['train_embeds'], data['train_labels'], augment=use_embed_aug, noise_level=noise_level)
        val_ds = EmbeddingDataset(data['val_embeds'], data['val_labels'], augment=False)
        test_ds = EmbeddingDataset(data['test_embeds'], data['test_labels'], augment=False)
    else:
        use_image_aug = 'image' in augmentations or 'both' in augmentations
        train_ds, val_ds, test_ds, country2id = create_image_datasets(cfg, use_image_aug)

    train_loader = DataLoader(train_ds, batch_size=data_cfg.batch_size, shuffle=shuffle_train, num_workers=data_cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=data_cfg.batch_size, shuffle=False, num_workers=data_cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=data_cfg.batch_size, shuffle=False, num_workers=data_cfg.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, country2id
