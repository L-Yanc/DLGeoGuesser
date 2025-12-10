from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import yaml
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
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
    """ Load the full YAML config as a plain dict. """
    with open(Path(config_path), "r") as f:
        return yaml.safe_load(f)


def build_data_config(cfg: Dict) -> DataConfig:
    """ Extract the data related fields from the full config. """
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
    """
    Simple dataset that wraps a Pandas DataFrame.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        image_root: Path,
        country2id: Dict[str, int],
        image_column: str,
        country_column: str,
        transform: Optional[transforms.Compose] = None,
    ):
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
        fname = row[self.image_column]
        country = row[self.country_column]

        img_path = self.image_root / fname
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        country_id = self.country2id[country]

        return {
            "image": img,
            "country_id": torch.tensor(country_id, dtype=torch.long),
            "country": country,
            "filename": fname,
        }


def build_label_mapping(df: pd.DataFrame, country_col: str) -> Dict[str, int]:
    """ Build country label mapping from the full dataset. """
    countries = sorted(df[country_col].unique())
    return {c: i for i, c in enumerate(countries)}


def get_dino_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """ Returns (train_transform, eval_transform) using DINOv2 normalisation. """
    train_t = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=DINO_MEAN, std=DINO_STD),
    ])
    eval_t = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=DINO_MEAN, std=DINO_STD),
    ])
    return train_t, eval_t


def create_datasets(
    cfg: Dict,
) -> Tuple[Dataset, Dataset, Dataset, Dict[str, int]]:
    """ Create train, val and test datasets by splitting the main metadata file. """
    data_cfg = build_data_config(cfg)
    
    df = pd.read_csv(data_cfg.metadata_csv)

    # Drop rows where the image file does not exist
    def _exists(fname: str) -> bool:
        return (data_cfg.root / fname).exists()
    
    initial_rows = len(df)
    df = df[df[data_cfg.image_column].apply(_exists)].reset_index(drop=True)
    if len(df) < initial_rows:
        print(f"[Dataset] Dropped {initial_rows - len(df)} rows with missing images.")

    country2id = build_label_mapping(df, data_cfg.country_column)
    train_t, eval_t = get_dino_transforms(data_cfg.image_size)

    # Create temporary datasets with train and eval transforms
    full_dataset_train = GeoHintsDataset(df, data_cfg.root, country2id, data_cfg.image_column, data_cfg.country_column, train_t)
    full_dataset_eval = GeoHintsDataset(df, data_cfg.root, country2id, data_cfg.image_column, data_cfg.country_column, eval_t)

    # Split indices
    total_size = len(df)
    val_size = int(total_size * data_cfg.val_split_size)
    test_size = int(total_size * data_cfg.test_split_size)
    train_size = total_size - val_size - test_size

    generator = torch.Generator().manual_seed(cfg.get("training", {}).get("seed", 42))
    indices = torch.randperm(total_size, generator=generator).tolist()
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Create subsets with correct transforms
    train_ds = torch.utils.data.Subset(full_dataset_train, train_indices)
    val_ds = torch.utils.data.Subset(full_dataset_eval, val_indices)
    test_ds = torch.utils.data.Subset(full_dataset_eval, test_indices)

    print(f"[Dataset] Split: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test.")
    
    return train_ds, val_ds, test_ds, country2id


def create_dataloaders(
    cfg: Dict,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """ Convenience wrapper to create datasets and wrap them in DataLoaders. """
    data_cfg = build_data_config(cfg)
    train_ds, val_ds, test_ds, country2id = create_datasets(cfg)

    train_loader = DataLoader(train_ds, batch_size=data_cfg.batch_size, shuffle=shuffle_train, num_workers=data_cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=data_cfg.batch_size, shuffle=False, num_workers=data_cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=data_cfg.batch_size, shuffle=False, num_workers=data_cfg.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, country2id
