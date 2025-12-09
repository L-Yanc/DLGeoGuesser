from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import yaml
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Hard coded CLIP image normalisation stats
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class DataConfig:
    root: Path
    train_csv: Path
    val_csv: Path
    test_csv: Path
    image_column: str
    country_column: str
    content_column: str
    image_size: int
    num_workers: int
    batch_size: int


def load_config(config_path: str | Path = "configs/clip_landscape.yaml") -> Dict:
    """
    Load the full YAML config as a plain dict.
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_data_config(cfg: Dict) -> DataConfig:
    """
    Extract the data related fields from the full config.
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    root = Path(data_cfg["root"])
    return DataConfig(
        root=root,
        train_csv=Path(data_cfg["train_csv"]),
        val_csv=Path(data_cfg["val_csv"]),
        test_csv=Path(data_cfg["test_csv"]),
        image_column=data_cfg.get("image_column", "filename"),
        country_column=data_cfg.get("country_column", "country"),
        content_column=data_cfg.get("content_column", "content"),
        image_size=int(data_cfg.get("image_size", 224)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        batch_size=int(train_cfg.get("batch_size", 32)),
    )


class GeoHintsDataset(Dataset):
    """
    Simple dataset wrapper over the geohints_processed CSV files.

    Each item returns
      image tensor
      country_id int
      content_id int
      raw metadata row if needed for debugging
    """

    def __init__(
        self,
        csv_path: Path,
        image_root: Path,
        country2id: Dict[str, int],
        content2id: Dict[str, int],
        image_column: str = "filename",
        country_column: str = "country",
        content_column: str = "content",
        transform: Optional[transforms.Compose] = None,
    ):
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.country2id = country2id
        self.content2id = content2id
        self.image_column = image_column
        self.country_column = country_column
        self.content_column = content_column
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found at {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        # Drop rows whose image file does not exist
        import os

        def _exists(fname: str) -> bool:
            return (self.image_root / fname).exists()

        orig_len = len(self.df)
        self.df = self.df[self.df[self.image_column].apply(_exists)].reset_index(drop=True)
        dropped = orig_len - len(self.df)
        if dropped > 0:
            print(f"[GeoHintsDataset] Dropped {dropped} missing-image rows from {self.csv_path}")

        # Basic sanity
        for col in [self.image_column, self.country_column, self.content_column]:
            if col not in self.df.columns:
                raise ValueError(f"Expected column '{col}' in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        fname = row[self.image_column]
        country = row[self.country_column]
        content = row[self.content_column]

        img_path = self.image_root / fname
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        try:
            country_id = self.country2id[country]
        except KeyError:
            raise KeyError(f"Unknown country label '{country}' in row {idx}")

        try:
            content_id = self.content2id[content]
        except KeyError:
            raise KeyError(f"Unknown content label '{content}' in row {idx}")

        sample = {
            "image": img,
            "country_id": torch.tensor(country_id, dtype=torch.long),
            "content_id": torch.tensor(content_id, dtype=torch.long),
            "country": country,
            "content": content,
            "filename": fname,
        }
        return sample


def build_label_mappings(train_csv: Path, country_col: str, content_col: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Build country and content label mappings based on the train split only.
    """
    df = pd.read_csv(train_csv)

    countries = sorted(df[country_col].unique())
    contents = sorted(df[content_col].unique())

    country2id = {c: i for i, c in enumerate(countries)}
    content2id = {c: i for i, c in enumerate(contents)}

    return country2id, content2id


def get_clip_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns (train_transform, eval_transform) using CLIP normalisation.
    """
    train_t = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )

    eval_t = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )

    return train_t, eval_t


def create_datasets(
    cfg: Dict,
) -> Tuple[GeoHintsDataset, GeoHintsDataset, GeoHintsDataset, Dict[str, int], Dict[str, int]]:
    """
    Create train, val and test datasets plus the label mappings.
    """
    data_cfg = build_data_config(cfg)

    country2id, content2id = build_label_mappings(
        data_cfg.train_csv,
        country_col=data_cfg.country_column,
        content_col=data_cfg.content_column,
    )

    train_t, eval_t = get_clip_transforms(data_cfg.image_size)

    train_ds = GeoHintsDataset(
        csv_path=data_cfg.train_csv,
        image_root=data_cfg.root,
        country2id=country2id,
        content2id=content2id,
        image_column=data_cfg.image_column,
        country_column=data_cfg.country_column,
        content_column=data_cfg.content_column,
        transform=train_t,
    )

    val_ds = GeoHintsDataset(
        csv_path=data_cfg.val_csv,
        image_root=data_cfg.root,
        country2id=country2id,
        content2id=content2id,
        image_column=data_cfg.image_column,
        country_column=data_cfg.country_column,
        content_column=data_cfg.content_column,
        transform=eval_t,
    )

    test_ds = GeoHintsDataset(
        csv_path=data_cfg.test_csv,
        image_root=data_cfg.root,
        country2id=country2id,
        content2id=content2id,
        image_column=data_cfg.image_column,
        country_column=data_cfg.country_column,
        content_column=data_cfg.content_column,
        transform=eval_t,
    )

    return train_ds, val_ds, test_ds, country2id, content2id


def create_dataloaders(
    cfg: Dict,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[str, int]]:
    """
    Convenience wrapper to create datasets and wrap them in DataLoaders.
    """
    data_cfg = build_data_config(cfg)
    train_ds, val_ds, test_ds, country2id, content2id = create_datasets(cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.batch_size,
        shuffle=shuffle_train,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, country2id, content2id
