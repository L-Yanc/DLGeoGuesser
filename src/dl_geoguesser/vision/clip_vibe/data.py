from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import (DataLoader, Dataset, WeightedRandomSampler,
                              random_split)
from torchvision import transforms
from transformers import CLIPImageProcessor

@dataclass
class DataConfig:
    root: Path
    metadata_csv: Path
    image_column: str
    country_column: str
    content_column: str
    image_size: int
    num_workers: int
    batch_size: int
    val_split_size: float
    test_split_size: float


def build_data_config(cfg: Dict) -> DataConfig:
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    return DataConfig(
        root=Path(data_cfg["root"]),
        metadata_csv=Path(data_cfg["metadata_csv"]),
        image_column=data_cfg.get("image_column", "filename"),
        country_column=data_cfg.get("country_column", "country"),
        content_column=data_cfg.get("content_column", "content"),
        image_size=int(data_cfg.get("image_size", 224)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        batch_size=int(train_cfg.get("batch_size", 32)),
        val_split_size=float(data_cfg.get("val_split_size", 0.15)),
        test_split_size=float(data_cfg.get("test_split_size", 0.15)),
    )


class ImageDataset(Dataset):
    """ Loads images and labels from a Pandas DataFrame for on-the-fly processing. """

    def __init__(self, df: pd.DataFrame, image_root: Path, country2id: Dict[str, int], image_column: str, country_column: str, content_column: str, transform: Optional[transforms.Compose] = None):
        self.df = df
        self.image_root = image_root
        self.country2id = country2id
        self.image_column = image_column
        self.country_column = country_column
        self.content_column = content_column
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
        content = row[self.content_column]
        return {"image": img, "country_id": torch.tensor(country_id, dtype=torch.long), "content": content}


class EmbeddingDataset(Dataset):
    """ A dataset for loading pre-computed embeddings and their labels. """

    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor, contents: List[str], augment: bool = False, noise_level: float = 0.01):
        self.embeddings = embeddings
        self.labels = labels
        self.contents = contents
        self.augment = augment
        self.noise_level = noise_level

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int):
        embedding = self.embeddings[idx]
        if self.augment:
            noise = torch.randn_like(embedding) * self.noise_level
            embedding += noise
        return {"embedding": embedding, "country_id": self.labels[idx], "content": self.contents[idx]}


def build_label_mapping(df: pd.DataFrame, country_col: str) -> Dict[str, int]:
    countries = sorted(df[country_col].unique())
    return {c: i for i, c in enumerate(countries)}


def get_clip_transforms(model_name: str, image_size: int, augment: bool) -> transforms.Compose:
    """
    Builds TorchVision transforms using the official normalization values
    from the model's Hugging Face image processor.
    """
    processor = CLIPImageProcessor.from_pretrained(model_name)

    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ])

    # For evaluation, use the processor's recommended sizing
    eval_size = processor.size.get("shortest_edge", image_size)
    return transforms.Compose([
        transforms.Resize(eval_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])


def create_image_datasets(cfg: Dict, use_image_aug: bool) -> Tuple[Dataset, Dataset, Dataset, Dict[str, int]]:
    data_cfg = build_data_config(cfg)
    df = pd.read_csv(data_cfg.metadata_csv)

    def _exists(fname: str) -> bool: return (data_cfg.root / fname).exists()
    df = df[df[data_cfg.image_column].apply(_exists)].reset_index(drop=True)

    use_stratification = cfg.get("training", {}).get("use_stratification", True)

    if use_stratification:
        print("Using stratified dataset splitting.")

        class_counts = df[data_cfg.country_column].value_counts()
        classes_to_remove = class_counts[class_counts == 1].index

        if len(classes_to_remove) > 0:
            print(f"Stage 1 Cleaning: Removing {len(classes_to_remove)} classes with only 1 sample globally.")
            df = df[~df[data_cfg.country_column].isin(classes_to_remove)].reset_index(drop=True)

        indices = np.arange(len(df))

        country2id = build_label_mapping(df, data_cfg.country_column)
        labels = df[data_cfg.country_column].map(country2id)

        val_test_size = data_cfg.val_split_size + data_cfg.test_split_size

        train_indices, val_test_indices, _, y_val_test = train_test_split(indices, labels, test_size=val_test_size, random_state=cfg["training"]["seed"], stratify=labels)

        val_test_labels = pd.Series(y_val_test, index=val_test_indices)

        subset_class_counts = val_test_labels.value_counts()

        subset_classes_to_move = subset_class_counts[subset_class_counts == 1].index

        if len(subset_classes_to_move) > 0:
            indices_to_move_mask = val_test_labels.isin(subset_classes_to_move)
            indices_to_move = val_test_labels[indices_to_move_mask].index
            train_indices = np.concatenate([train_indices, indices_to_move])
            cleaned_val_test_indices = val_test_labels[~indices_to_move_mask].index
            cleaned_y_val_test = val_test_labels[~indices_to_move_mask]
            print(f"Stage 2 Cleaning: Moved {len(indices_to_move)} samples from val/test to train set.")
        else:
            cleaned_val_test_indices, cleaned_y_val_test = val_test_indices, y_val_test

        if len(cleaned_val_test_indices) > 1:
            relative_test_size = data_cfg.test_split_size / (data_cfg.val_split_size + data_cfg.test_split_size)
            if relative_test_size >= 1.0:
                val_indices, test_indices = [], cleaned_val_test_indices
            elif relative_test_size <= 0.0:
                val_indices, test_indices = cleaned_val_test_indices, []
            else:
                val_indices, test_indices = train_test_split(cleaned_val_test_indices, test_size=relative_test_size, random_state=cfg["training"]["seed"], stratify=cleaned_y_val_test)
        else:
            val_indices, test_indices = cleaned_val_test_indices, []
    else:
        print("Using random (non-stratified) dataset splitting.")
        country2id = build_label_mapping(df, data_cfg.country_column)
        total_size = len(df)
        val_size, test_size = int(total_size * data_cfg.val_split_size), int(total_size * data_cfg.test_split_size)
        train_size = total_size - val_size - test_size
        generator = torch.Generator().manual_seed(cfg["training"]["seed"])
        train_indices, val_indices, test_indices = random_split(range(total_size), [train_size, val_size, test_size], generator=generator)

    model_name = cfg["model"]["pretrained_name"]
    train_transform = get_clip_transforms(model_name, data_cfg.image_size, augment=use_image_aug)
    eval_transform = get_clip_transforms(model_name, data_cfg.image_size, augment=False)

    full_dataset_train = ImageDataset(df, data_cfg.root, country2id, data_cfg.image_column, data_cfg.country_column, data_cfg.content_column, train_transform)
    full_dataset_eval = ImageDataset(df, data_cfg.root, country2id, data_cfg.image_column, data_cfg.country_column, data_cfg.content_column, eval_transform)

    train_ds = torch.utils.data.Subset(full_dataset_train, train_indices)
    val_ds = torch.utils.data.Subset(full_dataset_eval, val_indices)
    test_ds = torch.utils.data.Subset(full_dataset_eval, test_indices)

    return train_ds, val_ds, test_ds, country2id


def create_dataloaders(
    cfg: Dict,
    augmentations: str = 'none',
    precomputed_path: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    data_cfg = build_data_config(cfg)

    if precomputed_path and precomputed_path.exists():
        print(f"Loading pre-computed embeddings from {precomputed_path}")
        data = torch.load(precomputed_path)
        class2id = data['class2id']

        use_embed_aug = 'embedding' in augmentations or 'both' in augmentations
        noise_level = float(cfg.get("training", {}).get("embedding_noise_level", 0))

        train_ds = EmbeddingDataset(data['train_embeds'], data['train_labels'], data['train_contents'], augment=use_embed_aug, noise_level=noise_level)
        val_ds = EmbeddingDataset(data['val_embeds'], data['val_labels'], data['val_contents'], augment=False)
        test_ds = EmbeddingDataset(data['test_embeds'], data['test_labels'], data['test_contents'], augment=False)
        
        sampler = None
        shuffle = True

        train_loader = DataLoader(train_ds, batch_size=data_cfg.batch_size, sampler=sampler, shuffle=shuffle, num_workers=data_cfg.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=data_cfg.batch_size, shuffle=False, num_workers=data_cfg.num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=data_cfg.batch_size, shuffle=False, num_workers=data_cfg.num_workers, pin_memory=True)

        return train_loader, val_loader, test_loader, class2id
    else:
        use_image_aug = 'image' in augmentations or 'both' in augmentations
        train_ds, val_ds, test_ds, country2id = create_image_datasets(cfg, use_image_aug)

    sampler = None
    shuffle = True
    if cfg.get("training", {}).get("use_weighted_sampling", False) and not (precomputed_path and precomputed_path.exists()):
        print("Using weighted random sampler for training.")
        train_labels = [train_ds.dataset.df.iloc[i][data_cfg.country_column] for i in train_ds.indices]
        class_counts = pd.Series(train_labels).value_counts().to_dict()
        weights = [1.0 / class_counts[label] for label in train_labels]
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))
        shuffle = False

    train_loader = DataLoader(train_ds, batch_size=data_cfg.batch_size, sampler=sampler, shuffle=shuffle, num_workers=data_cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=data_cfg.batch_size, shuffle=False, num_workers=data_cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=data_cfg.batch_size, shuffle=False, num_workers=data_cfg.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, country2id
