from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

from src.dl_geoguesser.vision.clip_landscape.data import load_config


BASE_DIR = Path("models/clip_landscape/knn_baseline")


def load_split(split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load precomputed embeddings for a split.

    Expects files like:
      models/clip_landscape/knn_baseline/train_embeddings.pt
      models/clip_landscape/knn_baseline/val_embeddings.pt
      models/clip_landscape/knn_baseline/test_embeddings.pt
    """
    path = BASE_DIR / f"{split}_embeddings.pt"
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    payload = torch.load(path, map_location="cpu")

    feats = payload["embeddings"].float().cpu().numpy()
    country_ids = payload["country_ids"].long().cpu().numpy()
    content_ids = payload["content_ids"].long().cpu().numpy()

    return feats, country_ids, content_ids


def l2_normalise(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / norm


def knn_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    k: int,
    metric: str = "cosine",
) -> float:
    """
    Simple kNN classification using sklearn.
    Returns accuracy on eval split.
    """
    clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_eval)
    acc = (preds == y_eval).mean().item()
    return float(acc)


def main():
    cfg = load_config("configs/clip_landscape.yaml")
    k = int(cfg.get("eval", {}).get("k", 5))

    # Load embeddings
    X_train, y_train_country, y_train_content = load_split("train")
    X_val, y_val_country, y_val_content = load_split("val")
    X_test, y_test_country, y_test_content = load_split("test")

    # L2 normalise for cosine distance
    X_train_n = l2_normalise(X_train)
    X_val_n = l2_normalise(X_val)
    X_test_n = l2_normalise(X_test)

    print(f"[kNN] Using k={k} with cosine metric")
    print(f"[kNN] Train: {X_train_n.shape}, Val: {X_val_n.shape}, Test: {X_test_n.shape}")

    # Country accuracy
    val_country_acc = knn_eval(
        X_train_n, y_train_country,
        X_val_n, y_val_country,
        k=k,
        metric="cosine",
    )
    test_country_acc = knn_eval(
        X_train_n, y_train_country,
        X_test_n, y_test_country,
        k=k,
        metric="cosine",
    )

    # Content / vibe accuracy
    val_content_acc = knn_eval(
        X_train_n, y_train_content,
        X_val_n, y_val_content,
        k=k,
        metric="cosine",
    )
    test_content_acc = knn_eval(
        X_train_n, y_train_content,
        X_test_n, y_test_content,
        k=k,
        metric="cosine",
    )

    print(
        f"[kNN] VAL  - country_acc={val_country_acc:.3f}, "
        f"content_acc={val_content_acc:.3f}"
    )
    print(
        f"[kNN] TEST - country_acc={test_country_acc:.3f}, "
        f"content_acc={test_content_acc:.3f}"
    )

    # Optionally save results to a small text file for later reference
    out_dir = BASE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "knn_results.txt"
    with open(out_path, "w") as f:
        f.write(
            f"k={k}\n"
            f"val_country_acc={val_country_acc:.4f}\n"
            f"val_content_acc={val_content_acc:.4f}\n"
            f"test_country_acc={test_country_acc:.4f}\n"
            f"test_content_acc={test_content_acc:.4f}\n"
        )

    print(f"[kNN] Saved results to {out_path}")


if __name__ == "__main__":
    main()
