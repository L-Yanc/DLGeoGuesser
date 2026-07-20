"""
This script generates plots of model performance over epochs for the three main
vision models: YOLOv8, DINO, and CLIP.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Define paths
ROOT_DIR = Path(__file__).parent.parent.parent
YOLO_CSV_PATH = ROOT_DIR / "runs/yolo/M1Pro_run/results.csv"
DINO_JSON_PATH = ROOT_DIR / "runs/dino/dino_precomputed/stats.json"
CLIP_JSON_PATH = ROOT_DIR / "runs/clip_vibe/clip_vibe_precomputed_run/stats.json"
OUTPUT_DIR = ROOT_DIR / "runs/analysis_plots"

# Set plot style
sns.set_theme(style="darkgrid")


def plot_yolo_performance(csv_path: Path, output_dir: Path):
    """
    Plots the mAP50 and mAP50-95 metrics for a YOLOv8 training run.

    Args:
        csv_path (Path): Path to the results.csv file from a YOLO run.
        output_dir (Path): Directory to save the plot in.
    """
    if not csv_path.exists():
        print(f"YOLO results not found at: {csv_path}")
        return

    print(f"Processing YOLO data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Clean up column names by stripping leading/trailing whitespace
    df.columns = df.columns.str.strip()

    # Check if required columns exist
    required_cols = ['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing one of the required columns in {csv_path}: {required_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='epoch', y='metrics/mAP50(B)', label='Validation mAP@0.50')
    sns.lineplot(data=df, x='epoch', y='metrics/mAP50-95(B)', label='Validation mAP@0.50-0.95')

    plt.title('YOLOv8 Model Performance (mAP)', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Average Precision (mAP)', fontsize=12)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "yolo_training_performance.png"
    plt.savefig(output_path)
    print(f"  ✓ Saved YOLO performance plot to {output_path}")
    plt.close()


def plot_classifier_performance(json_path: Path, model_name: str, output_dir: Path):
    """
    Plots the training and validation accuracy for a classifier model.

    Args:
        json_path (Path): Path to the stats.json file.
        model_name (str): The name of the model (e.g., 'DINO', 'CLIP').
        output_dir (Path): Directory to save the plot in.
    """
    if not json_path.exists():
        print(f"{model_name} stats not found at: {json_path}")
        return

    print(f"Processing {model_name} data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    if 'epoch' not in df.columns or 'train_acc' not in df.columns or 'val_acc' not in df.columns:
        print(f"Error: JSON file {json_path} is missing required keys ('epoch', 'train_acc', 'val_acc').")
        return

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='epoch', y='train_acc', label='Training Accuracy')
    sns.lineplot(data=df, x='epoch', y='val_acc', label='Validation Accuracy')

    plt.title(f'{model_name} Model Performance (Accuracy)', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"{model_name.lower()}_training_performance.png"
    plt.savefig(output_path)
    print(f"  ✓ Saved {model_name} performance plot to {output_path}")
    plt.close()


def main():
    """
    Main function to generate and save all performance plots.
    """
    print(f"📊 Generating training performance plots...")
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Plots will be saved to: {OUTPUT_DIR.resolve()}")

    # Generate plots
    plot_yolo_performance(YOLO_CSV_PATH, OUTPUT_DIR)
    plot_classifier_performance(DINO_JSON_PATH, "DINO", OUTPUT_DIR)
    plot_classifier_performance(CLIP_JSON_PATH, "CLIP", OUTPUT_DIR)

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
