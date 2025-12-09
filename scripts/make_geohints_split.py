import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT / "data" / "processed" / "geohints_processed"
METADATA_PATH = DATA_DIR / "metadata.csv"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"

RANDOM_SEED = 42


def main():
    print(f"Loading metadata from {METADATA_PATH}")
    df = pd.read_csv(METADATA_PATH)
    print(f"Total rows: {len(df)}")

    # Check class counts to decide whether we can safely stratify
    counts = df["country"].value_counts()
    min_count = counts.min()
    print(f"Smallest country class has {min_count} samples")

    # If some countries are extremely rare, stratification can fail
    if min_count < 3:
        print(
            "[WARN] Some classes have < 3 samples, "
            "skipping stratify to avoid errors."
        )
        stratify_labels_full = None
    else:
        stratify_labels_full = df["country"]

    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,  # 30% for val+test combined
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=stratify_labels_full,
    )

    # Second split: val vs test (half-half of the temp set)
    if stratify_labels_full is None:
        stratify_labels_temp = None
    else:
        stratify_labels_temp = temp_df["country"]

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,  # 50% of 30% => 15% overall
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=stratify_labels_temp,
    )

    print(f"Train size: {len(train_df)}")
    print(f"Val size:   {len(val_df)}")
    print(f"Test size:  {len(test_df)}")

    TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print(f"Saved train split to {TRAIN_CSV}")
    print(f"Saved val split   to {VAL_CSV}")
    print(f"Saved test split  to {TEST_CSV}")


if __name__ == "__main__":
    main()
