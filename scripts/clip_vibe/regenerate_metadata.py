
import argparse
from pathlib import Path
import pandas as pd

def regenerate_metadata(images_dir: str):
    """
    Regenerates the metadata.csv file from the images in a directory.
    """
    images_path = Path(images_dir)
    if not images_path.exists() or not images_path.is_dir():
        print(f"Error: The specified image directory does not exist or is not a directory: {images_dir}")
        return

    metadata = []
    for image_path in images_path.glob("*.jpg"):
        filename = image_path.name
        # The class name is the part of the filename before the last '_'
        class_name = "_".join(filename.split("_")[:-1])
        metadata.append({
            "filename": filename,
            "class_name": class_name,
            "search_term": "", # We don't have this information anymore, so we leave it empty
        })

    if not metadata:
        print(f"No images found in {images_dir}")
        return

    df = pd.DataFrame(metadata)
    df.to_csv(images_path / "metadata.csv", index=False)
    print(f"Successfully regenerated metadata.csv with {len(df)} entries.")


def main():
    parser = argparse.ArgumentParser(description="Regenerate the metadata.csv file for the CLIP Vibe dataset.")
    parser.add_argument("--images_dir", type=str, default="data/processed/clip_vibe", help="Directory containing the images.")
    args = parser.parse_args()

    regenerate_metadata(args.images_dir)


if __name__ == "__main__":
    main()
