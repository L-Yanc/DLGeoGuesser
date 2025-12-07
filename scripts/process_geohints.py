
import argparse
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
ROOT_DIR = Path(__file__).parent.parent.resolve()
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw' / 'geohints'
PROCESSED_DIR = ROOT_DIR / 'data' / 'processed' / 'geohints_processed'
METADATA_FILE = PROCESSED_DIR / 'geohints_metadata.csv'

# --- CORE FUNCTIONS ---

def get_content_from_slug(slug):
    """Extracts a content label from the URL slug."""
    # e.g., 'meta_signs_animalWarning' -> 'animalWarning'
    # e.g., 'meta_architecture' -> 'architecture'
    parts = slug.split('_')
    if len(parts) > 1:
        return parts[-1]
    return slug

def process_geohints_data():
    """
    Processes the raw scraped GeoHints data into a structured, flat dataset.
    - Moves all images into a single folder with a descriptive name.
    - Creates a metadata CSV file for easy lookup.
    """
    if not RAW_DATA_DIR.exists():
        print(f"Error: Raw data directory not found at {RAW_DATA_DIR}")
        return

    PROCESSED_DIR.mkdir(exist_ok=True)
    
    metadata = []
    
    image_files = list(RAW_DATA_DIR.glob('*/*.jpg'))
    
    if not image_files:
        print(f"No JPG images found in {RAW_DATA_DIR}. Please ensure scraping and fixing are complete.")
        return

    for old_image_path in tqdm(image_files, desc="Processing images"):
        url_slug = old_image_path.parent.name
        original_filename = old_image_path.name
        
        # Extract info from old filename and path
        content = get_content_from_slug(url_slug)
        parts = original_filename.rsplit('-', 1)
        if len(parts) == 2:
            country = parts[0]
            uid = parts[1].split('.')[0]
        else:
            country = "unknown"
            uid = "unknown"

        # Create new filename and path
        new_filename = f"{country}_{content}_{uid}.jpg"
        new_image_path = PROCESSED_DIR / new_filename
        
        # Move and rename the file
        shutil.move(old_image_path, new_image_path)
        
        # Append metadata for the table
        metadata.append({
            'filename': new_filename,
            'content': content,
            'country': country
        })

    # Create and save the DataFrame
    if metadata:
        df = pd.DataFrame(metadata)
        # Save the metadata file in the same processed directory
        df.to_csv(PROCESSED_DIR / 'metadata.csv', index=False)
        print(f"\nProcessing complete.")
        print(f"Processed images moved to: {PROCESSED_DIR}")
        print(f"Metadata saved to: {PROCESSED_DIR / 'metadata.csv'}")
        print(f"Total images processed: {len(df)}")
    else:
        print("No metadata was generated.")


def main(args):
    """Main function to orchestrate the data processing."""
    print("--- Starting GeoHints Data Processing ---")
    process_geohints_data()
    print("--- GeoHints Data Processing Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process scraped GeoHints images.")
    args = parser.parse_args()
    main(args)
