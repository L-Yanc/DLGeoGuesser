import argparse
import os
import subprocess
import io
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
ROOT_DIR = Path(__file__).parent.parent.resolve()
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw' / 'geohints'

# --- CORE FUNCTIONS ---

def is_svg(file_path):
    """
    Checks if a file is an SVG by inspecting its first few bytes.
    Handles cases where the file might not exist.
    """
    if not file_path.exists():
        return False
    try:
        with open(file_path, 'rb') as f:
            header = f.read(100)
        return b'<svg' in header.lower() or b'<?xml' in header.lower()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

def fix_svg_extensions():
    """
    1. Renames any .jpg files that are actually SVGs to .svg.
    2. Converts all .svg files to .jpg using rsvg-convert and Pillow.
    """
    if not RAW_DATA_DIR.exists():
        print(f"Error: Raw data directory not found at {RAW_DATA_DIR}")
        return

    # --- Pass 1: Rename wrongly named .jpg files ---
    print("Pass 1: Checking for .jpg files that are actually SVGs...")
    all_files_initial = list(RAW_DATA_DIR.glob('*/*')) # Check all files
    
    for file_path in tqdm(all_files_initial, desc="Renaming JPGs to SVGs"):
        if file_path.suffix.lower() == '.jpg' and is_svg(file_path):
            svg_path = file_path.with_suffix('.svg')
            try:
                if not svg_path.exists():
                    file_path.rename(svg_path)
            except OSError as e:
                print(f"Error renaming {file_path} to {svg_path}: {e}")

    # --- Pass 2: Convert all SVG files to JPG ---
    print("\nPass 2: Converting all SVG files to JPG...")
    svg_files = list(RAW_DATA_DIR.glob('*/*.svg'))
    
    if not svg_files:
        print("No SVG files found to convert.")
        return

    for svg_path in tqdm(svg_files, desc="Converting SVGs to JPGs"):
        jpg_path = svg_path.with_suffix('.jpg')
        
        try:
            # Get PNG data from rsvg-convert
            command = ['rsvg-convert', '-h', '500', str(svg_path)]
            result = subprocess.run(command, check=True, capture_output=True)
            
            # Use Pillow to convert the PNG data to JPG
            png_data = io.BytesIO(result.stdout)
            with Image.open(png_data) as img:
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                img.save(jpg_path, 'jpeg')

        except subprocess.CalledProcessError as e:
            print(f"Error converting {svg_path}: {e.stderr.decode()}")
        except FileNotFoundError:
            print("Error: rsvg-convert does not seem to be installed or is not in the system's PATH.")
            return
        except Exception as e:
            print(f"An error occurred while processing {svg_path}: {e}")

    print(f"\nFixing process complete.")

def main(args):
    """Main function to orchestrate the fixing process."""
    print("--- Starting SVG Extension Fix ---")
    fix_svg_extensions()
    print("--- SVG Extension Fix Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix SVG files saved with .jpg extension and convert all SVGs.")
    args = parser.parse_args()
    main(args)