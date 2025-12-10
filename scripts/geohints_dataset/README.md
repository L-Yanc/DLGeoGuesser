# GeoHints Dataset Scripts

This directory contains scripts for scraping, processing, and preparing the GeoHints dataset.

## Execution Order

The scripts should be run in the following order:

### 1. Scrape Images

This script downloads images from the URLs listed in `geohints_urls.txt` and saves them in `data/raw/geohints`.

**Command:**
```bash
python scripts/geohints_dataset/scrape_geohints.py
```

### 2. Fix SVG Extensions

The scraper sometimes saves SVG images with a `.jpg` extension. This script identifies and corrects these, and then converts all SVG files to JPG format. It requires `rsvg-convert` to be installed (`brew install librsvg` on macOS).

**Command:**
```bash
python scripts/geohints_dataset/fix_svg_extensions.py
```

### 3. Process and Structure Data

This script processes the raw scraped images, renames them into a consistent format (`{country}_{content}_{uid}.jpg`), moves them to `data/processed/geohints_processed`, and generates a `metadata.csv` file.

**Command:**
```bash
python scripts/geohints_dataset/process_geohints.py
```

### 4. Create Data Splits

This script takes the `metadata.csv` and splits it into `train.csv`, `val.csv`, and `test.csv` for use in training and evaluation.

**Command:**
```bash
python scripts/geohints_dataset/make_geohints_split.py
```
