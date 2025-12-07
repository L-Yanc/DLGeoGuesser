
import argparse
import os
import requests
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from tqdm import tqdm

# --- CONFIGURATION ---
ROOT_DIR = Path(__file__).parent.parent.resolve()
URL_FILE = ROOT_DIR / 'scripts' / 'geohints_urls.txt'
RAW_OUTPUT_DIR = ROOT_DIR / 'data' / 'raw' / 'geohints'

# --- HELPER FUNCTIONS ---

def get_url_slug(url):
    """Creates a filesystem-friendly slug from a URL."""
    path = urlparse(url).path
    return path.replace('/', '_').strip('_')

def download_image(session, url, save_path):
    """Downloads an image from a URL to a specified path."""
    try:
        if save_path.exists():
            return True
        
        response = session.get(url, stream=True, timeout=15)
        response.raise_for_status()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False

# --- SCRAPING LOGIC ---

def scrape_page(url, session):
    """
    Scrapes a single page for images and their country labels using multiple strategies.
    """
    url_slug = get_url_slug(url)
    url_dir = RAW_OUTPUT_DIR / url_slug
    url_dir.mkdir(exist_ok=True)
    
    print(f"Scraping page: {url}")
    try:
        response = session.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page {url}: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    search_results = soup.find('div', id='search-results')
    
    if not search_results:
        print(f"Warning: No 'search-results' div found on {url}")
        return

    processed_images = set()
    country_counts = {}

    # --- Strategy 1: Regular structure (div per image) ---
    for container in tqdm(search_results.find_all('div', class_='text-white text-md p-2'), desc=f"S1 {url_slug}"):
        country_span = container.find('span', class_='font-bold')
        img_tag = container.find('img')

        if country_span and img_tag and img_tag.has_attr('src'):
            country = country_span.text.strip().lower().replace(' ', '_')
            img_url = urljoin(url, img_tag['src'])
            
            if img_url not in processed_images:
                uid = country_counts.get(country, 0) + 1
                country_counts[country] = uid
                filename = f"{country}-{uid}.jpg"
                save_path = url_dir / filename

                if download_image(session, img_url, save_path):
                    processed_images.add(img_url)
                    time.sleep(0.05)

    # --- Strategy 2: Grouped structure (header + sibling container) ---
    for header in tqdm(search_results.find_all('span', class_='font-bold text-xl mt-4'), desc=f"S2 {url_slug}"):
        country = header.text.strip().lower().replace(' ', '_')
        
        # Find the next sibling that is a div and contains the images
        image_container = header.find_next_sibling('div')
        
        if image_container:
            for img_tag in image_container.find_all('img'):
                if img_tag.has_attr('src'):
                    img_url = urljoin(url, img_tag['src'])
                    
                    if img_url not in processed_images:
                        uid = country_counts.get(country, 0) + 1
                        country_counts[country] = uid
                        filename = f"{country}-{uid}.jpg"
                        save_path = url_dir / filename

                        if download_image(session, img_url, save_path):
                            processed_images.add(img_url)
                            time.sleep(0.05)

def main(args):
    """Main function to orchestrate the scraping process."""
    print("--- Starting GeoHints Image Scraping ---")
    
    if not URL_FILE.exists():
        print(f"Error: URL file not found at {URL_FILE}")
        return

    RAW_OUTPUT_DIR.mkdir(exist_ok=True)
    
    with open(URL_FILE, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
    
    for url in urls:
        scrape_page(url, session)

    print("--- GeoHints Image Scraping Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape images from GeoHints.")
    args = parser.parse_args()
    main(args)
