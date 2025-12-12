import argparse
import time
from pathlib import Path
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

SEARCH_TERMS = {
    "Street view African city": "african_city",
    "Street view Asian city": "asian_city",
    "Street view European city": "european_city",
    "Street view South American city": "south_american_city",
    "Street view North American city": "north_american_city",
    "Street view Middle Eastern city": "middle_eastern_city",
    "Street view tropical beach": "tropical_beach",
    "Street view desert landscape": "desert_landscape",
    "Street view mountain village": "mountain_village",
    "Street view rural farmland": "rural_farmland",
    "Street view urban park": "urban_park",
    "Street view industrial area": "industrial_area",
    "Street view coastal town": "coastal_town",
    "Street view forest road": "forest_road",
    "Street view snowy landscape": "snowy_landscape",
    "Street view suburban neighborhood": "suburban_neighborhood",
    "Street view mountain pass": "mountain_pass",
    "Street view island village": "island_village",
    "Street view historic town center": "historic_town_center",
    "Street view roadside market": "roadside_market",
}

def scrape_ddg_images(search_term: str, class_name: str, num_images: int, output_dir: Path, metadata: list):
    """
    Scrapes DuckDuckGo Images for a given query and saves the images to a directory.
    """
    # Set up the Firefox driver
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")
    try:
        driver = webdriver.Firefox(options=options)
    except Exception as e:
        print(f"Failed to initialize Firefox driver: {e}")
        print("Please ensure you have Firefox and geckodriver installed and in your PATH.")
        return

    # Construct the search URL
    url = f"https://duckduckgo.com/?q={search_term.replace(' ', '+')}&t=h_&iax=images&ia=images"
    print(f"Fetching URL: {url}")
    driver.get(url)

    # Wait for the images to be loaded
    try:
        image_container = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "/html/body/div[2]/div[6]/div[4]/div/div[2]/div/div[2]"))
        )
        WebDriverWait(image_container, 20).until(
            EC.presence_of_all_elements_located((By.TAG_NAME, "img"))
        )
    except Exception as e:
        print(f"Images did not load in time for query: {search_term} - {e}")
        driver.quit()
        return
        
    # The user provided an XPath: /html/body/div[2]/div[6]/div[4]/div/div[2]/div/div[2]
    try:
        print("Finding image container with XPath...")
        image_container = driver.find_element(By.XPATH, "/html/body/div[2]/div[6]/div[4]/div/div[2]/div/div[2]")
        print("Image container found.")
        # Find all images within the container
        image_elements = image_container.find_elements(By.TAG_NAME, "img")
        print(f"Found {len(image_elements)} image elements.")
    except Exception as e:
        print(f"Could not find image container for query: {search_term} - {e}")
        driver.quit()
        return

    image_urls = []
    for img_element in image_elements[:num_images]:
        src = img_element.get_attribute("src")
        if src and "ico" not in src:
            image_urls.append(src)
    
    print(f"Found {len(image_urls)} image URLs.")

    driver.quit()

    # Download the images
    print(f"Downloading {len(image_urls)} images...")
    saved_images = 0
    for i, url in enumerate(image_urls):
        if saved_images >= num_images:
            break
        try:
            response = requests.get(url, timeout=10, stream=True)
            if response.status_code == 200:
                content_type = response.headers.get("content-type")
                if content_type and content_type.startswith("image/"):
                    filename = f"{class_name}_{i+1}.jpg"
                    with open(output_dir / filename, "wb") as f:
                        f.write(response.content)
                    
                    metadata.append({
                        "filename": filename,
                        "class_name": class_name,
                        "search_term": search_term,
                    })
                    saved_images += 1
            else:
                print(f"Failed to download {url}, status code: {response.status_code}")
        except Exception as e:
            print(f"Could not download image {url}: {e}")
    print(f"Finished downloading images. Saved {saved_images} images.")

def main():
    parser = argparse.ArgumentParser(description="Scrape images from DuckDuckGo Images.")
    parser.add_argument("--num_images", type=int, default=50, help="Number of images to scrape per class.")
    parser.add_argument("--output_dir", type=str, default="data/processed/clip_vibe", help="Directory to save the images.")
    args = parser.parse_args()

    images_dir = Path(args.output_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    for search_term, class_name in SEARCH_TERMS.items():
        print(f"Scraping images for class: {class_name} (Search term: {search_term})")
        scrape_ddg_images(search_term, class_name, args.num_images, images_dir, metadata)
        time.sleep(1) # Be a good citizen

    df = pd.DataFrame(metadata)
    df.to_csv(images_dir / "metadata.csv", index=False)
    print(f"Saved metadata to {images_dir / 'metadata.csv'}")

if __name__ == "__main__":
    main()