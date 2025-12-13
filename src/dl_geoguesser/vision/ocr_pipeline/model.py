import easyocr
import numpy as np
from langdetect import detect

# This dictionary maps the 8 custom YOLO model classes to whether they are likely to contain text.
# This is derived from the `class_translation.json` file.
YOLO_CLASSES_WITH_TEXT = {
    "banner": True,
    "barrier": False,
    "fire-hydrant": False,
    "mailbox": False,
    "sign": True,
    "traffic_light": False,
    "traffic_sign": True,
    "utility_pole": False,
}


class MultiLangOCR:
    """
    A class to perform OCR and language detection on images.
    Uses multiple EasyOCR models for different language families:
    - Latin-based languages
    - Arabic-based languages
    - Bengali-based languages
    - Cyrillic-based languages
    - Devanagari-based languages
    """

    def __init__(self, max_retries: int = 3):
        """
        Initializes the OcrAndLangDetector.

        This is very slow and memory-intensive as it initializes multiple OCR models
        for different language families.
        
        Args:
            max_retries: Maximum number of retry attempts for downloading models.
        """
        import time
        
        latin_lang_list = ['af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'en', 'es', 'et', 'fr', 'ga', 'hr', 'hu', 'id', 'is', 'it', 'ku', 'la', 'lt',
                           'lv', 'mi', 'ms', 'mt', 'nl', 'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv', 'sw', 'tl', 'tr', 'uz', 'vi', 'en']
        arabic_lang_list = ['ar', 'fa', 'ug', 'ur', 'en']
        bengali_lang_list = ['bn', 'as', 'en']
        cyrillic_lang_list = ['ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava', 'dar', 'inh', 'che', 'lbe', 'lez', 'tab', 'tjk', 'en']
        devanagari_lang_list = ['hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom', 'en']  # 'sa' and 'bgc' are not supported
        # other_lang_list = ['th', 'ch_sim', 'ch_tra', 'ja', 'ko', 'te', 'kn']

        self.readers = {}
        
        language_families = {
            'latin': latin_lang_list,
            'arabic': arabic_lang_list,
            'bengali': bengali_lang_list,
            'cyrillic': cyrillic_lang_list,
            'devanagari': devanagari_lang_list,
        }

        for family_name, lang_list in language_families.items():
            success = False
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        print(f"  Retry {attempt}/{max_retries-1} for {family_name}...")
                        time.sleep(2)  # Wait before retry
                    self.readers[family_name] = easyocr.Reader(lang_list, gpu=False, download_enabled=True)
                    success = True
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"  ⚠️  Failed to load {family_name} after {max_retries} attempts: {e}")
                    else:
                        print(f"  Download interrupted for {family_name}, retrying...")
            
            if success:
                print(f"  ✓ Loaded {family_name} language family")

        if not self.readers:
            raise RuntimeError("Failed to initialize any OCR language models")
        
        print(f"  Successfully loaded {len(self.readers)}/{len(language_families)} language families")

        # for lang in other_lang_list:
        #     try:
        #         self.readers[lang] = easyocr.Reader([lang])
        #     except Exception as e:
        #         print(f"Warning: Failed to initialize easyocr reader for: {lang}. Error: {e}")

    def extract_text(self, image: np.ndarray) -> dict:
        """
        Extracts text from an image using multiple language models and returns
        all results.

        Args:
            image (np.ndarray): The image to extract text from.

        Returns:
            dict: A dictionary where keys are reader names and values are the
                  lists of results from each reader.
                  Keys: 'latin', 'arabic', 'bengali', 'cyrillic', 'devanagari'
        """
        all_results = {}
        for lang, reader in self.readers.items():
            all_results[lang] = reader.readtext(image)

        return all_results

    def detect_language(self, text: str) -> str:
        """
        Detects the language of a given text.

        Args:
            text (str): The text to detect the language of.

        Returns:
            str: The detected language code (e.g., 'en', 'fr').
        """
        try:
            return detect(text)
        except:
            return "unknown"

    def class_has_text(self, class_name: str) -> bool:
        """
        Checks if a given class is likely to contain text.

        Args:
            class_name (str): The name of the class.

        Returns:
            bool: True if the class is likely to contain text, False otherwise.
        """
        return YOLO_CLASSES_WITH_TEXT.get(class_name, False)

