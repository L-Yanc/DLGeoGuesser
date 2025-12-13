import argparse
import pprint

import cv2
import numpy as np
from langdetect import detect_langs

from dl_geoguesser.vision.ocr_pipeline.model import MultiLangOCR
from dl_geoguesser.vision.yolo_detector.model import YOLOv8Detector


def process_yolo_predictions(yolo_predictions: dict, image: np.ndarray) -> dict:
    """
    Processes YOLO predictions to extract text and detect language from relevant objects.

    Args:
        yolo_predictions (dict): The output from the YOLOv8Detector's predict method.
        image (np.ndarray): The original image.

    Returns:
        dict: A dictionary mapping bounding box tuples to a dictionary containing
              the concatenated text and a list of top language predictions.
              Example:
              {
                  (100, 150, 120, 180): {
                      "text": "Hello World",
                      "languages": [
                          {"lang": "en", "confidence": 0.99},
                          {"lang": "de", "confidence": 0.01}
                      ]
                  },
                  ...
              }
    """
    ocr_detector = MultiLangOCR()
    output = {}

    for class_name, instances in yolo_predictions.items():
        if ocr_detector.class_has_text(class_name):
            for instance in instances:
                bbox_crop = instance["bbox_crop"]
                x1, y1, x2, y2 = bbox_crop

                # Crop the image
                cropped_image = image[y1:y2, x1:x2]

                # Extract text from all readers
                all_ocr_results = ocr_detector.extract_text(cropped_image)

                # Find the best result among all readers based on average confidence
                best_result = []
                max_avg_confidence = 0.0
                for _, result in all_ocr_results.items():
                    if result:
                        current_avg_conf = sum(res[2] for res in result) / len(result)
                        if current_avg_conf > 0.2 and current_avg_conf > max_avg_confidence:
                            max_avg_confidence = current_avg_conf
                            best_result = result

                # Process results if any text was found
                if best_result:
                    concatenated_text = " ".join([res[1] for res in best_result])

                    try:
                        lang_predictions = detect_langs(concatenated_text)
                        top_langs = [{"lang": p.lang, "confidence": p.prob} for p in lang_predictions[:3]]
                    except Exception:
                        top_langs = []

                    output[tuple(bbox_crop)] = {
                        "text": concatenated_text,
                        "languages": top_langs
                    }

    return output


def main():
    """
    Main entrypoint for the OCR pipeline module.
    Provides a CLI for running OCR on a single image.
    Can use YOLO predictions if weights are provided.
    """
    parser = argparse.ArgumentParser(description="OCR on a single image CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to source image to run multilingual OCR on.")
    parser.add_argument("--yolo_weights", type=str, default=None, help="Path to YOLOv8 weights to use for object detection.")

    args = parser.parse_args()

    # Load the image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        return
    
    # Convert BGR (from cv2) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if args.yolo_weights:
        print(f"Using YOLOv8 model: {args.yolo_weights}")
        yolo_detector = YOLOv8Detector(model_path=args.yolo_weights)
        yolo_predictions = yolo_detector.predict(image)
        print("YOLO Predictions:")
        pprint.pprint(yolo_predictions)
        print("-" * 30)
    else:
        # Get image dimensions
        height, width, _ = image.shape
        # Create a dummy YOLO prediction for the full image
        yolo_predictions = {
            "sign": [
                {"confidence": 1.0, "bbox_crop": [0, 0, width, height], "scale": width * height}
            ]
        }

    # Run OCR pipeline on the predictions
    print("OCR Results:")
    ocr_results = process_yolo_predictions(yolo_predictions, image)
    pprint.pprint(ocr_results)


if __name__ == "__main__":
    main()
