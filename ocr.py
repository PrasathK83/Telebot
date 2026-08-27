import re

from PIL import Image, ImageOps, ImageEnhance
import easyocr
import numpy as np


# ------------------ OCR SETUP ------------------

print("Loading EasyOCR...")
ocr_reader = easyocr.Reader(["en"], gpu=False)
print("EasyOCR loaded successfully.")


# ------------------ OCR PIPELINE ------------------

def prepare_ocr_images(image: Image.Image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_width, original_height = image.size
    max_dimension = 1800
    largest_dimension = max(original_width, original_height)

    if largest_dimension > max_dimension:
        scale = max_dimension / largest_dimension
        new_size = (
            max(1, int(original_width * scale)),
            max(1, int(original_height * scale)),
        )
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    else:
        min_dimension = 1100
        if largest_dimension < min_dimension:
            scale = min_dimension / largest_dimension
            new_size = (
                max(1, int(original_width * scale)),
                max(1, int(original_height * scale)),
            )
            image = image.resize(new_size, Image.Resampling.LANCZOS)

    original = image.copy()
    gray = ImageOps.grayscale(image)
    gray = ImageEnhance.Contrast(gray).enhance(1.8)
    gray = ImageEnhance.Sharpness(gray).enhance(1.5)
    enhanced = gray.convert("RGB")

    return [("original", original), ("enhanced", enhanced)]


def run_easyocr(image: Image.Image):
    image_array = np.array(image)
    return ocr_reader.readtext(
        image_array,
        detail=1,
        paragraph=False,
        canvas_size=1800,
        mag_ratio=1.2,
        text_threshold=0.50,
        low_text=0.20,
        link_threshold=0.20,
        width_ths=0.7,
        height_ths=0.7,
    )


def normalize_ocr_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def extract_detections(results):
    detections = []

    for detection in results:
        if len(detection) < 3:
            continue

        text = str(detection[1]).strip()

        try:
            confidence = float(detection[2])
        except (TypeError, ValueError):
            continue

        if confidence < 0.20 or not text:
            continue

        position = detection[0]
        min_x = min(point[0] for point in position)
        min_y = min(point[1] for point in position)

        detections.append({
            "text": text,
            "confidence": confidence,
            "x": min_x,
            "y": min_y,
            "normalized": normalize_ocr_text(text),
        })

    return detections


def deduplicate_detections(detections):
    unique = {}

    for item in detections:
        key = item["normalized"]
        if not key:
            continue
        if key not in unique or item["confidence"] > unique[key]["confidence"]:
            unique[key] = item

    return list(unique.values())


def extract_text_from_image(image_path: str) -> str:
    print("Starting OCR...")

    image = Image.open(image_path)
    prepared_images = prepare_ocr_images(image)
    all_detections = []

    for name, ocr_image in prepared_images:
        try:
            results = run_easyocr(ocr_image)
            detections = extract_detections(results)
            print(f"OCR pass '{name}': {len(detections)} regions")
            all_detections.extend(detections)
        except Exception as exc:
            print(f"OCR pass '{name}' failed:", exc)

    if not all_detections:
        return ""

    unique_detections = deduplicate_detections(all_detections)
    unique_detections.sort(key=lambda item: (item["y"], item["x"]))

    return "\n".join(item["text"] for item in unique_detections).strip()
