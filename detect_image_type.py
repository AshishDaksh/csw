"""
detect_image_type.py
--------------------
Single function that takes an image path and returns its type.

Returns one of:
    "PURE_IMAGE"      → Natural photo/graphic, no meaningful text
    "NATIVE_TEXT"     → Screenshot or digitally rendered document
    "CONVERTED_OCR"   → Scanned/photographed physical document

Usage:
    from detect_image_type import detect_image_type

    image_type = detect_image_type("path/to/image.png")
    print(image_type)  # "NATIVE_TEXT"

Requirements:
    pip install opencv-python-headless Pillow pytesseract numpy
    apt install tesseract-ocr
"""

import math
import warnings
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ExifTags

warnings.filterwarnings("ignore")

ImageType = Literal["PURE_IMAGE", "NATIVE_TEXT", "CONVERTED_OCR"]

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


def detect_image_type(file_path: str) -> ImageType:
    """
    Classify an image as PURE_IMAGE, NATIVE_TEXT, or CONVERTED_OCR.

    Args:
        file_path: Path to the image file.

    Returns:
        "PURE_IMAGE"    — photo/graphic, no text worth extracting
        "NATIVE_TEXT"   — digital screenshot, use Tesseract/Docling
        "CONVERTED_OCR" — scanned/photographed doc, use LLM vision
    
    Raises:
        FileNotFoundError: if path does not exist
        ValueError: if format is unsupported or image can't be read
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format '{path.suffix}'. Supported: {SUPPORTED_EXTENSIONS}")

    img_bgr = cv2.imread(str(file_path))
    if img_bgr is None:
        raise ValueError(f"Could not read image: {file_path}")

    pure_score = 0.0
    native_score = 0.0
    converted_score = 0.0

    # ── Signal 1: Camera EXIF ────────────────────────────────────────────────
    try:
        pil_img = Image.open(file_path)
        exif = pil_img._getexif()
        if exif:
            tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            if tags.get("Make") or tags.get("Model"):
                pure_score += 1.5   # camera photo
    except Exception:
        pass

    # ── Signal 2: Color diversity + saturation ───────────────────────────────
    h, w = img_bgr.shape[:2]
    scale = min(1.0, 500 / max(h, w))
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    pixels = small.reshape(-1, 3)
    unique_ratio = min(1.0, len(np.unique(pixels, axis=0)) / len(pixels))
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1])) / 255.0

    if unique_ratio > 0.15 and sat_mean > 0.15:
        pure_score += 1.2           # rich colors → photo
    elif unique_ratio < 0.05:
        native_score += 0.8         # few colors → screenshot palette

    # ── Signal 3: Grayscale ──────────────────────────────────────────────────
    b, g, r = cv2.split(img_bgr)
    channel_diff = np.mean(np.abs(b.astype(int) - g.astype(int))) + \
                   np.mean(np.abs(g.astype(int) - r.astype(int)))
    if channel_diff < 5.0:
        converted_score += 0.8      # grayscale → typical scan

    # ── Signal 4: Text detection via Tesseract ───────────────────────────────
    char_count = 0
    text_conf = 0.0
    text_uniformity = 0.0
    try:
        import pytesseract
        from pytesseract import Output

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        data = pytesseract.image_to_data(Image.fromarray(rgb),
                                         output_type=Output.DICT,
                                         config="--psm 3")
        confs   = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) > 0]
        texts   = [t for t, c in zip(data["text"], data["conf"])
                   if str(c).lstrip("-").isdigit() and int(c) > 0 and t.strip()]
        heights = [data["height"][i] for i, c in enumerate(data["conf"])
                   if str(c).lstrip("-").isdigit() and int(c) > 0
                   and data["text"][i].strip() and data["height"][i] > 5]

        if confs:
            text_conf  = sum(confs) / len(confs) / 100.0
            char_count = sum(len(t) for t in texts)
        if len(heights) > 3:
            mean_h = np.mean(heights)
            text_uniformity = max(0.0, 1.0 - np.std(heights) / (mean_h + 1e-6))
    except Exception:
        pass

    has_text = char_count > 20 and text_conf > 0.3

    if not has_text:
        pure_score += 1.5           # no readable text → photo/graphic
    else:
        # ── Signal 5: Edge sharpness ─────────────────────────────────────────
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
        sharpness_norm = min(1.0, sharpness / 3000.0)

        if sharpness_norm > 0.6:
            native_score += 1.5     # pixel-perfect edges → screenshot
        elif sharpness_norm > 0.25:
            native_score += 0.5; converted_score += 0.5
        else:
            converted_score += 1.0  # soft edges → scanned/photographed

        # ── Signal 6: Text height uniformity ─────────────────────────────────
        if text_uniformity > 0.7:
            native_score += 1.2     # uniform font sizes → digital render
        elif text_uniformity > 0.4:
            native_score += 0.4; converted_score += 0.4
        else:
            converted_score += 0.8  # variable heights → physical doc

        # ── Signal 7: Local noise ─────────────────────────────────────────────
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
        noise = min(1.0, float(np.mean(np.abs(gray.astype(np.float32) - blurred))) / 30.0)

        if noise > 0.4:
            converted_score += 1.0  # noisy → scan artifact
        elif noise < 0.15:
            native_score += 0.6     # clean → digital source

        # ── Signal 8: Page skew ───────────────────────────────────────────────
        try:
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
            if lines is not None and len(lines) >= 5:
                angles = [abs(math.degrees(l[0][1]) - 90)
                          for l in lines[:50]
                          if abs(math.degrees(l[0][1]) - 90) < 45]
                if angles and abs(float(np.median(angles))) > 1.5:
                    converted_score += 0.8  # tilted page → physical scan
        except Exception:
            pass

    # ── Signal 9: File format hint ───────────────────────────────────────────
    fmt = path.suffix.lower()
    if fmt == ".png":
        native_score += 0.3
    elif fmt in (".jpg", ".jpeg"):
        converted_score += 0.3; pure_score += 0.2

    # ── Decision ─────────────────────────────────────────────────────────────
    scores = {
        "PURE_IMAGE":    pure_score,
        "NATIVE_TEXT":   native_score,
        "CONVERTED_OCR": converted_score,
    }
    return max(scores, key=scores.get)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python detect_image_type.py <image_path>")
        sys.exit(0)
    result = detect_image_type(sys.argv[1])
    print(result)
