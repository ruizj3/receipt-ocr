import argparse
import contextlib
import csv
import io
import os
import re
import sys
from importlib.util import find_spec
from inspect import signature
from datetime import datetime
from typing import List, Optional, Dict, Tuple

from pydantic import BaseModel
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np

from PIL import Image
import tempfile
import subprocess
import pytesseract

# Disable PaddleOCR model source connectivity check (speeds up initialization)
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

def preprocess_image_advanced(image_path: str) -> np.ndarray:
    """
    Advanced image preprocessing for better OCR results:
    - Load and convert to grayscale
    - Increase contrast with CLAHE
    - Denoise
    - Deskew/rotate correction
    - Adaptive thresholding
    - Morphological operations to clean up
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Sharpen the image
    kernel_sharpening = np.array([[-1,-1,-1], 
                                   [-1, 9,-1], 
                                   [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel_sharpening)
    
    # Adaptive thresholding (better than global threshold)
    binary = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Morphological operations to remove noise
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned


def rectify_receipt(image: np.ndarray) -> np.ndarray:
    """Perspective-correct a clearly detected receipt quadrilateral; otherwise return the input."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=2
    )

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    image_area = float(height * width)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.15 or area > image_area * 0.98:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            candidates.append((area, polygon.reshape(4, 2).astype(np.float32)))

    if not candidates:
        return image

    points = candidates[0][1]
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    ordered = np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ],
        dtype=np.float32,
    )
    top_left, top_right, bottom_right, bottom_left = ordered
    output_width = int(
        max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left))
    )
    output_height = int(
        max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left))
    )
    if output_width < 100 or output_height < 100:
        return image

    destination = np.array(
        [[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(image, transform, (output_width, output_height))


def detect_and_correct_rotation(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct image rotation using text orientation detection.
    """
    # Try to detect orientation using Tesseract's OSD (Orientation and Script Detection)
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image
        osd = pytesseract.image_to_osd(rgb)
        rotation = int(re.search(r'Rotate: (\d+)', osd).group(1))
        
        if rotation != 0:
            print(f"[INFO] Detected rotation: {rotation} degrees, correcting...")
            # Rotate image
            if rotation == 90:
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                return cv2.rotate(image, cv2.ROTATE_180)
            elif rotation == 270:
                return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception as e:
        print(f"[WARN] Could not detect rotation: {e}")
    
    return image


def save_fallback_text(
    image_path: str, ocr_text: str, suffix: str = "fallback"
) -> None:
    """
    Save the OCR text for fallback receipts to a .txt file.
    Output goes to: ./receipts_ocr_text/<receiptname>_<suffix>.txt
    """
    # Ensure output directory exists
    out_dir = "./receipts_ocr_text"
    os.makedirs(out_dir, exist_ok=True)

    # Extract just the base filename without directories
    base = os.path.basename(image_path)          # "Safeway20250726.jpg"
    stem = os.path.splitext(base)[0]             # "Safeway20250726"

    # Build output path
    txt_path = os.path.join(out_dir, f"{stem}_{suffix}.txt")

    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(ocr_text)
        print(f"[FALLBACK] Saved OCR text to {txt_path}")
    except Exception as e:
        print(f"[WARN] Failed to save OCR fallback text: {e}")


def ocr_with_tesseract(image: np.ndarray, config: str = '') -> str:
    """
    Extract text using Tesseract OCR with custom configuration.
    """
    # Default config optimized for receipts
    if not config:
        # PSM 6 = Assume a single uniform block of text
        # PSM 4 = Assume a single column of text of variable sizes
        config = '--psm 6 --oem 3'
    
    text = pytesseract.image_to_string(image, config=config)
    return text


def try_paddleocr(image_path: str) -> str:
    """
    Use PaddleOCR - often more accurate than Tesseract for receipts.
    Fast and robust for various image qualities.
    """
    if find_spec("paddle") is None:
        print("[INFO] PaddleOCR runtime is not installed (missing 'paddle'); skipping.")
        return ""

    try:
        from paddleocr import PaddleOCR

        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARN] PaddleOCR could not load image: {image_path}")
            return ""
        height, width = image.shape[:2]
        max_side = 2400
        scale = min(1.0, max_side / max(height, width))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
            print(f"[OCR] Resized PaddleOCR input to {image.shape[1]}x{image.shape[0]}")

        parameters = signature(PaddleOCR).parameters
        orientation_option = (
            "use_textline_orientation"
            if "use_textline_orientation" in parameters
            else "use_angle_cls"
        )
        print("[OCR] Initializing PaddleOCR (downloading models on first run)...")
        ocr = PaddleOCR(**{orientation_option: True, "lang": "en"})

        print("[OCR] Running PaddleOCR on image...")
        if hasattr(ocr, "predict"):
            predict_parameters = signature(ocr.predict).parameters
            input_parameter = "input" if "input" in predict_parameters else "img"
            result = ocr.predict(**{input_parameter: image})
        else:
            ocr_parameters = signature(ocr.ocr).parameters
            result = ocr.ocr(image, cls=True) if "cls" in ocr_parameters else ocr.ocr(image)

        lines = []
        for page in result or []:
            if isinstance(page, dict):
                page_data = page.get("res", page)
            else:
                page_data = getattr(page, "json", page)
                if callable(page_data):
                    page_data = page_data()
                if isinstance(page_data, dict):
                    page_data = page_data.get("res", page_data)

            if isinstance(page_data, dict) and "rec_texts" in page_data:
                lines.extend(str(line) for line in page_data["rec_texts"] if line)
            elif isinstance(page_data, (list, tuple)):
                lines.extend(
                    str(line[1][0])
                    for line in page_data
                    if isinstance(line, (list, tuple))
                    and len(line) > 1
                    and isinstance(line[1], (list, tuple))
                    and line[1]
                )
        return '\n'.join(lines)
    except ImportError as e:
        print(f"[INFO] PaddleOCR not available: {e}")
        return ""
    except Exception as e:
        import traceback
        print(f"[WARN] PaddleOCR failed: {e}")
        print(f"[DEBUG] {traceback.format_exc()}")
        return ""

def try_easyocr(image_path: str) -> str:
    """
    Fallback OCR using EasyOCR (more robust for difficult images).
    Only use if other methods completely fail.
    """
    try:
        import easyocr

        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARN] EasyOCR could not load image: {image_path}")
            return ""

        print("[OCR] Loading EasyOCR (this may take a while on first run)...")
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        horizontal_batches, _ = reader.detect(image)
        height, width = image.shape[:2]
        boxes = horizontal_batches[0] if horizontal_batches else []
        valid_boxes = _clip_easyocr_horizontal_boxes(boxes, width, height)
        if not valid_boxes:
            valid_boxes = [[0, width, 0, height]]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = reader.recognize(
            gray, valid_boxes, [], detail=0, paragraph=True, reformat=False
        )
        return '\n'.join(result)
    except ImportError:
        print("[INFO] EasyOCR not available. Install with: pip install easyocr")
        return ""
    except Exception as e:
        print(f"[WARN] EasyOCR failed: {e}")
        return ""


def _clip_easyocr_horizontal_boxes(
    boxes: List[List[int]], width: int, height: int
) -> List[List[int]]:
    valid_boxes = []
    for box in boxes:
        if len(box) != 4:
            continue
        x_min = max(0, min(width, int(box[0])))
        x_max = max(0, min(width, int(box[1])))
        y_min = max(0, min(height, int(box[2])))
        y_max = max(0, min(height, int(box[3])))
        if x_max > x_min and y_max > y_min:
            valid_boxes.append([x_min, x_max, y_min, y_max])
    return valid_boxes


def score_ocr_quality(text: str) -> int:
    """Return a receipt-aware text quality score from 0 to 100."""
    if not text:
        return 0

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    words = re.findall(r"[A-Za-z]{2,}", text)
    prices = re.findall(r"\$?\d{1,5}[.,]\d{2}\b", text)
    item_lines = sum(
        bool(re.search(r"[A-Za-z]{2,}", line) and re.search(r"\d+[.,]\d{2}", line))
        for line in lines
    )
    readable_ratio = sum(char.isalnum() or char.isspace() for char in text) / max(len(text), 1)

    score = (
        min(20, len(words) * 1.5)
        + min(20, len(prices) * 4)
        + min(30, item_lines * 8)
        + min(15, readable_ratio * 15)
        + min(15, len(lines) * 1.5)
    )
    return int(round(score))


def _group_tesseract_words(data: Dict[str, List]) -> List[List[Dict[str, object]]]:
    grouped: Dict[Tuple[int, int, int], List[Dict[str, object]]] = {}
    for index, raw_text in enumerate(data.get("text", [])):
        word = str(raw_text).strip()
        if not word:
            continue
        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        grouped.setdefault(key, []).append(
            {
                "text": word,
                "left": int(data["left"][index]),
                "top": int(data["top"][index]),
                "width": int(data["width"][index]),
                "height": int(data["height"][index]),
                "confidence": float(data["conf"][index]),
            }
        )
    return [sorted(words, key=lambda word: int(word["left"])) for words in grouped.values()]


def _text_and_confidence(data: Dict[str, List]) -> Tuple[str, float, List[List[Dict[str, object]]]]:
    lines = _group_tesseract_words(data)
    text = "\n".join(" ".join(str(word["text"]) for word in line) for line in lines)
    confidences = [
        float(word["confidence"])
        for line in lines
        for word in line
        if float(word["confidence"]) >= 0
    ]
    confidence = float(np.mean(confidences)) if confidences else 0.0
    return text, confidence, lines


def _score_tesseract_result(
    text: str,
    confidence: float,
    lines: List[List[Dict[str, object]]],
    image_width: int,
    parsed_item_count: int = 0,
) -> Tuple[float, float]:
    price_centers = []
    for line in lines:
        prices = [
            float(word["left"]) + float(word["width"]) / 2
            for word in line
            if re.fullmatch(r"\$?\d{1,5}[.,]\d{2}", str(word["text"]).strip())
        ]
        if prices:
            price_centers.append(max(prices) / max(image_width, 1))

    if len(price_centers) >= 2:
        spread = float(np.std(price_centers))
        alignment = max(0.0, min(100.0, 100.0 - spread * 500))
    elif price_centers:
        alignment = 35.0
    else:
        alignment = 0.0

    quality = score_ocr_quality(text)
    item_lines = sum(
        bool(re.search(r"[A-Za-z]{2,}", line)
             and re.search(r"\d{1,5}[.,]\d{2}", line))
        for line in text.splitlines()
    )
    parse_coverage = min(1.0, parsed_item_count / item_lines) if item_lines else 0.0
    parse_score = parse_coverage * 100
    score = (
        confidence * 0.50
        + quality * 0.20
        + alignment * 0.15
        + parse_score * 0.15
    )
    return score, alignment


def _scaled_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape[:2]
    longest_side = max(height, width)
    if longest_side < 1800:
        scale = 1800 / longest_side
    else:
        scale = min(1.0, 3600 / longest_side)
    if abs(scale - 1.0) > 0.01:
        gray = cv2.resize(
            gray,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
        )
    return gray


def _ocr_image_candidates(image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    rectified = rectify_receipt(image)
    candidates: List[Tuple[str, np.ndarray]] = []
    for name, source in (("original", image), ("rectified", rectified)):
        gray = _scaled_gray(source)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        variants = [("gray", gray), ("contrast", clahe)]
        if name == "rectified":
            threshold = cv2.adaptiveThreshold(
                clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
            )
            variants.append(("adaptive", threshold))

        angles = (0, 90, 180, 270) if name == "original" else (0,)
        for variant_name, variant in variants:
            for angle in angles:
                rotated = variant if angle == 0 else cv2.rotate(
                    variant,
                    {
                        90: cv2.ROTATE_90_CLOCKWISE,
                        180: cv2.ROTATE_180,
                        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                    }[angle],
                )
                candidates.append((f"{name}/{variant_name}/{angle}", rotated))
    return candidates


def _refine_price_column(
    image: np.ndarray, lines: List[List[Dict[str, object]]]
) -> Optional[str]:
    height, width = image.shape[:2]
    price_positions: Dict[int, List[float]] = {}
    for line_index, line in enumerate(lines):
        for word in line:
            if re.fullmatch(r"\$?\d{1,5}[.,]\d{2}", str(word["text"]).strip()):
                center = float(word["left"]) + float(word["width"]) / 2
                if center > width * 0.55:
                    price_positions.setdefault(line_index, []).append(center)

    centers = [max(positions) for positions in price_positions.values()]
    if len(centers) < 2:
        return None

    bins: Dict[int, List[float]] = {}
    for center in centers:
        bins.setdefault(int(center / max(width * 0.06, 1)), []).append(center)
    repeated_bins = [values for values in bins.values() if len(values) >= 2]
    if not repeated_bins:
        return None
    column_center = float(np.median(max(repeated_bins, key=lambda values: (len(values), np.median(values)))))
    column_left = max(0, int(column_center - width * 0.10))
    refined_lines = []
    price_pattern = re.compile(r"\$?\d{1,5}[.,]\d{2}")

    for line in lines:
        line_top = min(int(word["top"]) for word in line)
        line_bottom = max(int(word["top"]) + int(word["height"]) for word in line)
        left_words = [
            str(word["text"])
            for word in line
            if float(word["left"]) + float(word["width"]) / 2 < column_left
        ]
        if not left_words or not re.search(r"[A-Za-z]{2,}", " ".join(left_words)):
            refined_lines.append(" ".join(str(word["text"]) for word in line))
            continue

        crop_top = max(0, line_top - max(3, (line_bottom - line_top) // 3))
        crop_bottom = min(height, line_bottom + max(3, (line_bottom - line_top) // 3))
        price_crop = image[crop_top:crop_bottom, column_left:]
        if price_crop.size == 0:
            refined_lines.append(" ".join(str(word["text"]) for word in line))
            continue
        price_crop = cv2.resize(price_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        price_text = pytesseract.image_to_string(
            price_crop,
            config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.$,",
        )
        matches = price_pattern.findall(price_text.replace(" ", ""))
        if matches:
            price = matches[-1].replace(",", ".").lstrip("$")
            refined_lines.append(" ".join(left_words + [price]))
        else:
            refined_lines.append(" ".join(str(word["text"]) for word in line))

    return "\n".join(refined_lines)


def _token_agreement(first: str, second: str) -> float:
    first_tokens = set(re.findall(r"[a-z0-9]+", first.lower()))
    second_tokens = set(re.findall(r"[a-z0-9]+", second.lower()))
    if not first_tokens or not second_tokens:
        return 0.0
    return len(first_tokens & second_tokens) / len(first_tokens | second_tokens)


OLLAMA_RECEIPT_PROMPT = (
    "Transcribe every visible line on this receipt verbatim. Keep one printed "
    "line per output line. Preserve item descriptions, digits, decimal points, "
    "symbols, and all printed price columns in their original left-to-right "
    "order, including repeated prices. Do not calculate, normalize, summarize, "
    "or omit text, and do not add labels or commentary. Return only the "
    "transcription."
)


def ocr_with_ollama(path: str, model: str) -> str:
    """Transcribe a receipt image with an Ollama vision model."""
    try:
        import ollama
    except ImportError as error:
        raise RuntimeError(
            "Ollama mode requires the Python package. Install it with: pip install ollama"
        ) from error

    print(f"[OCR] Sending image to Ollama model {model}...")
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": OLLAMA_RECEIPT_PROMPT,
                "images": [path],
            }
        ],
    )
    text = response["message"]["content"].strip()
    if not text:
        raise RuntimeError(f"Ollama model {model} returned no text")
    return text


def ocr_receipt_to_text(
    path: str,
    save_output: bool = True,
    metadata: Optional[Dict[str, object]] = None,
    engine: str = "tesseract",
    ollama_model: str = "qwen2.5vl:7b",
) -> str:
    """Recognize receipt text, rank image variants, and retry other engines only at low confidence."""
    print(f"[OCR] Processing {os.path.basename(path)}...")
    if metadata is not None:
        metadata.clear()
    image = cv2.imread(path)
    if image is None:
        print(f"[ERROR] Could not load image: {path}")
        if save_output:
            save_fallback_text(path, "[OCR FAILED]")
        return ""

    if engine == "ollama":
        text = ocr_with_ollama(path, ollama_model)
        if metadata is not None:
            metadata.update(
                {
                    "selected_engine": f"Ollama/{ollama_model}",
                    "tesseract_confidence": None,
                    "tesseract_score": None,
                    "low_confidence": False,
                }
            )
        if save_output:
            model_suffix = _safe_model_suffix(ollama_model)
            save_fallback_text(path, text, f"ollama_{model_suffix}")
        return text
    if engine != "tesseract":
        raise ValueError(f"Unsupported OCR engine: {engine}")

    best = None
    candidate_results = []
    try:
        candidates = _ocr_image_candidates(image)
        for name, candidate_image in candidates:
            data = pytesseract.image_to_data(
                candidate_image,
                config="--psm 6 --oem 3",
                output_type=pytesseract.Output.DICT,
            )
            text, confidence, lines = _text_and_confidence(data)
            parsed_items = _parse_items_from_tesseract_lines(lines)
            score, alignment = _score_tesseract_result(
                text, confidence, lines, candidate_image.shape[1], len(parsed_items)
            )
            result = {
                "name": name,
                "image": candidate_image,
                "data": data,
                "text": text,
                "confidence": confidence,
                "lines": lines,
                "alignment": alignment,
                "score": score,
                "parsed_items": parsed_items,
            }
            candidate_results.append(result)
            if best is None or score > best["score"]:
                best = result

        if best is not None:
            shortlist = sorted(
                candidate_results, key=lambda result: result["score"], reverse=True
            )[:3]
            for candidate_result in shortlist:
                name = candidate_result["name"]
                candidate_image = candidate_result["image"]
                data = pytesseract.image_to_data(
                    candidate_image,
                    config="--psm 4 --oem 3",
                    output_type=pytesseract.Output.DICT,
                )
                text, confidence, lines = _text_and_confidence(data)
                parsed_items = _parse_items_from_tesseract_lines(lines)
                score, alignment = _score_tesseract_result(
                    text, confidence, lines, candidate_image.shape[1], len(parsed_items)
                )
                if score > best["score"]:
                    best = {
                        "name": f"{name}/psm4",
                        "image": candidate_image,
                        "data": data,
                        "text": text,
                        "confidence": confidence,
                        "lines": lines,
                        "alignment": alignment,
                        "score": score,
                        "parsed_items": parsed_items,
                    }
    except Exception as error:
        print(f"[WARN] Tesseract candidate evaluation failed: {error}")

    best_text = best["text"] if best else ""
    best_confidence = best["confidence"] if best else 0.0
    best_score = best["score"] if best else 0.0
    low_confidence = best_confidence < 58 or best_score < 55
    selected_engine = "Tesseract" if best_text.strip() else "none"
    if metadata is not None:
        metadata.update(
            {
                "selected_engine": selected_engine,
                "tesseract_confidence": best_confidence,
                "tesseract_score": best_score,
                "low_confidence": low_confidence,
            }
        )
    if best and best_text:
        refined_text = _refine_price_column(best["image"], best["lines"])
        if refined_text and score_ocr_quality(refined_text) >= score_ocr_quality(best_text):
            best_text = refined_text
        best["parsed_items"] = _parse_items_from_tesseract_lines(
            best["lines"], best_text
        )
        print(
            f"[OCR] Tesseract selected {best['name']} "
            f"(confidence: {best_confidence:.1f}, score: {best_score:.1f})"
        )

    if low_confidence:
        print("[OCR] Low Tesseract confidence; checking alternate OCR engines...")
        tesseract_rank = score_ocr_quality(best_text) * 0.65 + best_confidence * 0.25
        for engine_name, engine in (("PaddleOCR", try_paddleocr), ("EasyOCR", try_easyocr)):
            alternative = engine(path)
            if not alternative.strip():
                continue
            agreement = _token_agreement(best_text, alternative)
            alternative_rank = score_ocr_quality(alternative) * 0.65 + agreement * 10
            print(
                f"[OCR] {engine_name} score: {alternative_rank:.1f} "
                f"(agreement: {agreement:.2f})"
            )
            if alternative_rank > tesseract_rank + 5:
                best_text = alternative
                tesseract_rank = alternative_rank
                selected_engine = engine_name

    if metadata is not None:
        metadata["selected_engine"] = selected_engine
        metadata["parsed_items"] = (
            best["parsed_items"]
            if best and selected_engine == "Tesseract"
            else []
        )

    if best_text.strip():
        if save_output:
            save_fallback_text(path, best_text)
        return best_text

    print("[ERROR] All OCR strategies failed")
    if save_output:
        save_fallback_text(path, "[OCR FAILED]")
    return ""

# ========== CONFIG ==========

RECEIPT_DIR = "./receipts"  # folder containing your receipt images


# ========== SCHEMAS FOR STRUCTURED OUTPUT ==========

class ReceiptItem(BaseModel):
    product_name: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total_price: Optional[float] = None
    currency: Optional[str] = None


class Receipt(BaseModel):
    # We ignore these; LLM only returns items, they just get defaulted to None.
    store_name: Optional[str] = None
    purchase_date: Optional[str] = None
    items: List[ReceiptItem]


# ========== FALLBACK PARSING FROM OCR TEXT ==========

def parse_items_from_ocr_text(ocr_text: str) -> List[ReceiptItem]:
    """
    Enhanced rule-based parser for OCR text from receipts.
    
    Handles multiple formats:
    - '2 BANANAS            0.78'
    - 'BANANAS              0.39'
    - 'SIG BISCUIT SNDMCH    8.49  8.49'  (price appears twice)
    - 'LUC GREEK YOGURT      4.29  4.29 S'
    - Product codes followed by descriptions
    
    Heuristics:
    - Skip lines that look like totals/taxes/payments/headers.
    - Look for prices in format X.XX
    - Extract product names and prices
    """
    items: List[ReceiptItem] = []
    
    # Keywords to skip
    skip_keywords = [
        "subtotal", "total", "tax", "change", "tender", "visa", "mastercard",
        "debit", "credit", "balance", "cash", "amount", "amt due", "payment",
        "card", "ref:", "auth:", "aid", "tvr", "****", "----", "store director",
        "main:", "phone", "cashier", "refrig", "frozen", "your cashier",
        "savings", "discount", "coupon", "you saved",
    ]

    for raw_line in ocr_text.splitlines():
        line = raw_line.strip()
        if not line or len(line) < 3:
            continue

        lower = line.lower()
        
        # Skip header/footer/payment lines
        if any(kw in lower for kw in skip_keywords):
            continue
        
        # Skip lines that are mostly numbers (like barcodes, card numbers)
        if re.match(r'^[0-9\s]+$', line):
            continue
        
        # Skip lines with date/time patterns
        if re.search(r'\d{2}/\d{2}/\d{2,4}', line):
            continue
        
        # Find all prices in the line (format: optional $ then digits.XX)
        prices = re.findall(r'\$?(\d+\.\d{2})\b', line)
        
        if not prices:
            continue
        
        # Convert to floats
        price_values = [float(p) for p in prices]
        
        # Take the last price as the item price (often repeated or has total at end)
        price = price_values[-1]
        
        # Extract product description
        # First, remove product codes (long strings of digits like 2113009836)
        desc = re.sub(r'\b\d{8,}\b', '', line)  # Remove 8+ digit codes
        
        # Remove all prices from description (but keep the text before them)
        for p in prices:
            desc = desc.replace(p, '')
        
        # Remove dollar signs
        desc = desc.replace('$', '')
        
        # Remove special chars but preserve letters and spaces
        desc = re.sub(r'[_*+§&#@%]+', ' ', desc)
        
        # Remove leading short junk text (like 'yg,', 'Me', 'a,', 'f')
        # But keep meaningful text like 'SIG'
        desc = re.sub(r'^[^A-Z]*', '', desc)  # Remove everything before first capital
        
        # Remove trailing junk (numbers, punctuation, single letters like 'ff', '&f', 'El')
        desc = re.sub(r'[\s,\.0-9]+$', '', desc)
        desc = re.sub(r'\s+[a-z]{1,2}$', '', desc, flags=re.IGNORECASE)  # Remove trailing 1-2 letter codes
        
        # Clean up whitespace
        desc = re.sub(r'\s{2,}', ' ', desc).strip()
        
        if not desc or len(desc) < 3:
            continue
        
        # Try to extract quantity if present at the start (1-99 only)
        qty_match = re.match(r'^(\d{1,2})\s+(.+)', desc)
        if qty_match:
            qty_val = float(qty_match.group(1))
            if qty_val < 100:
                quantity = qty_val
                desc = qty_match.group(2)
            else:
                quantity = None
        else:
            quantity = None

        items.append(
            ReceiptItem(
                product_name=desc,
                quantity=quantity,
                unit_price=None,
                total_price=price,
                currency=None,
            )
        )

    return items


def _parse_items_from_tesseract_lines(
    lines: List[List[Dict[str, object]]], refined_text: Optional[str] = None
) -> List[ReceiptItem]:
    """Parse positioned Tesseract words, using the rightmost price in each row."""
    items: List[ReceiptItem] = []
    refined_lines = refined_text.splitlines() if refined_text else []
    price_pattern = re.compile(r"\$?\d{1,5}[.,]\d{2}")

    for line_index, words in enumerate(lines):
        price_words = [
            word for word in words
            if price_pattern.fullmatch(str(word["text"]).strip())
        ]
        if not price_words:
            continue

        rightmost_price = max(
            price_words,
            key=lambda word: int(word["left"]) + int(word["width"]),
        )
        price_left = int(rightmost_price["left"])
        description_words = [
            str(word["text"])
            for word in words
            if int(word["left"]) + int(word["width"]) / 2 < price_left
            and not price_pattern.fullmatch(str(word["text"]).strip())
        ]
        description = " ".join(description_words).strip()
        if not re.search(r"[A-Za-z]{2,}", description):
            continue

        price_text = str(rightmost_price["text"]).strip()
        if line_index < len(refined_lines):
            refined_prices = price_pattern.findall(refined_lines[line_index])
            if refined_prices:
                price_text = refined_prices[-1]
        price_text = price_text.replace(",", ".").lstrip("$")
        row_items = parse_items_from_ocr_text(f"{description} {price_text}")
        items.extend(row_items)

    return items

def fallback_receipt_from_ocr_text(image_path: str, ocr_text: str) -> Optional[Receipt]:
    """
    Build a Receipt from OCR text using simple regex parsing.
    Returns None if no items could be found.
    """
    items = parse_items_from_ocr_text(ocr_text)
    if not items:
        return None
    return Receipt(items=items)


# ========== FILENAME PARSING ==========

def extract_store_and_date_from_filename(path: str) -> Tuple[Optional[str], Optional[datetime]]:
    """
    Given a filename like:  RedApple20250424.jpg
    Extract:
      - store = 'RedApple'
      - date = datetime(2025, 4, 24)
    """
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)

    # Allow common duplicate-name suffixes after the YYYYMMDD date.
    m = re.fullmatch(r"(?P<store>.+?)(?P<date>\d{8})(?:(?:\(\d+\))|(?:[_-]\d+))*", name)
    if not m:
        return None, None

    date_str = m.group("date")
    store = m.group("store")

    try:
        dt = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        dt = None

    return store, dt


# ========== OCR-BASED EXTRACTION ==========

def extract_receipt_data_from_ocr(
    image_path: str,
    ocr_metadata: Optional[Dict[str, object]] = None,
    ocr_engine: str = "tesseract",
    ollama_model: str = "qwen2.5vl:7b",
) -> Tuple[Optional[Receipt], Optional[str]]:
    """
    Extract structured data from a receipt image using OCR only (no LLM).
    Returns (Receipt or None, error_reason or None).
    error_reason examples: "ocr-error", "parse-error".
    """
    try:
        # Get OCR text from the receipt
        ocr_text = ocr_receipt_to_text(
            image_path,
            metadata=ocr_metadata,
            engine=ocr_engine,
            ollama_model=ollama_model,
        )
        
        # Tesseract parsing keeps word positions; other engines use transcript text.
        parsed_items = (
            ocr_metadata.get("parsed_items")
            if ocr_metadata and ocr_metadata.get("selected_engine") == "Tesseract"
            else None
        )
        if isinstance(parsed_items, list) and parsed_items:
            receipt = Receipt(items=parsed_items)
        else:
            receipt = fallback_receipt_from_ocr_text(image_path, ocr_text)
        
        if receipt is None or not receipt.items:
            print(f"[WARN] No items could be parsed from {image_path}")
            return None, "parse-error"
        
        return receipt, None
        
    except Exception as e:
        print(f"[ERROR] OCR processing failed for {image_path}: {e}")
        return None, "ocr-error"


# ========== HELPERS ==========

def iter_receipt_images(root_dir: str):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    for base, _, files in os.walk(root_dir):
        for f in files:
            if os.path.splitext(f.lower())[1] in exts:
                yield os.path.join(base, f)


def normalize_raw_name(raw_name: str) -> str:
    name = raw_name.strip().lower()
    name = " ".join(name.split())
    #for ch in [",", ";", ":", ".", "#", "*", "!", "?", "_", "-", "(", ")", "[", "]", "{", "}", "/","\\"]:
    for ch in [",", ";", ":", ".", "#", "*",]:
        name = name.replace(ch, "")
    return name


def fuzzy_canonical_name(raw_name: str, existing: List[str], score_cutoff=88) -> str:
    base = normalize_raw_name(raw_name)

    if not existing:
        return base

    best = process.extractOne(
        base,
        existing,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=score_cutoff,
    )

    if best is None:
        if len(base) < 5:
            return base

        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True
        )
        normalized_existing = [normalize_raw_name(name) for name in existing]
        matrix = vectorizer.fit_transform([base, *normalized_existing])
        similarities = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        ranked_indices = similarities.argsort()[::-1]
        best_index = int(ranked_indices[0])
        second_best = similarities[ranked_indices[1]] if len(ranked_indices) > 1 else 0.0
        if similarities[best_index] >= 0.72 and similarities[best_index] - second_best >= 0.08:
            return existing[best_index]
        return base

    best_name, best_score, _ = best
    return best_name


# ========== MAIN PIPELINE ==========
PRICE_REVIEW_THRESHOLD = 100.0


def _price_review_flags(
    price: Optional[float], ocr_metadata: Dict[str, object]
) -> List[str]:
    flags = []
    if ocr_metadata.get("low_confidence"):
        flags.append("low_ocr_confidence")
    if price is not None and price > PRICE_REVIEW_THRESHOLD:
        flags.append("price_over_100")
    if price is not None and price <= 0:
        flags.append("nonpositive_price")
    return flags


class _TeeOutput:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def build_price_history(
    receipt_dir: str,
    processing_summaries: Optional[List[Tuple[str, str]]] = None,
    ocr_metadata_by_file: Optional[Dict[str, Dict[str, object]]] = None,
    ocr_engine: str = "tesseract",
    ollama_model: str = "qwen2.5vl:7b",
):
    """
    Go through all receipt images, extract data via OCR, and build a price history per product.
    Returns:
        - price_history: { product_name -> list[(store_name, date, price, filename)] }
                - successes: list of image paths processed successfully
                - failures:  list of (image_path, reason) for failures
    """
    price_history: Dict[
        str, List[Tuple[Optional[str], Optional[datetime], Optional[float], str]]
    ] = {}
    successes: List[str] = []
    failures: List[Tuple[str, str]] = []
    if processing_summaries is None:
        processing_summaries = []

    for image_path in iter_receipt_images(receipt_dir):
        filename = os.path.basename(image_path)
        output = io.StringIO()
        ocr_metadata: Dict[str, object] = {}
        with contextlib.redirect_stdout(_TeeOutput(sys.stdout, output)):
            print(f"🧾 Processing {image_path} ...")
            receipt, error_reason = extract_receipt_data_from_ocr(
                image_path, ocr_metadata, ocr_engine, ollama_model
            )

        if ocr_metadata_by_file is not None:
            ocr_metadata_by_file[filename] = ocr_metadata

        store_name, purchase_dt = extract_store_and_date_from_filename(image_path)
        summary = output.getvalue().strip()

        if receipt is None:
            reason = error_reason or "unknown"
            failures.append((image_path, reason))
            processing_summaries.append(
                (filename, f"{summary}\nResult: failed ({reason})")
            )
            continue

        successes.append(image_path)
        receipt_review_flags = set()

        for item in receipt.items:
            if not item.product_name:
                continue

            existing_products = list(price_history.keys())
            canonical_name = fuzzy_canonical_name(
                item.product_name, existing_products, score_cutoff=88
            )

            if canonical_name not in price_history:
                price_history[canonical_name] = []

            price = item.unit_price if item.unit_price is not None else item.total_price
            receipt_review_flags.update(_price_review_flags(price, ocr_metadata))
            price_history[canonical_name].append(
                (store_name, purchase_dt, price, filename)
            )

        if receipt_review_flags:
            result = "Result: parsed, review required (" + ", ".join(
                sorted(receipt_review_flags)
            ) + f"; {len(receipt.items)} items)"
        else:
            result = (
                f"Result: parsed, not accuracy-verified ({len(receipt.items)} items; "
                "no automatic review flags)"
            )
        processing_summaries.append(
            (filename, f"{summary}\n{result}")
        )
        del receipt  # drop reference ASAP

    # Sort each product's entries by date (None at the end)
    for product, entries in price_history.items():
        price_history[product] = sorted(
            entries,
            key=lambda x: (x[1] is None, x[1] or datetime.max),
        )

    return price_history, successes, failures


def _write_price_history_csv(
    path: str,
    price_history: Dict[
        str, List[Tuple[Optional[str], Optional[datetime], Optional[float], str]]
    ],
    ocr_metadata_by_file: Dict[str, Dict[str, object]],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "filename",
                "store_name",
                "product_name",
                "date",
                "price",
                "selected_ocr_engine",
                "tesseract_confidence",
                "review_flags",
            ]
        )
        for product, entries in price_history.items():
            for store_name, dt, price, filename in entries:
                metadata = ocr_metadata_by_file.get(filename, {})
                writer.writerow(
                    [
                        filename,
                        store_name or "",
                        product,
                        dt.strftime("%Y-%m-%d") if dt else "",
                        price if price is not None else "",
                        metadata.get("selected_engine", ""),
                        metadata.get("tesseract_confidence", ""),
                        ";".join(_price_review_flags(price, metadata)),
                    ]
                )


def _safe_model_suffix(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", model).strip("_") or "model"


def _output_csv_paths(ocr_engine: str, ollama_model: str) -> Tuple[str, str]:
    suffix = f"_ollama_{_safe_model_suffix(ollama_model)}" if ocr_engine == "ollama" else ""
    return (
        f"receipt_price_history{suffix}.csv",
        f"receipt_processing_summary{suffix}.csv",
    )


def print_price_history(
    price_history: Dict[
        str, List[Tuple[Optional[str], Optional[datetime], Optional[float], str]]
    ]
):
    print("\n===== PRICE HISTORY =====")
    for product, entries in sorted(price_history.items()):
        print(f"\n📦 {product}")
        for store_name, dt, price, _ in entries:
            store_str = store_name or "unknown-store"
            date_str = dt.strftime("%Y-%m-%d") if dt else "unknown-date"
            if isinstance(price, (int, float)) and price is not None:
                price_str = f"{price:.2f}"
            else:
                price_str = "unknown"
            print(f"  {date_str} @ {store_str}: {price_str}")


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Extract receipt text and build price history.")
    parser.add_argument(
        "--ocr-engine",
        choices=("tesseract", "ollama"),
        default="tesseract",
        help="OCR engine to use (default: tesseract).",
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen2.5vl:7b",
        help="Ollama vision model to use when --ocr-engine=ollama.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    processing_summaries: List[Tuple[str, str]] = []
    ocr_metadata_by_file: Dict[str, Dict[str, object]] = {}
    price_history, successes, failures = build_price_history(
        RECEIPT_DIR,
        processing_summaries,
        ocr_metadata_by_file,
        args.ocr_engine,
        args.ollama_model,
    )
    print_price_history(price_history)

    # Keep model output separate so OCR engines can be compared side by side.
    out_csv, summary_csv = _output_csv_paths(args.ocr_engine, args.ollama_model)
    _write_price_history_csv(out_csv, price_history, ocr_metadata_by_file)

    print(f"\n✅ Done. CSV written to {out_csv}")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "summary"])
        writer.writerows(processing_summaries)
    print(f"[INFO] Receipt processing summaries written to {summary_csv}")

    # summary
    print("\n===== RECEIPT PROCESSING SUMMARY =====")
    print(f"Total images processed: {len(successes) + len(failures)}")
    print(f"  ✅ Parsed (not accuracy-verified): {len(successes)}")
    print(f"  ❌ Failures:  {len(failures)}")

    if successes:
        print("\nSuccessful receipts:")
        for path in successes:
            print(f"  - {path}")

    if failures:
        print("\nFailed receipts:")
        for path, reason in failures:
            print(f"  - {path} (reason: {reason})")


if __name__ == "__main__":
    main()
