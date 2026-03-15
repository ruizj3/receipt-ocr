import os
import re
from datetime import datetime
from typing import List, Optional, Dict, Tuple

from pydantic import BaseModel
from rapidfuzz import process, fuzz
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


def save_fallback_text(image_path: str, ocr_text: str) -> None:
    """
    Save the OCR text for fallback receipts to a .txt file.
    Output goes to: ./receipts_ocr_text/<receiptname>_fallback.txt
    """
    # Ensure output directory exists
    out_dir = "./receipts_ocr_text"
    os.makedirs(out_dir, exist_ok=True)

    # Extract just the base filename without directories
    base = os.path.basename(image_path)          # "Safeway20250726.jpg"
    stem = os.path.splitext(base)[0]             # "Safeway20250726"

    # Build output path
    txt_path = os.path.join(out_dir, f"{stem}_fallback.txt")

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
    try:
        from paddleocr import PaddleOCR
        
        # Initialize PaddleOCR (lang='en' for English)
        # use_angle_cls=True for rotation detection
        print("[OCR] Initializing PaddleOCR (downloading models on first run)...")
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        
        print("[OCR] Running PaddleOCR on image...")
        result = ocr.ocr(image_path, cls=True)
        
        # Extract text from result
        if result and result[0]:
            lines = [line[1][0] for line in result[0]]
            text = '\n'.join(lines)
            return text
        return ""
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
        print("[OCR] Loading EasyOCR (this may take a while on first run)...")
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        result = reader.readtext(image_path, detail=0, paragraph=True)
        return '\n'.join(result)
    except ImportError:
        print("[INFO] EasyOCR not available. Install with: pip install easyocr")
        return ""
    except Exception as e:
        print(f"[WARN] EasyOCR failed: {e}")
        return ""


def score_ocr_quality(text: str) -> int:
    """
    Score OCR text quality based on:
    - Number of readable words
    - Presence of common receipt keywords
    - Ratio of letters to garbage characters
    """
    if not text:
        return 0
    
    score = 0
    
    # Count words (sequences of 3+ letters)
    words = re.findall(r'[A-Za-z]{3,}', text)
    score += len(words) * 10
    
    # Bonus for common receipt words
    receipt_keywords = ['total', 'subtotal', 'tax', 'item', 'price', 'sale', 
                       'card', 'visa', 'payment', 'balance', 'store', 'receipt']
    for keyword in receipt_keywords:
        if keyword.lower() in text.lower():
            score += 50
    
    # Bonus for dollar amounts
    prices = re.findall(r'\$?\d+\.\d{2}', text)
    score += len(prices) * 20
    
    # Penalty for too many non-ascii or special chars
    ascii_ratio = sum(c.isalnum() or c.isspace() for c in text) / max(len(text), 1)
    score += int(ascii_ratio * 100)
    
    return score

def ocr_receipt_to_text(path: str) -> str:
    """
    Extract text from receipt using multiple strategies:
    1. Try all 4 rotations with Tesseract and pick the best based on text quality (fast, works well)
    2. Advanced preprocessing if needed
    3. Try PaddleOCR if available (slower first run, but more accurate)
    4. Fallback to EasyOCR if all else fails
    """
    print(f"[OCR] Processing {os.path.basename(path)}...")
    
    # Strategy 0: Try all 4 rotations with Tesseract and score them (fast and effective)
    print("[OCR] Trying Tesseract with rotation detection...")
    try:
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Scale up for better OCR (2x)
        height, width = gray.shape
        scaled = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        
        # Try all 4 rotations
        rotations = [
            (scaled, 0, "0°"),
            (cv2.rotate(scaled, cv2.ROTATE_90_CLOCKWISE), 90, "90°"),
            (cv2.rotate(scaled, cv2.ROTATE_180), 180, "180°"),
            (cv2.rotate(scaled, cv2.ROTATE_90_COUNTERCLOCKWISE), 270, "270°"),
        ]
        
        best_text = ""
        best_score = 0
        best_angle = "0°"
        
        for rotated, angle_deg, angle_str in rotations:
            # Try two PSM modes and pick best
            for psm in ['6', '4']:
                text = ocr_with_tesseract(rotated, f'--psm {psm} --oem 3')
                score = score_ocr_quality(text)
                
                if score > best_score:
                    best_score = score
                    best_text = text
                    best_angle = f"{angle_str} PSM{psm}"
        
        if best_text.strip() and best_score > 100:
            print(f"[OCR] Best result at {best_angle} (score: {best_score}): {len(best_text)} characters")
            save_fallback_text(path, best_text)
            return best_text
    except Exception as e:
        print(f"[WARN] Tesseract rotation scoring failed: {e}")
    
    # Strategy 1: Try PaddleOCR (if Tesseract failed or user wants better accuracy)
    print("[OCR] Trying PaddleOCR (may take a while on first run)...")
    paddle_text = try_paddleocr(path)
    if paddle_text.strip():
        score = score_ocr_quality(paddle_text)
        print(f"[OCR] PaddleOCR extracted {len(paddle_text)} characters (score: {score})")
        save_fallback_text(path, paddle_text)
        return paddle_text
    
    # Strategy 2: Advanced preprocessing with Tesseract
    print("[OCR] Trying advanced preprocessing with Tesseract...")
    try:
        preprocessed = preprocess_image_advanced(path)
        
        # Try different PSM modes
        configs = [
            '--psm 4 --oem 3',  # Single column
            '--psm 3 --oem 3',  # Fully automatic
        ]
        
        best_text = ""
        best_score = 0
        
        for config in configs:
            text = ocr_with_tesseract(preprocessed, config)
            score = score_ocr_quality(text)
            if score > best_score:
                best_score = score
                best_text = text
        
        if best_text.strip() and best_score > 100:
            print(f"[OCR] Advanced method extracted {len(best_text)} characters (score: {best_score})")
            save_fallback_text(path, best_text)
            return best_text
    
    except Exception as e:
        print(f"[WARN] Advanced preprocessing failed: {e}")
    
    # Strategy 3: Fallback to EasyOCR
    print("[OCR] Trying EasyOCR as last resort...")
    easy_text = try_easyocr(path)
    if easy_text.strip():
        score = score_ocr_quality(easy_text)
        print(f"[OCR] EasyOCR extracted {len(easy_text)} characters (score: {score})")
        save_fallback_text(path, easy_text)
        return easy_text
    
    # Last resort: return empty
    print("[ERROR] All OCR strategies failed")
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
        "debit", "credit", "balance", "cash", "amount due", "amt due", "payment",
        "card", "ref:", "auth:", "aid", "tvr", "****", "----", "store director",
        "main:", "phone", "cashier", "refrig", "frozen", "your cashier",
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

    # Match an 8-digit date at the end: YYYYMMDD
    m = re.search(r'(\d{8})$', name)
    if not m:
        return None, None

    date_str = m.group(1)      # e.g. "20250424"
    store = name[:m.start()]   # everything before the date

    try:
        dt = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        dt = None

    return store, dt


# ========== OCR-BASED EXTRACTION ==========

def extract_receipt_data_from_ocr(
    image_path: str,
) -> Tuple[Optional[Receipt], Optional[str]]:
    """
    Extract structured data from a receipt image using OCR only (no LLM).
    Returns (Receipt or None, error_reason or None).
    error_reason examples: "ocr-error", "parse-error".
    """
    try:
        # Get OCR text from the receipt
        ocr_text = ocr_receipt_to_text(image_path)
        
        # Parse items from OCR text using rule-based parsing
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
        return base

    best_name, best_score, _ = best
    return best_name


# ========== MAIN PIPELINE ==========
def build_price_history(receipt_dir: str):
    """
    Go through all receipt images, extract data via OCR, and build a price history per product.
    Returns:
      - price_history: { product_name -> list[(store_name, date, price)] }
      - successes: list of image paths processed successfully
      - failures:  list of (image_path, reason) for failures
    """
    price_history: Dict[str, List[Tuple[Optional[str], Optional[datetime], Optional[float]]]] = {}
    successes: List[str] = []
    failures: List[Tuple[str, str]] = []

    for image_path in iter_receipt_images(receipt_dir):
        print(f"🧾 Processing {image_path} ...")

        store_name, purchase_dt = extract_store_and_date_from_filename(image_path)

        receipt, error_reason = extract_receipt_data_from_ocr(image_path)

        if receipt is None:
            failures.append((image_path, error_reason or "unknown"))
            continue

        successes.append(image_path)

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
            price_history[canonical_name].append((store_name, purchase_dt, price))

        del receipt  # drop reference ASAP

    # Sort each product's entries by date (None at the end)
    for product, entries in price_history.items():
        price_history[product] = sorted(
            entries,
            key=lambda x: (x[1] is None, x[1] or datetime.max),  # x = (store, date, price)
        )

    return price_history, successes, failures


def print_price_history(
    price_history: Dict[str, List[Tuple[Optional[str], Optional[datetime], Optional[float]]]]
):
    print("\n===== PRICE HISTORY =====")
    for product, entries in sorted(price_history.items()):
        print(f"\n📦 {product}")
        for store_name, dt, price in entries:
            store_str = store_name or "unknown-store"
            date_str = dt.strftime("%Y-%m-%d") if dt else "unknown-date"
            if isinstance(price, (int, float)) and price is not None:
                price_str = f"{price:.2f}"
            else:
                price_str = "unknown"
            print(f"  {date_str} @ {store_str}: {price_str}")


def main():
    price_history, successes, failures = build_price_history(RECEIPT_DIR)
    print_price_history(price_history)

    # Write CSV with store, product, date, price
    out_csv = "receipt_price_history.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("store_name,product_name,date,price\n")
        for product, entries in price_history.items():
            for store_name, dt, price in entries:
                store_str = store_name or ""
                date_str = dt.strftime("%Y-%m-%d") if dt else ""
                price_str = str(price) if price is not None else ""
                f.write(f"\"{store_str}\",\"{product}\",{date_str},{price_str}\n")

    print(f"\n✅ Done. CSV written to {out_csv}")

    # summary
    print("\n===== RECEIPT PROCESSING SUMMARY =====")
    print(f"Total images processed: {len(successes) + len(failures)}")
    print(f"  ✅ Successes: {len(successes)}")
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
