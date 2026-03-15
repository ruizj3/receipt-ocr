# Receipt OCR Improvements

## Overview
This version uses enhanced OCR techniques without LLM dependencies, focusing on better image preprocessing and text extraction accuracy.

## Key Improvements

### 1. **Multi-Strategy OCR Approach**
The system now tries multiple OCR strategies in sequence:

- **Strategy 1**: Advanced preprocessing + rotation detection
  - Denoising using Non-Local Means algorithm
  - Adaptive thresholding (better than global threshold)
  - Morphological operations to clean noise
  - Automatic rotation detection and correction
  - Multiple Tesseract PSM modes (6, 4, 3)

- **Strategy 2**: Brute-force rotation testing
  - Tests all 4 possible rotations (0°, 90°, 180°, 270°)
  - Selects the rotation with the most extracted text

- **Strategy 3**: EasyOCR fallback
  - Uses deep learning-based OCR when Tesseract fails
  - More robust for difficult/degraded images

### 2. **Enhanced Parsing**
Improved regex patterns to handle multiple receipt formats:
- Standard format: `PRODUCT_NAME    PRICE`
- With quantity: `2 PRODUCT_NAME    PRICE`
- With duplicated prices: `PRODUCT_NAME    8.49  8.49`
- With product codes: `2113009836  PRODUCT_NAME  8.49`
- With sale indicators: `PRODUCT_NAME  4.29 S`

### 3. **Better Skip Logic**
Enhanced detection of non-item lines:
- Header/footer text
- Payment information
- Date/time stamps
- Barcodes and card numbers
- Tax and total lines

## Installation

### Required Dependencies
```bash
pip install -r requirements.txt
```

### System Requirements
You'll also need Tesseract OCR installed on your system:

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

## Usage

### Test Single Receipt
```bash
python test_ocr.py path/to/receipt.jpg
```

### Process All Receipts
```bash
python receipt_scanner.py
```

## How It Works

1. **Image Loading**: Loads receipt image
2. **Preprocessing**: 
   - Denoises image
   - Applies adaptive thresholding
   - Removes artifacts
3. **Rotation Detection**: Automatically detects and corrects orientation
4. **OCR Extraction**: Runs Tesseract with optimized settings
5. **Parsing**: Extracts items using enhanced regex patterns
6. **Output**: Saves to CSV with product, store, date, and price

## Troubleshooting

### Poor OCR Results
If OCR quality is poor:
1. Check image quality (should be at least 300 DPI)
2. Ensure receipt is well-lit and in focus
3. Try the EasyOCR fallback (install with `pip install easyocr`)
4. Review the saved `*_fallback.txt` files to see raw OCR output

### No Items Parsed
If items aren't being parsed:
1. Check the OCR text in `receipts_ocr_text/` folder
2. Adjust the parsing patterns in `parse_items_from_ocr_text()`
3. Add your receipt format to the skip keywords if needed

### Missing Dependencies
```bash
# Install all Python packages
pip install rapidfuzz pydantic opencv-python Pillow pytesseract numpy

# Optional: Install EasyOCR for better accuracy
pip install easyocr
```

## File Structure
```
receipt-ocr/
├── receipt_scanner.py       # Main scanner with improved OCR
├── test_ocr.py             # Test script for single receipts
├── requirements.txt        # Python dependencies
├── receipts/              # Input: receipt images
├── receipts_ocr_text/     # Output: extracted OCR text
└── receipt_price_history.csv  # Output: parsed data
```

## Performance Notes

- **Tesseract**: Fast, works well for clear receipts
- **EasyOCR**: Slower but more accurate for difficult images
- **Processing time**: ~2-5 seconds per receipt with Tesseract, ~10-15 with EasyOCR

## Limitations

- Works best with English-language receipts
- Requires reasonable image quality (not too blurry/dark)
- May struggle with unusual receipt layouts
- Handwritten receipts are not supported
