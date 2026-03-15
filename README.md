# receipt-ocr
Overhauled receipt text extraction to track price history.

Code generated with input from Claude Sonnet 4.5 2026-01-12

source ~/.bashrc
source receipt_history_env/bin/activate
From within receipt_price_history folder:python receipt_scanner.py

Summary - OCR Library Comparison & Current Setup
PaddleOCR is generally considered superior to Tesseract for receipt OCR because:

✅ Deep learning-based (more accurate)
✅ Built-in rotation detection
✅ Better handling of low-quality images
✅ Faster than EasyOCR
⚠️ Requires model download on first run (can be slow)
Current Implementation:
Your system now uses a multi-tier OCR strategy:

Tesseract (Primary - fast & effective)

Tests all 4 rotations (0°, 90°, 180°, 270°)
Tries multiple PSM modes
Scores results and picks the best
✅ Works well: 8/14 items on Trader Joe's receipt
PaddleOCR (Backup - if installed)

Runs if Tesseract fails
More accurate but slower first run
Optional enhancement
Advanced Preprocessing (Fallback)

CLAHE, denoising, morphological ops
EasyOCR (Last Resort)

Deep learning alternative
Current Performance:
Safeway receipt: 2/2 items ✅
Trader Joe's receipt: 8/14 items ✅
Automatic rotation detection working perfectly
Clean product names (removes artifacts like "ff", "&f")
