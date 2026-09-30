# receipt-ocr
Overhauled receipt text extraction to track price history.

Code generated with input from Claude Sonnet 4.5 2026-01-12

source ~/.bashrc
source receipt_history_env/bin/activate
python receipt_scanner.py

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

## Docker

Build and run the Python API from this directory:

```sh
docker build -t receipt-ocr-api:local .
docker run --rm -p 8000:8000 receipt-ocr-api:local
```

The API accepts receipt images at `/api/extract` and returns parsed line items.
With the container running, check its health and submit a sample receipt from a
second terminal:

```sh
curl http://localhost:8000/api/health
curl -X POST -F "file=@receipts/Safeway20250505.jpg" \
	http://localhost:8000/api/extract
```

For the complete UI and API workflow, use Docker Compose from the sibling
`receipt-ocr-ui` repository; see its [Docker instructions](../receipt-ocr-ui/README.md#docker).

For batch processing of the mounted `receipts/` directory, run:

```sh
docker run --rm --entrypoint python -v "$PWD:/workspace" -w /workspace \
	receipt-ocr-api:local /app/receipt_scanner.py
```

The image includes Tesseract and the Python OCR dependencies.
