# receipt-ocr
Overhauled receipt text extraction to track price history.

Code generated with input from Claude Sonnet 4.5 2026-01-12

source ~/.bashrc
source receipt_history_env/bin/activate
python receipt_scanner.py

The default command uses the existing Tesseract-first OCR pipeline. To run the
same batch through an Ollama vision model instead, install and start Ollama,
install its Python client, and download the model:

```sh
ollama pull qwen2.5vl:7b
pip install ollama
python receipt_scanner.py --ocr-engine ollama
```

Use `--ollama-model MODEL_NAME` to select another installed vision model. The
Ollama mode sends each receipt image directly to the model for transcription;
the prompt preserves printed lines and price columns, and the existing
rule-based item parser still converts that text to line items. It writes
model-specific `receipt_price_history_ollama_<model>.csv` and
`receipt_processing_summary_ollama_<model>.csv` files so the results can be
compared with the default Tesseract CSVs.
Ollama mode does not silently fall back to Tesseract if the client, service, or
model is unavailable. Its transcript is saved as a model-specific `_ollama_*.txt`
file, leaving the existing `_fallback.txt` reference text unchanged. The
selected engine is recorded in the output CSV.

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
