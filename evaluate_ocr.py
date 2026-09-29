"""Compare OCR output with saved fallback text; fallback files are prior OCR, not ground truth."""

import argparse
import os
import re
from pathlib import Path
from statistics import mean
from typing import List, Tuple

from rapidfuzz.distance import Levenshtein

from receipt_scanner import ocr_receipt_to_text


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def evaluate_receipts(
    receipt_dir: Path, reference_dir: Path, limit: int = 0
) -> List[Tuple[str, float, float]]:
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    image_paths = sorted(
        path for path in receipt_dir.rglob("*") if path.suffix.lower() in image_extensions
    )
    if limit > 0:
        image_paths = image_paths[:limit]

    results = []
    for image_path in image_paths:
        reference_path = reference_dir / f"{image_path.stem}_fallback.txt"
        if not reference_path.exists():
            print(f"[SKIP] No prior OCR reference for {image_path.name}")
            continue

        reference = normalize_text(reference_path.read_text(encoding="utf-8"))
        recognized = normalize_text(ocr_receipt_to_text(str(image_path), save_output=False))
        error_rate = Levenshtein.normalized_distance(reference, recognized)
        results.append((image_path.name, error_rate, 1.0 - error_rate))
        print(f"{image_path.name}: normalized edit distance {error_rate:.3f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, default=Path("receipts"))
    parser.add_argument("--references", type=Path, default=Path("receipts_ocr_text"))
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of receipts to evaluate")
    args = parser.parse_args()

    results = evaluate_receipts(args.receipts, args.references, args.limit)
    if not results:
        print("No receipt/reference pairs were evaluated.")
        return

    print(
        f"Mean normalized edit distance: {mean(result[1] for result in results):.3f} "
        "(lower is more similar to prior OCR, not necessarily more accurate)."
    )


if __name__ == "__main__":
    main()
