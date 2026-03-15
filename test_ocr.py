#!/usr/bin/env python3
"""
Quick test script to verify OCR improvements on a single receipt.
"""

import os
import sys
from receipt_scanner import ocr_receipt_to_text, parse_items_from_ocr_text

def test_receipt(image_path: str):
    """Test OCR and parsing on a single receipt image."""
    
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return
    
    print(f"Testing OCR on: {image_path}\n")
    print("=" * 70)
    
    # Extract text
    print("\n1. Extracting text with improved OCR...\n")
    ocr_text = ocr_receipt_to_text(image_path)
    
    print("\n2. OCR Text Output:")
    print("-" * 70)
    print(ocr_text)
    print("-" * 70)
    
    # Parse items
    print("\n3. Parsing items from OCR text...\n")
    items = parse_items_from_ocr_text(ocr_text)
    
    print(f"\n4. Parsed {len(items)} items:")
    print("-" * 70)
    for i, item in enumerate(items, 1):
        qty_str = f"{item.quantity}x " if item.quantity else ""
        print(f"{i}. {qty_str}{item.product_name}: ${item.total_price:.2f}")
    print("-" * 70)
    
    if not items:
        print("\n⚠️  WARNING: No items were parsed!")
        print("The OCR text may need manual review.")
    else:
        print(f"\n✅ Success! Extracted {len(items)} items from receipt.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default test receipt
        image_path = "./receipts/Safeway20250519.jpg"
    
    test_receipt(image_path)
