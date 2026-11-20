import csv
from datetime import datetime

from receipt_scanner import (
    _write_price_history_csv,
    extract_store_and_date_from_filename,
    fuzzy_canonical_name,
    parse_items_from_ocr_text,
)


def test_parser_ignores_savings_and_discount_lines():
    text = """\
QFC MILK 2.79
QFC SAVINGS 1.50
SC QFC DISCOUNT 0.75
AMOUNT 4.29
"""

    items = parse_items_from_ocr_text(text)

    assert [(item.product_name, item.total_price) for item in items] == [
        ("QFC MILK", 2.79)
    ]


def test_tfidf_canonicalizes_ocr_name_variant_after_fuzzy_miss():
    canonical = fuzzy_canonical_name(
        "low fat greek yogurt strawberry", ["lowfat greek yogrt strawberry"]
    )

    assert canonical == "lowfat greek yogrt strawberry"


def test_tfidf_keeps_distinct_product_names_separate():
    canonical = fuzzy_canonical_name("chicken broth", ["chicken breast"])

    assert canonical == "chicken broth"


def test_filename_parser_accepts_duplicate_receipt_suffixes():
    assert extract_store_and_date_from_filename("QFC20260519(1).jpg") == (
        "QFC",
        datetime(2026, 5, 19),
    )
    assert extract_store_and_date_from_filename("Costco20250217_2.jpeg") == (
        "Costco",
        datetime(2025, 2, 17),
    )


def test_history_csv_escapes_fields_and_marks_review_flags(tmp_path):
    history = {
        'product "quoted"': [
            ("Store, Inc.", datetime(2026, 1, 2), 120.5, "receipt.jpg")
        ]
    }
    ocr_metadata = {
        "receipt.jpg": {
            "selected_engine": "EasyOCR",
            "tesseract_confidence": 42.0,
            "low_confidence": True,
        }
    }
    output_path = tmp_path / "history.csv"

    _write_price_history_csv(str(output_path), history, ocr_metadata)

    with output_path.open(newline="", encoding="utf-8") as output_file:
        row = next(csv.DictReader(output_file))
    assert row["product_name"] == 'product "quoted"'
    assert row["store_name"] == "Store, Inc."
    assert row["selected_ocr_engine"] == "EasyOCR"
    assert row["review_flags"] == "low_ocr_confidence;price_over_100"