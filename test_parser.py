from receipt_scanner import fuzzy_canonical_name, parse_items_from_ocr_text


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