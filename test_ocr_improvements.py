from unittest.mock import patch
import types

import numpy as np
import cv2

from receipt_scanner import (
    OLLAMA_RECEIPT_PROMPT,
    ReceiptItem,
    _parse_items_from_tesseract_lines,
    _refine_price_column,
    _score_tesseract_result,
    _clip_easyocr_horizontal_boxes,
    _output_csv_paths,
    extract_receipt_data_from_ocr,
    ocr_receipt_to_text,
    parse_args,
    rectify_receipt,
    try_paddleocr,
)


def test_tesseract_confidence_and_price_alignment_raise_score():
    aligned_lines = [
        [
            {"text": "MILK", "left": 100, "top": 10, "width": 100, "height": 20, "confidence": 90},
            {"text": "2.79", "left": 850, "top": 10, "width": 50, "height": 20, "confidence": 90},
        ],
        [
            {"text": "BREAD", "left": 100, "top": 40, "width": 120, "height": 20, "confidence": 90},
            {"text": "3.49", "left": 852, "top": 40, "width": 50, "height": 20, "confidence": 90},
        ],
    ]
    text = "MILK 2.79\nBREAD 3.49"

    high_score, alignment = _score_tesseract_result(text, 90, aligned_lines, 1000)
    low_score, _ = _score_tesseract_result(text, 35, aligned_lines, 1000)

    assert alignment > 90
    assert high_score > low_score


def test_tesseract_candidate_score_rewards_successfully_parsed_rows():
    lines = [
        [
            {"text": "MILK", "left": 100, "top": 10, "width": 100, "height": 20, "confidence": 90},
            {"text": "2.79", "left": 850, "top": 10, "width": 50, "height": 20, "confidence": 90},
        ],
        [
            {"text": "BREAD", "left": 100, "top": 40, "width": 120, "height": 20, "confidence": 90},
            {"text": "3.49", "left": 852, "top": 40, "width": 50, "height": 20, "confidence": 90},
        ],
    ]
    text = "MILK 2.79\nBREAD 3.49"

    text_only_score, _ = _score_tesseract_result(text, 90, lines, 1000)
    parsed_score, _ = _score_tesseract_result(text, 90, lines, 1000, 2)

    assert parsed_score > text_only_score


def test_tesseract_parser_uses_word_positions_to_select_rightmost_price():
    lines = [[
        {"text": "MILK", "left": 100, "top": 10, "width": 80, "height": 20, "confidence": 90},
        {"text": "3.49", "left": 850, "top": 10, "width": 50, "height": 20, "confidence": 90},
        {"text": "2.79", "left": 700, "top": 10, "width": 50, "height": 20, "confidence": 90},
    ]]

    items = _parse_items_from_tesseract_lines(lines)

    assert [(item.product_name, item.total_price) for item in items] == [("MILK", 3.49)]


def test_receipt_extraction_uses_structured_tesseract_items():
    positioned_item = ReceiptItem(product_name="MILK", total_price=2.79)

    def fake_ocr(*args, **kwargs):
        kwargs["metadata"]["selected_engine"] = "Tesseract"
        kwargs["metadata"]["parsed_items"] = [positioned_item]
        return "MILK 99.99"

    with patch("receipt_scanner.ocr_receipt_to_text", side_effect=fake_ocr):
        receipt, error = extract_receipt_data_from_ocr("receipt.jpg", {})

    assert error is None
    assert receipt is not None
    assert [(item.product_name, item.total_price) for item in receipt.items] == [
        ("MILK", 2.79)
    ]


def test_receipt_extraction_ignores_tesseract_items_after_engine_fallback():
    positioned_item = ReceiptItem(product_name="MILK", total_price=2.79)

    def fake_ocr(*args, **kwargs):
        kwargs["metadata"]["selected_engine"] = "EasyOCR"
        kwargs["metadata"]["parsed_items"] = [positioned_item]
        return "BREAD 3.49"

    with patch("receipt_scanner.ocr_receipt_to_text", side_effect=fake_ocr):
        receipt, error = extract_receipt_data_from_ocr("receipt.jpg", {})

    assert error is None
    assert receipt is not None
    assert [(item.product_name, item.total_price) for item in receipt.items] == [
        ("BREAD", 3.49)
    ]


def test_price_column_reocr_preserves_rows_and_uses_whitelisted_text():
    image = np.full((100, 1000), 255, dtype=np.uint8)
    lines = [
        [
            {"text": "MILK", "left": 100, "top": 10, "width": 120, "height": 20, "confidence": 80},
            {"text": "2.79", "left": 850, "top": 10, "width": 50, "height": 20, "confidence": 60},
        ],
        [
            {"text": "BREAD", "left": 100, "top": 50, "width": 150, "height": 20, "confidence": 80},
            {"text": "3.49", "left": 852, "top": 50, "width": 50, "height": 20, "confidence": 60},
        ],
    ]

    with patch("receipt_scanner.pytesseract.image_to_string", return_value="8.49") as ocr:
        text = _refine_price_column(image, lines)

    assert text == "MILK 8.49\nBREAD 8.49"
    assert ocr.call_count == 2
    assert "tessedit_char_whitelist=0123456789.$," in ocr.call_args.kwargs["config"]


def test_rectify_receipt_leaves_blank_image_unchanged():
    image = np.full((120, 80, 3), 255, dtype=np.uint8)

    result = rectify_receipt(image)

    assert result.shape == image.shape


def test_rectify_receipt_corrects_skewed_paper():
    canvas = np.zeros((800, 600, 3), dtype=np.uint8)
    receipt = np.full((600, 300, 3), 255, dtype=np.uint8)
    source_corners = np.float32([[0, 0], [299, 0], [299, 599], [0, 599]])
    skewed_corners = np.float32([[100, 70], [490, 110], [450, 730], [130, 700]])
    skewed = cv2.warpPerspective(
        receipt,
        cv2.getPerspectiveTransform(source_corners, skewed_corners),
        (600, 800),
    )
    canvas = np.maximum(canvas, skewed)

    corrected = rectify_receipt(canvas)

    assert corrected.shape[0] > corrected.shape[1]
    assert abs(corrected.shape[1] - 392) < 20


def test_paddleocr_skips_when_runtime_is_missing():
    with patch("receipt_scanner.find_spec", return_value=None):
        assert try_paddleocr("not-opened.jpg") == ""


def test_paddleocr_uses_predict_and_caps_input_size():
    received_shapes = []

    class FakePaddleOCR:
        def __init__(self, use_textline_orientation=True, lang="en"):
            pass

        def predict(self, input):
            received_shapes.append(input.shape)
            return [{"res": {"rec_texts": ["MILK", "2.79"]}}]

    paddleocr_module = types.ModuleType("paddleocr")
    paddleocr_module.PaddleOCR = FakePaddleOCR
    image = np.zeros((3000, 200, 3), dtype=np.uint8)

    with (
        patch.dict("sys.modules", {"paddleocr": paddleocr_module}),
        patch("receipt_scanner.find_spec", return_value=object()),
        patch("receipt_scanner.cv2.imread", return_value=image),
    ):
        text = try_paddleocr("receipt.jpg")

    assert text == "MILK\n2.79"
    assert received_shapes == [(2400, 160, 3)]


def test_easyocr_boxes_are_clipped_and_empty_boxes_rejected():
    boxes = [[-10, 50, 5, 30], [900, 1100, 10, 40], [1200, 1300, 20, 50]]

    assert _clip_easyocr_horizontal_boxes(boxes, 1000, 60) == [
        [0, 50, 5, 30],
        [900, 1000, 10, 40],
    ]


def test_ollama_ocr_uses_requested_model_and_image():
    requests = []
    ollama_module = types.ModuleType("ollama")
    ollama_module.chat = lambda **kwargs: (
        requests.append(kwargs) or {"message": {"content": "MILK 2.79"}}
    )
    metadata = {}

    with (
        patch.dict("sys.modules", {"ollama": ollama_module}),
        patch("receipt_scanner.cv2.imread", return_value=np.zeros((20, 20, 3))),
    ):
        text = ocr_receipt_to_text(
            "receipt.jpg",
            save_output=False,
            metadata=metadata,
            engine="ollama",
            ollama_model="qwen2.5vl:7b",
        )

    assert text == "MILK 2.79"
    assert requests[0]["model"] == "qwen2.5vl:7b"
    assert requests[0]["messages"][0]["content"] == OLLAMA_RECEIPT_PROMPT
    assert "all printed price columns" in OLLAMA_RECEIPT_PROMPT
    assert "left-to-right order" in OLLAMA_RECEIPT_PROMPT
    assert requests[0]["messages"][0]["images"] == ["receipt.jpg"]
    assert metadata["selected_engine"] == "Ollama/qwen2.5vl:7b"
    assert metadata["tesseract_confidence"] is None


def test_ollama_ocr_saves_to_a_separate_model_specific_file():
    ollama_module = types.ModuleType("ollama")
    ollama_module.chat = lambda **kwargs: {"message": {"content": "MILK 2.79"}}

    with (
        patch.dict("sys.modules", {"ollama": ollama_module}),
        patch("receipt_scanner.cv2.imread", return_value=np.zeros((20, 20, 3))),
        patch("receipt_scanner.save_fallback_text") as save_text,
    ):
        ocr_receipt_to_text(
            "receipt.jpg", engine="ollama", ollama_model="qwen2.5vl:7b"
        )

    save_text.assert_called_once_with(
        "receipt.jpg", "MILK 2.79", "ollama_qwen2_5-vl"
    )


def test_ocr_engine_cli_defaults_to_tesseract_and_accepts_ollama():
    defaults = parse_args([])
    ollama_args = parse_args(
        ["--ocr-engine", "ollama", "--ollama-model", "qwen2.5vl:7b"]
    )

    assert defaults.ocr_engine == "tesseract"
    assert defaults.ollama_model == "qwen2.5vl:7b"
    assert ollama_args.ocr_engine == "ollama"
    assert ollama_args.ollama_model == "qwen2.5vl:7b"


def test_ollama_csv_paths_are_separate_and_model_specific():
    assert _output_csv_paths("tesseract", "qwen2.5vl:7b") == (
        "receipt_price_history.csv",
        "receipt_processing_summary.csv",
    )
    assert _output_csv_paths("ollama", "qwen2.5vl:7b") == (
        "receipt_price_history_ollama_qwen2_5vl_7b.csv",
        "receipt_processing_summary_ollama_qwen2_5vl_7b.csv",
    )
