from unittest.mock import patch
import types

import numpy as np
import cv2

from receipt_scanner import (
    _refine_price_column,
    _score_tesseract_result,
    _clip_easyocr_horizontal_boxes,
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
