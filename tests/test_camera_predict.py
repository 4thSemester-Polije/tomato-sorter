import cv2
import numpy as np

from tools.camera_predict import analyze, format_summary

BG = (235, 235, 235)
RESULT = {"architecture": "resnext101", "grade": "B",
          "ripeness": {"class": "setengah_matang", "level": 2, "confidence": 0.8, "probabilities": {}},
          "defect": {"class": "healthy", "level": 1, "confidence": 0.9, "probabilities": {}}}


def _frame(with_tomato):
    img = np.full((480, 640, 3), BG, dtype=np.uint8)
    if with_tomato:
        cv2.circle(img, (320, 240), 90, (30, 30, 200), -1)
    return img


def test_analyze_calls_model_for_a_valid_tomato_and_summarizes():
    calls = []

    def predict(jpeg):
        calls.append(len(jpeg))
        return RESULT

    outcome = analyze(_frame(True), _frame(False), None, predict)

    assert outcome["valid"] and len(calls) == 1 and calls[0] > 0
    text = format_summary(outcome)
    assert "GRADE B" in text and "setengah_matang (80%)" in text and "resnext101" in text


def test_analyze_skips_model_when_no_tomato():
    def predict(jpeg):
        raise AssertionError("API tidak boleh dipanggil tanpa tomat valid")

    outcome = analyze(_frame(False), _frame(False), None, predict)

    assert outcome["valid"] is False
    assert "GRADE_TIMEOUT" in format_summary(outcome)
