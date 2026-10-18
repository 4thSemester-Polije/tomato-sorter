import pytest

from controller.protocol import format_grade, parse_message


def test_format_grade():
    assert format_grade(12, "b") == "GRADE,12,B\n"


@pytest.mark.parametrize("grade", ["D", "", "AB"])
def test_format_grade_invalid(grade):
    with pytest.raises(ValueError):
        format_grade(1, grade)


def test_format_grade_negative_id():
    with pytest.raises(ValueError):
        format_grade(-1, "A")


def test_parse_valid():
    assert parse_message("READY\n") == {"type": "READY"}
    assert parse_message("DETECT,12") == {"type": "DETECT", "id": 12}
    assert parse_message("ACK,12") == {"type": "ACK", "id": 12}
    assert parse_message("DONE,12,B\r\n") == {"type": "DONE", "id": 12, "grade": "B"}
    assert parse_message("ERR,3,grade_timeout") == {"type": "ERR", "id": 3, "code": "GRADE_TIMEOUT"}


@pytest.mark.parametrize(
    "line",
    ["", "   ", "HELLO", "ACK", "ACK,x", "DONE,1,Z", "DONE,1", "READY,1", "DETECT", "DETECT,x", "# servo1 kick", "#DETECT,1"],
)
def test_parse_invalid_or_debug(line):
    assert parse_message(line) is None
