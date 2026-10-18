from controller.serial_link import SerialLink
from tests.mock_serial import MockSerial


def test_send_grade_writes_formatted_line():
    mock = MockSerial()
    SerialLink(serial_obj=mock).send_grade(7, "C")
    assert mock.written == [b"GRADE,7,C\n"]


def test_full_cycle_with_debug_noise():
    mock = MockSerial([b"READY\n", b"# servo1 kick\n", b"DETECT,7\n", b"garbage\n", b"ACK,7\n", b"DONE,7,C\n"])
    link = SerialLink(serial_obj=mock)
    assert link.wait_for({"READY"}, timeout_s=1)["type"] == "READY"
    assert link.wait_for({"DETECT"}, timeout_s=1) == {"type": "DETECT", "id": 7}
    assert link.wait_for({"ACK"}, timeout_s=1) == {"type": "ACK", "id": 7}
    assert link.wait_for({"DONE", "ERR"}, timeout_s=1) == {"type": "DONE", "id": 7, "grade": "C"}


def test_wait_for_times_out():
    link = SerialLink(serial_obj=MockSerial())
    assert link.wait_for({"DONE"}, timeout_s=0.2) is None
