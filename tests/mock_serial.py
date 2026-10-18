"""Objek serial tiruan untuk pengujian tanpa hardware."""


class MockSerial:
    def __init__(self, incoming=None):
        self.incoming = list(incoming or [])  # baris (bytes) yang akan "dikirim Arduino"
        self.written = []

    def write(self, data):
        self.written.append(data)

    def flush(self):
        pass

    def readline(self):
        return self.incoming.pop(0) if self.incoming else b""

    def close(self):
        pass
