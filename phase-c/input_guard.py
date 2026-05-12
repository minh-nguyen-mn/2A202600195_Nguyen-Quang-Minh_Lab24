import re
import time

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


VN_PII = {
    "cccd": r"\b\d{12}\b",
    "phone_vn": r"(\+84|0)\d{9,10}",
    "tax_code": r"\b\d{10}(-\d{3})?\b",
    "email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
}


class InputGuard:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def scrub_vn(self, text):
        for name, pattern in VN_PII.items():
            text = re.sub(
                pattern,
                f"[{name.upper()}]",
                text,
            )

        return text

    def scrub_ner(self, text):
        results = self.analyzer.analyze(
            text=text,
            language="en",
        )

        return self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
        ).text

    def sanitize(self, text):
        start = time.perf_counter()

        text = self.scrub_vn(text)

        text = self.scrub_ner(text)

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        return text, latency_ms


if __name__ == "__main__":
    guard = InputGuard()

    sample = "CCCD 012345678901 phone 0987654321"

    out, latency = guard.sanitize(sample)

    print(out)
    print(latency)