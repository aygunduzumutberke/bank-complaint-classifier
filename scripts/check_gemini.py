"""Make one real Gemini call to confirm the API key and model work. Never prints the key.

Usage:
    python scripts/check_gemini.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import GEMINI_MODEL  # noqa: E402  (also loads .env)
from src.llm_classifier import LLMClassifier  # noqa: E402

SAMPLE = (
    "Üç gün önce başka bir bankadaki hesabıma 15.000 TL EFT gönderdim. Para hesabımdan çıktı "
    "ama karşı hesaba geçmedi. Müşteri hizmetleri sadece bekleyin diyor."
)


def main() -> None:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        sys.exit("GEMINI_KEY_MISSING: .env has no GEMINI_API_KEY value")
    print(f"Key found ({len(key)} characters). Model: {GEMINI_MODEL}")
    result = LLMClassifier(provider="gemini", cache_path=None).classify(SAMPLE)
    print(f"GEMINI_OK category={result['category']} urgency={result['urgency']} "
          f"latency={result['latency_ms']} ms")
    print(f"summary: {result['summary']}")


if __name__ == "__main__":
    main()
