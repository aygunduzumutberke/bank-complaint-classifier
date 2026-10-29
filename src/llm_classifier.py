"""LLM complaint classifier using Gemini structured output.

The model must answer with JSON that matches `ComplaintLabel`; the category field is an
enum, so it cannot invent a class outside the seven we score. Results are cached on
disk, and calls are spaced out to stay inside the free-tier rate limit.

Providers:
    gemini  real model, needs GEMINI_API_KEY in .env
    mock    keyword rules, no API key; only for trying the app and running tests
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Iterable
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import CATEGORIES, CATEGORY_IDS, GEMINI_MODEL, MAX_CHARS, REQUESTS_PER_MINUTE, RESULTS_DIR

PROMPT_VERSION = "v1"  # bump when the prompt changes, so old cached answers are not reused
CACHE_PATH = RESULTS_DIR / "llm_cache.jsonl"
RETRYABLE_CODES = {429, 500, 503}
MAX_RETRIES = 5

Category = Enum("Category", {c: c for c in CATEGORY_IDS}, type=str)


class Urgency(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Sentiment(str, Enum):
    neutral = "neutral"
    negative = "negative"
    very_negative = "very_negative"


class ComplaintLabel(BaseModel):
    category: Category = Field(description="The single product area the complaint is about")
    urgency: Urgency = Field(description="How quickly the bank should act")
    sentiment: Sentiment = Field(description="The customer's tone")
    summary: str = Field(description="One short sentence, written in the same language as the complaint")


def _category_guide() -> str:
    return "\n".join(f"- {cid}: {info['description']}" for cid, info in CATEGORIES.items())


SYSTEM_PROMPT = f"""You classify customer complaints sent to a retail bank.

Pick exactly one category, the product the complaint is mainly about:
{_category_guide()}

Rules:
- If a complaint mentions several products, choose the one the customer wants fixed.
- A wrong entry on a credit report is credit_reporting, even if it comes from a card or loan.
- Calls or letters demanding payment of a debt are debt_collection, whatever the original product.

Urgency:
- high: fraud, unauthorized transactions, identity theft, scams, money the customer cannot
  access, foreclosure, repossession, lawsuits or legal threats
- medium: money lost to fees, interest or wrong charges; repeated failed attempts to fix a problem
- low: service quality, delays or information requests with no direct financial loss

The complaint may be in any language, including Turkish. "XXXX" marks masked personal data.
Write the summary in the complaint's own language."""


class LLMClassifier:
    def __init__(
        self,
        provider: str = "gemini",
        model: str = GEMINI_MODEL,
        requests_per_minute: int = REQUESTS_PER_MINUTE,
        cache_path: Path | None = CACHE_PATH,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        if provider not in {"gemini", "mock"}:
            raise ValueError(f"unknown provider: {provider}")
        self.provider = provider
        self.model = model if provider == "gemini" else "keyword-rules"
        self.min_interval = 60.0 / max(requests_per_minute, 1)
        self.max_retries = max_retries
        self._last_call = 0.0
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self._client = None
        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
            from google import genai  # imported here so tests and mock mode need no SDK

            from google.genai import types

            # 90 s per request; without a timeout a stuck request can hang the whole batch.
            self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=90_000))

    # ---- cache -------------------------------------------------------------
    def _cache_key(self, text: str) -> str:
        raw = f"{self.provider}|{self.model}|{PROMPT_VERSION}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load_cache(self) -> dict[str, dict]:
        if not self.cache_path or not self.cache_path.exists():
            return {}
        cache = {}
        with self.cache_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    cache[entry["key"]] = entry["result"]
        return cache

    def _save_to_cache(self, key: str, result: dict) -> None:
        self.cache[key] = result
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "result": result}, ensure_ascii=False) + "\n")

    # ---- classification ----------------------------------------------------
    def classify(self, text: str) -> dict:
        """Classify one complaint. Returns category, urgency, sentiment, summary and timing."""
        text = re.sub(r"\s+", " ", str(text)).strip()[:MAX_CHARS]
        key = self._cache_key(text)
        if key in self.cache:
            return {**self.cache[key], "cached": True}

        start = time.perf_counter()
        label = self._call_gemini(text) if self.provider == "gemini" else mock_classify(text)
        result = {
            **label.model_dump(mode="json"),
            "provider": self.provider,
            "model": self.model,
            "latency_ms": round(1000 * (time.perf_counter() - start)),
        }
        self._save_to_cache(key, result)
        return {**result, "cached": False}

    def classify_many(self, texts: Iterable[str], show_progress: bool = True) -> list[dict]:
        """Classify a batch. A complaint that still fails after retries gets category "error"."""
        texts = list(texts)
        results = []
        for i, text in enumerate(texts, start=1):
            try:
                results.append(self.classify(text))
            except Exception as exc:  # keep the batch going; the failure is recorded and scored as wrong
                results.append({"category": "error", "urgency": None, "sentiment": None,
                                "summary": f"ERROR: {exc}", "cached": False})
            if show_progress and (i % 10 == 0 or i == len(texts)):
                print(f"  {i}/{len(texts)} classified")
        return results

    def _wait_for_rate_limit(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _call_gemini(self, text: str) -> ComplaintLabel:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ComplaintLabel,
            temperature=0.0,  # same input -> same answer, which makes evaluation repeatable
        )
        for attempt in range(self.max_retries):
            self._wait_for_rate_limit()
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=f"Complaint:\n{text}", config=config
                )
                return response.parsed or ComplaintLabel.model_validate_json(response.text)
            except errors.APIError as exc:
                retryable = exc.code in RETRYABLE_CODES and attempt < self.max_retries - 1
                wait = min(60, 5 * 2**attempt)
                message = " ".join(str(getattr(exc, "message", "") or exc).split())[:300]
                print(f"    Gemini error {exc.code}: {message}" + (f" -> retrying in {wait} s" if retryable else ""))
                if not retryable:
                    raise
                time.sleep(wait)  # back off: 5, 10, 20, 40 s
        raise RuntimeError("unreachable")


# ---- mock provider: transparent keyword rules, English + Turkish -------------
_MOCK_KEYWORDS = {
    "bank_account": ["checking", "savings", "overdraft", "deposit", "account closed", "frozen",
                     "vadesiz", "mevduat", "hesabım", "hesap"],
    "card": ["credit card", "card", "chargeback", "rewards", "prepaid", "kredi kartı", "kart"],
    "mortgage": ["mortgage", "escrow", "foreclosure", "home loan", "refinance", "konut kredisi", "ipotek"],
    "consumer_loan": ["payday", "auto loan", "car loan", "student loan", "personal loan", "repossess",
                      "title loan", "ihtiyaç kredisi", "taşıt kredisi", "kredi taksit"],
    "money_transfer": ["wire", "transfer", "remittance", "money order", "zelle", "paypal", "venmo",
                       "havale", "eft", "fast", "para transfer"],
    "debt_collection": ["collection", "collector", "debt", "validate", "icra", "tahsilat", "borç"],
    "credit_reporting": ["credit report", "credit bureau", "credit score", "equifax", "experian",
                         "transunion", "findeks", "kredi notu", "kredi sicil"],
}
_HIGH = ["fraud", "unauthorized", "stolen", "identity theft", "scam", "foreclosure", "repossess",
         "lawsuit", "sue ", "cannot access", "dolandır", "izinsiz", "çalın", "haciz", "icra"]
_MEDIUM = ["fee", "charge", "refund", "interest", "late", "ücret", "iade", "faiz", "kesinti"]
_VERY_NEGATIVE = ["unacceptable", "outrageous", "scam", "disgusting", "rezalet", "skandal", "berbat"]


def mock_classify(text: str) -> ComplaintLabel:
    lower = text.lower()
    scores = {cat: sum(lower.count(word) for word in words) for cat, words in _MOCK_KEYWORDS.items()}
    category = max(scores, key=scores.get) if any(scores.values()) else "bank_account"
    if any(w in lower for w in _HIGH):
        urgency = Urgency.high
    elif any(w in lower for w in _MEDIUM):
        urgency = Urgency.medium
    else:
        urgency = Urgency.low
    sentiment = Sentiment.very_negative if any(w in lower for w in _VERY_NEGATIVE) else Sentiment.negative
    first_sentence = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    return ComplaintLabel(category=category, urgency=urgency, sentiment=sentiment, summary=first_sentence[:200])
