"""LLM classifier tests. The Gemini SDK is replaced by a fake, so no API key or network is needed."""

import sys
import types
from types import SimpleNamespace

import pytest

import src.llm_classifier as llm
from src.config import CATEGORY_IDS
from src.llm_classifier import ComplaintLabel, LLMClassifier, mock_classify


def test_mock_classify_returns_valid_labels():
    label = mock_classify("Someone made unauthorized charges on my credit card.")
    assert label.category.value == "card"
    assert label.urgency.value == "high"


def test_mock_classify_understands_turkish():
    assert mock_classify("Gönderdiğim EFT karşı hesaba ulaşmadı.").category.value == "money_transfer"
    assert mock_classify("Findeks kredi notum yanlış görünüyor.").category.value == "credit_reporting"


def test_second_call_comes_from_cache(tmp_path):
    clf = LLMClassifier(provider="mock", cache_path=tmp_path / "cache.jsonl")
    first = clf.classify("My mortgage escrow payment doubled.")
    second = clf.classify("My   mortgage escrow payment doubled.  ")  # same text after cleaning
    assert first["cached"] is False and second["cached"] is True
    assert second["category"] == first["category"] == "mortgage"
    # a new classifier reads the cache file back
    reloaded = LLMClassifier(provider="mock", cache_path=tmp_path / "cache.jsonl")
    assert reloaded.classify("My mortgage escrow payment doubled.")["cached"] is True


class FakeAPIError(Exception):
    def __init__(self, code):
        super().__init__(f"status {code}")
        self.code = code


@pytest.fixture
def fake_gemini(monkeypatch):
    """Install a fake google.genai whose responses each test scripts in advance."""
    script = []
    calls = []

    class Models:
        def generate_content(self, model, contents, config):
            calls.append({"model": model, "contents": contents, "config": config})
            outcome = script.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return SimpleNamespace(parsed=outcome, text=outcome.model_dump_json())

    genai = types.ModuleType("google.genai")
    genai.Client = lambda api_key: SimpleNamespace(models=Models())
    genai.errors = SimpleNamespace(APIError=FakeAPIError)
    genai.types = SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs)
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    return script, calls


LABEL = ComplaintLabel(category="debt_collection", urgency="high", sentiment="negative", summary="Collector threats.")


def test_gemini_retries_rate_limit_then_succeeds(fake_gemini, tmp_path):
    script, calls = fake_gemini
    script += [FakeAPIError(429), FakeAPIError(503), LABEL]
    clf = LLMClassifier(provider="gemini", cache_path=tmp_path / "c.jsonl", requests_per_minute=10_000)
    result = clf.classify("They threatened to sue me over a debt that is not mine.")
    assert result["category"] == "debt_collection"
    assert len(calls) == 3
    config = calls[0]["config"]
    assert config["response_schema"] is ComplaintLabel and config["temperature"] == 0.0


def test_gemini_does_not_retry_client_errors(fake_gemini, tmp_path):
    script, calls = fake_gemini
    script += [FakeAPIError(400)]
    clf = LLMClassifier(provider="gemini", cache_path=tmp_path / "c.jsonl", requests_per_minute=10_000)
    with pytest.raises(FakeAPIError):
        clf.classify("Some complaint text that is long enough.")
    assert len(calls) == 1


def test_batch_marks_failures_as_error(fake_gemini, tmp_path):
    script, _ = fake_gemini
    script += [LABEL, FakeAPIError(400)]
    clf = LLMClassifier(provider="gemini", cache_path=tmp_path / "c.jsonl", requests_per_minute=10_000)
    results = clf.classify_many(["first complaint text", "second complaint text"], show_progress=False)
    assert [r["category"] for r in results] == ["debt_collection", "error"]


def test_schema_only_allows_project_categories():
    schema = ComplaintLabel.model_json_schema()
    enum_values = schema["$defs"]["Category"]["enum"]
    assert enum_values == CATEGORY_IDS
