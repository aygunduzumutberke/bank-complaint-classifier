"""Classical baseline: TF-IDF features + Logistic Regression.

This is the yardstick for the LLM: if a cheap, fast model trained on the same data is
just as accurate, the LLM has to justify itself with something else (no training data
needed, other languages, summaries, urgency).

Usage:
    python -m src.baseline
"""

from __future__ import annotations

import json
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR, PROCESSED_DIR, RANDOM_SEED, RESULTS_DIR
from src.metrics import evaluate

MODEL_PATH = MODELS_DIR / "baseline.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),  # single words and two-word phrases ("late fee")
                    min_df=2,  # ignore terms seen in only one complaint
                    max_features=50_000,
                    sublinear_tf=True,  # dampen words repeated many times in one text
                    # CFPB masks personal data as "XXXX"; those tokens appear in every class.
                    stop_words=sorted(ENGLISH_STOP_WORDS | {"xx", "xxxx"}),
                    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # words only, no numbers
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=RANDOM_SEED),
            ),
        ]
    )


def load_model() -> Pipeline:
    return joblib.load(MODEL_PATH)


def main() -> None:
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    model = build_pipeline()
    start = time.perf_counter()
    model.fit(train["text"], train["category"])
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.predict(test["text"])
    predict_seconds = time.perf_counter() - start

    metrics = evaluate(test["category"], predictions)
    metrics["train_seconds"] = round(train_seconds, 2)
    metrics["ms_per_complaint"] = round(1000 * predict_seconds / len(test), 3)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    (RESULTS_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    test.assign(predicted=predictions).to_csv(RESULTS_DIR / "baseline_predictions.csv", index=False)

    print(f"Baseline on {len(test):,} test complaints")
    print(f"  accuracy  {metrics['accuracy']:.3f}")
    print(f"  macro F1  {metrics['macro_f1']:.3f}")
    print(f"  {metrics['ms_per_complaint']} ms per complaint, trained in {train_seconds:.1f} s")


if __name__ == "__main__":
    main()
