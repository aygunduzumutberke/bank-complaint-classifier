"""One place for evaluation, so the classical model and the LLM are scored identically."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.config import CATEGORY_IDS


def evaluate(y_true: Sequence[str], y_pred: Sequence[str], labels: list[str] | None = None) -> dict:
    """Accuracy, macro/weighted F1, per-class scores and the confusion matrix.

    Predictions outside `labels` (for example a failed LLM call marked "error") count
    as wrong answers rather than being dropped, so failures cannot inflate the score.
    """
    labels = labels or CATEGORY_IDS
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "n": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4
        ),
        "per_class": {
            label: {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
                "support": int(s),
            }
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
