"""Write results/RESULTS.md and the figures from the latest evaluation run.

Usage:
    python -m src.report
"""

from __future__ import annotations

import json

import pandas as pd

from src.config import CATEGORIES, PROCESSED_DIR, RESULTS_DIR
from src.plots import confusion_heatmap, f1_by_category

FIG_DIR = RESULTS_DIR / "figures"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    comparison = json.loads((RESULTS_DIR / "comparison.json").read_text())
    preds = pd.read_csv(RESULTS_DIR / "llm_predictions.csv")
    llm, base = comparison["llm"], comparison["baseline"]
    train_size = len(pd.read_csv(PROCESSED_DIR / "train.csv"))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    f1_by_category(comparison).savefig(FIG_DIR / "f1_by_category.png", dpi=160)
    confusion_heatmap(llm, "LLM: actual vs predicted category").savefig(FIG_DIR / "confusion_llm.png", dpi=160)
    confusion_heatmap(base, "Baseline: actual vs predicted category").savefig(
        FIG_DIR / "confusion_baseline.png", dpi=160
    )

    lines = [
        "# Results",
        "",
        f"Both models were scored on the same **{comparison['n']} test complaints** "
        f"({comparison['per_class']} per category). The baseline was trained on {train_size:,} "
        f"complaints; the LLM saw no training examples (zero-shot). LLM: `{llm['model']}`.",
        "",
        "| Model | Accuracy | Macro F1 | Speed |",
        "| --- | --- | --- | --- |",
        f"| LLM (zero-shot) | {_pct(llm['accuracy'])} | {llm['macro_f1']:.3f} | "
        f"~{'–' if llm['median_latency_ms'] is None else llm['median_latency_ms']} ms per complaint |",
        f"| TF-IDF + Logistic Regression | {_pct(base['accuracy'])} | {base['macro_f1']:.3f} | "
        "< 1 ms per complaint |",
        "",
        f"Only the LLM was right on {comparison['only_llm_right']} complaints, only the baseline on "
        f"{comparison['only_baseline_right']}, and both were wrong on {comparison['both_wrong']}. "
        f"Failed LLM calls (scored as wrong): {llm['errors']}.",
        "",
        "![F1 by category](figures/f1_by_category.png)",
        "",
        "## F1 by category",
        "",
        "| Category | LLM | Baseline |",
        "| --- | --- | --- |",
    ]
    for cid in llm["labels"]:
        lines.append(
            f"| {CATEGORIES[cid]['label_en']} | {llm['per_class'][cid]['f1']:.3f} | "
            f"{base['per_class'][cid]['f1']:.3f} |"
        )
    lines += [
        "",
        "## Where the models get confused",
        "",
        "![LLM confusion matrix](figures/confusion_llm.png)",
        "![Baseline confusion matrix](figures/confusion_baseline.png)",
        "",
        "## Urgency assigned by the LLM",
        "",
        "| Urgency | Complaints |",
        "| --- | --- |",
    ]
    for level in ("high", "medium", "low"):
        lines.append(f"| {level} | {int((preds['urgency'] == level).sum())} |")

    wrong = preds[~preds["llm_correct"]].head(5)
    if len(wrong):
        lines += ["", "## Examples the LLM got wrong", "", "| Actual | LLM said | Complaint (first 160 chars) |",
                  "| --- | --- | --- |"]
        for _, row in wrong.iterrows():
            snippet = str(row["text"])[:160].replace("|", "/")
            lines.append(f"| {row['category']} | {row['llm_category']} | {snippet}… |")

    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_DIR / 'RESULTS.md'} and figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
