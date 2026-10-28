"""Score the LLM and the baseline on the same test complaints and write the comparison.

The free Gemini tier allows only a few requests per minute, so the LLM is scored on a
stratified subset of the test set. The baseline is re-scored on exactly that subset so
the two numbers are comparable.

Usage:
    python -m src.evaluate_llm --per-class 30              # 210 complaints, ~30 min on the free tier
    python -m src.evaluate_llm --per-class 5 --provider mock   # quick offline check
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from src.baseline import load_model
from src.config import PROCESSED_DIR, RANDOM_SEED, RESULTS_DIR
from src.llm_classifier import LLMClassifier
from src.metrics import evaluate


def stratified_subset(test: pd.DataFrame, per_class: int) -> pd.DataFrame:
    """Shuffle once, then take the first `per_class` rows of every category."""
    shuffled = test.sample(frac=1, random_state=RANDOM_SEED)
    return shuffled.groupby("category").head(per_class).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=30, help="Test complaints per category")
    parser.add_argument("--provider", choices=["gemini", "mock"], default="gemini")
    args = parser.parse_args()

    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    subset = stratified_subset(test, args.per_class)
    print(f"Evaluating on {len(subset)} complaints ({args.per_class} per category)")

    classifier = LLMClassifier(provider=args.provider)
    start = time.perf_counter()
    llm_results = classifier.classify_many(subset["text"])
    wall_seconds = time.perf_counter() - start

    llm_df = pd.DataFrame(llm_results)
    subset = subset.assign(
        llm_category=llm_df["category"],
        urgency=llm_df["urgency"],
        sentiment=llm_df["sentiment"],
        summary=llm_df["summary"],
        baseline_category=load_model().predict(subset["text"]),
    )
    subset["llm_correct"] = subset["llm_category"] == subset["category"]
    subset["baseline_correct"] = subset["baseline_category"] == subset["category"]

    llm_metrics = evaluate(subset["category"], subset["llm_category"])
    fresh = llm_df[~llm_df["cached"].astype(bool) & (llm_df["category"] != "error")]
    llm_metrics.update(
        provider=classifier.provider,
        model=classifier.model,
        errors=int((llm_df["category"] == "error").sum()),
        median_latency_ms=int(fresh["latency_ms"].median()) if len(fresh) else None,
        wall_seconds=round(wall_seconds, 1),
    )
    comparison = {
        "n": len(subset),
        "per_class": args.per_class,
        "llm": llm_metrics,
        "baseline": evaluate(subset["category"], subset["baseline_category"]),
        "both_wrong": int((~subset["llm_correct"] & ~subset["baseline_correct"]).sum()),
        "only_llm_right": int((subset["llm_correct"] & ~subset["baseline_correct"]).sum()),
        "only_baseline_right": int((~subset["llm_correct"] & subset["baseline_correct"]).sum()),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    subset.to_csv(RESULTS_DIR / "llm_predictions.csv", index=False)
    (RESULTS_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2))

    print(f"\n{'':10}{'accuracy':>10}{'macro F1':>10}")
    for name in ("llm", "baseline"):
        m = comparison[name]
        print(f"{name:10}{m['accuracy']:>10.3f}{m['macro_f1']:>10.3f}")
    print(f"\nLLM errors: {llm_metrics['errors']}  |  saved to {RESULTS_DIR}")
    print("Next: python -m src.report")


if __name__ == "__main__":
    main()
