"""Turn the raw CFPB export into a balanced, cleaned train/test split.

The full CFPB file is several gigabytes, so it is read in chunks and only a uniform
random sample per category is kept in memory.

Usage:
    python -m src.prepare_data --input data/raw/complaints.csv.zip
    python -m src.prepare_data --demo          # synthetic sample, for a quick test run
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORY_IDS,
    MAX_CHARS,
    PROCESSED_DIR,
    RANDOM_SEED,
    SAMPLE_PATH,
    map_product,
)

CFPB_COLUMNS = {
    "Complaint ID": "complaint_id",
    "Date received": "date_received",
    "Product": "product",
    "Issue": "issue",
    "Company": "company",
    "Consumer complaint narrative": "text",
}
MIN_CHARS = 50
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Collapse whitespace and cut very long narratives.

    CFPB already masks personal data as "XXXX"; those tokens are kept because they
    carry no information either model could leak.
    """
    text = _WHITESPACE.sub(" ", str(text)).strip()
    return text[:MAX_CHARS]


def _prepare_chunk(chunk: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    chunk = chunk.rename(columns=CFPB_COLUMNS)
    chunk = chunk.dropna(subset=["text"])
    chunk["category"] = chunk["product"].map(map_product)
    chunk = chunk.dropna(subset=["category"])
    chunk["text"] = chunk["text"].map(clean_text)
    chunk = chunk[chunk["text"].str.len() >= MIN_CHARS]
    # A random key per row: keeping the smallest keys per class gives a uniform
    # random sample without ever holding the whole file in memory.
    chunk["_key"] = rng.random(len(chunk))
    return chunk


def sample_balanced(path: str, per_class: int, chunksize: int = 200_000) -> pd.DataFrame:
    """Stream a CFPB csv/zip and keep up to `per_class` random complaints per category."""
    rng = np.random.default_rng(RANDOM_SEED)
    kept = pd.DataFrame()
    reader = pd.read_csv(
        path,
        usecols=list(CFPB_COLUMNS),
        dtype=str,
        chunksize=chunksize,
        on_bad_lines="skip",
    )
    for i, chunk in enumerate(reader, start=1):
        kept = pd.concat([kept, _prepare_chunk(chunk, rng)], ignore_index=True)
        kept = kept.sort_values("_key").groupby("category", group_keys=False).head(per_class)
        print(f"  chunk {i}: {len(kept):,} complaints kept so far")
    kept = kept.drop_duplicates(subset="text").drop(columns="_key")
    return kept.reset_index(drop=True)


def split_and_save(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split so every category has the same share in train and test."""
    train, test = train_test_split(
        df, test_size=test_size, stratify=df["category"], random_state=RANDOM_SEED
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", help="Path to complaints.csv or complaints.csv.zip from CFPB")
    parser.add_argument("--demo", action="store_true", help="Use the small synthetic sample instead")
    parser.add_argument("--per-class", type=int, default=1500, help="Complaints kept per category")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    if args.demo:
        print(f"Using synthetic demo data: {SAMPLE_PATH}")
        df = pd.read_csv(SAMPLE_PATH)
    elif args.input:
        print(f"Reading {args.input} in chunks ...")
        df = sample_balanced(args.input, args.per_class)
    else:
        parser.error("give --input <path> or --demo")

    missing = set(CATEGORY_IDS) - set(df["category"])
    if missing:
        print(f"Warning: no complaints found for {sorted(missing)}")

    train, test = split_and_save(df, args.test_size)
    print("\nComplaints per category (train / test):")
    counts = pd.DataFrame(
        {"train": train["category"].value_counts(), "test": test["category"].value_counts()}
    ).fillna(0).astype(int)
    print(counts.to_string())
    print(f"\nSaved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
