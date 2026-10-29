"""Download CFPB complaints WITH narratives from a public Hugging Face mirror.

Why a mirror: on 14 Aug 2026 the CFPB stopped publishing complaint narratives, and its API
no longer returns the narrative text. The dataset `Mouwiya/cfpb-consumer-complaints`
(16.6M complaints, Jan 2025 - Jan 2026, Parquet) is a public copy made before that.

The mirror is split into 67 Parquet shards of ~15 MB. Rows are not ordered by product, so a
few shards give enough complaints for every category. Output uses the official CFPB column
names, so src/prepare_data.py reads it unchanged.

Usage:
    python scripts/download_cfpb.py --shards 4
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RAW_DIR, map_product  # noqa: E402

DATASET = "Mouwiya/cfpb-consumer-complaints"
SHARD_URL = f"https://huggingface.co/api/datasets/{DATASET}/parquet/default/train/{{i}}.parquet"
OUTPUT = RAW_DIR / "cfpb_sample.csv"
COLUMNS = {  # mirror column -> official CFPB CSV column
    "complaint_id": "Complaint ID",
    "date_received": "Date received",
    "product": "Product",
    "sub_product": "Sub-product",
    "issue": "Issue",
    "company": "Company",
    "consumer_complaint_narrative": "Consumer complaint narrative",
}


def download(url: str, path: Path, retries: int = 4) -> None:
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "bank-complaint-classifier"})
            with urllib.request.urlopen(request, timeout=300) as response, path.open("wb") as f:
                while chunk := response.read(1 << 20):
                    f.write(chunk)
            return
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"    retrying after error: {exc}")
            time.sleep(5 * (attempt + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", type=int, default=4, help="How many ~15 MB shards to download")
    parser.add_argument("--first-shard", type=int, default=0)
    args = parser.parse_args()

    shard_dir = RAW_DIR / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(args.first_shard, args.first_shard + args.shards):
        path = shard_dir / f"{i}.parquet"
        if not path.exists():  # re-runs skip shards already on disk
            print(f"Downloading shard {i} ...")
            download(SHARD_URL.format(i=i), path)
        df = pd.read_parquet(path, columns=list(COLUMNS))
        df = df[df["consumer_complaint_narrative"].fillna("").str.len() > 0]
        print(f"  shard {i}: {len(df):,} complaints with a narrative")
        frames.append(df)

    data = pd.concat(frames, ignore_index=True).rename(columns=COLUMNS)
    data["category"] = data["Product"].map(map_product)
    print("\nComplaints with a narrative per category:")
    print(data["category"].value_counts(dropna=False).to_string())
    data.drop(columns="category").to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(data):,} complaints to {OUTPUT}")


if __name__ == "__main__":
    main()
