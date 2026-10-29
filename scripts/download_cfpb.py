"""Download a balanced sample of CFPB complaints (with narratives) through the public search API.

The full CFPB export is several GB. This script asks the API for only what the project
needs: up to N recent complaints per category, each with a written narrative. The output
uses the same column names as the official CSV, so src/prepare_data.py reads it unchanged.

Usage:
    python scripts/download_cfpb.py --per-category 1600 --since 2024-01-01
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CATEGORY_IDS, RAW_DIR, map_product  # noqa: E402

API = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
PAGE_SIZE = 100  # the API's maximum
HEADERS = {"User-Agent": "bank-complaint-classifier (portfolio project)", "Accept": "application/json"}
OUTPUT = RAW_DIR / "cfpb_sample.csv"
COLUMNS = {  # API field -> official CSV column name
    "complaint_id": "Complaint ID",
    "date_received": "Date received",
    "product": "Product",
    "sub_product": "Sub-product",
    "issue": "Issue",
    "company": "Company",
    "complaint_what_happened": "Consumer complaint narrative",
}


def get_json(params: list[tuple[str, str]], retries: int = 4) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
                return json.load(r)
        except Exception as exc:  # network hiccups: wait and retry
            if attempt == retries - 1:
                raise
            print(f"    retrying after error: {exc}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _find_buckets(node) -> list[dict]:
    """Return the first list of aggregation buckets under `node`, wherever the API nests it."""
    if isinstance(node, dict):
        if isinstance(node.get("buckets"), list):
            return node["buckets"]
        for value in node.values():
            found = _find_buckets(value)
            if found:
                return found
    return []


def products_by_category(since: str) -> dict[str, list[str]]:
    """Ask the API which product names exist, then group them into project categories."""
    data = get_json([("size", "1"), ("has_narrative", "true"), ("date_received_min", since)])
    names = [b["key"] for b in _find_buckets(data.get("aggregations", {}).get("product", {}))]
    grouped: dict[str, list[str]] = {c: [] for c in CATEGORY_IDS}
    for name in names:
        category = map_product(name)
        if category:
            grouped[category].append(name)
    return grouped


def fetch_category(products: list[str], since: str, wanted: int) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while len(rows) < wanted:
        params = [("product", p) for p in products] + [
            ("has_narrative", "true"),
            ("date_received_min", since),
            ("sort", "created_date_desc"),
            ("size", str(PAGE_SIZE)),
            ("frm", str(offset)),
            ("no_aggs", "true"),
        ]
        hits = get_json(params).get("hits", {}).get("hits", [])
        if not hits:
            break
        for hit in hits:
            src = hit.get("_source", {})
            if src.get("complaint_what_happened"):
                rows.append({csv_col: src.get(api_field, "") for api_field, csv_col in COLUMNS.items()})
        offset += PAGE_SIZE
        time.sleep(0.5)  # be polite to a public government API
    return rows[:wanted]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-category", type=int, default=1600)
    parser.add_argument("--since", default="2024-01-01", help="Only complaints received on or after this date")
    args = parser.parse_args()

    grouped = products_by_category(args.since)
    print("CFPB products found per category:")
    for category, products in grouped.items():
        print(f"  {category}: {products}")
    missing = [c for c, p in grouped.items() if not p]
    if missing:
        sys.exit(f"No CFPB products matched {missing}; the API's product names may have changed.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS.values()))
        writer.writeheader()
        for category, products in grouped.items():
            rows = fetch_category(products, args.since, args.per_category)
            writer.writerows(rows)
            total += len(rows)
            print(f"  {category}: {len(rows)} complaints downloaded")
    print(f"Saved {total:,} complaints to {OUTPUT}")


if __name__ == "__main__":
    main()
