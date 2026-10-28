import zipfile

import pandas as pd

from src.config import CATEGORY_IDS, map_product
from src.metrics import evaluate
from src.prepare_data import clean_text, sample_balanced

LONG = " is a complaint long enough to pass the minimum length filter of fifty characters."


def test_map_product_handles_old_and_new_names():
    assert map_product("Credit card or prepaid card") == "card"
    assert map_product("Credit card") == "card"
    assert map_product("Credit reporting, credit repair services, or other personal consumer reports") == "credit_reporting"
    assert map_product("Bank account or service") == "bank_account"
    assert map_product("Debt or credit management") is None
    assert map_product(None) is None


def test_clean_text_collapses_whitespace_and_truncates():
    assert clean_text("  a \n\n b\tc ") == "a b c"
    assert len(clean_text("x" * 10_000)) == 3000


def test_metrics_count_failed_calls_as_wrong():
    m = evaluate(["card", "mortgage"], ["card", "error"])
    assert m["accuracy"] == 0.5
    assert m["per_class"]["mortgage"]["recall"] == 0.0
    assert len(m["confusion_matrix"]) == len(CATEGORY_IDS)


def _fake_cfpb(tmp_path, as_zip):
    rows = []
    products = ["Mortgage", "Debt collection", "Credit card", "Debt or credit management"]
    for i in range(40):
        product = products[i % len(products)]
        rows.append({
            "Date received": "2024-01-01", "Product": product, "Sub-product": "", "Issue": "x",
            "Consumer complaint narrative": f"Complaint {i} about {product}{LONG}" if i % 10 else None,
            "Company": "Bank", "Complaint ID": str(i),
        })
    rows.append({**rows[1], "Consumer complaint narrative": "too short", "Complaint ID": "999"})
    csv_path = tmp_path / "complaints.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    if not as_zip:
        return csv_path
    zip_path = tmp_path / "complaints.csv.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(csv_path, "complaints.csv")
    return zip_path


def test_sample_balanced_keeps_in_scope_rows_with_narratives(tmp_path):
    for as_zip in (False, True):
        df = sample_balanced(str(_fake_cfpb(tmp_path, as_zip)), per_class=5, chunksize=7)
        assert set(df["category"]) == {"mortgage", "debt_collection", "card"}  # out-of-scope product dropped
        assert df["category"].value_counts().max() <= 5
        assert df["text"].str.len().min() >= 50  # missing and short narratives dropped
        assert df["text"].is_unique
