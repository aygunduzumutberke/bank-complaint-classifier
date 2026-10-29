"""Project-wide settings: paths, the category scheme and how CFPB products map onto it."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # API key and model name live in .env, which git ignores
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_PATH = DATA_DIR / "sample" / "demo_complaints.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"

RANDOM_SEED = 42

# Long narratives are cut before they reach either model, so both see the same text
# and LLM token usage stays predictable.
MAX_CHARS = 3000

# Gemini settings come from the environment (.env), never from code.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", "8"))

# The seven classes the project predicts. "team" is the business routing target:
# where a bank would send the complaint once it is classified.
CATEGORIES: dict[str, dict[str, str]] = {
    "bank_account": {
        "label_en": "Bank account",
        "label_tr": "Mevduat / Vadesiz hesap",
        "team": "Retail Banking Operations",
        "description": "Checking or savings accounts: opening, closing, deposits, "
        "withdrawals, account fees, frozen accounts, overdrafts.",
    },
    "card": {
        "label_en": "Credit & prepaid card",
        "label_tr": "Kredi kartı / Ön ödemeli kart",
        "team": "Card Services",
        "description": "Credit cards and prepaid cards: billing disputes, card fees, "
        "interest, rewards, chargebacks, unauthorized card transactions, card limits.",
    },
    "mortgage": {
        "label_en": "Mortgage",
        "label_tr": "Konut kredisi",
        "team": "Mortgage Servicing",
        "description": "Home loans: applications, escrow, payments to the servicer, "
        "loan modification, refinancing, foreclosure.",
    },
    "consumer_loan": {
        "label_en": "Consumer loan",
        "label_tr": "İhtiyaç / Taşıt / Öğrenim kredisi",
        "team": "Consumer Lending",
        "description": "Personal, payday, title, vehicle and student loans and leases: "
        "loan terms, repayments, payoff amounts, repossession.",
    },
    "money_transfer": {
        "label_en": "Money transfer",
        "label_tr": "Para transferi (Havale/EFT)",
        "team": "Payments & Transfers",
        "description": "Sending or receiving money: wire transfers, P2P apps, "
        "international remittances, money orders, virtual currency, transfer scams.",
    },
    "debt_collection": {
        "label_en": "Debt collection",
        "label_tr": "Borç tahsilatı / İcra",
        "team": "Collections",
        "description": "Attempts to collect a debt: collection calls and letters, debt "
        "the consumer does not recognise, threats, lawsuits, validation of the debt.",
    },
    "credit_reporting": {
        "label_en": "Credit reporting",
        "label_tr": "Kredi notu / Kredi sicili",
        "team": "Credit Bureau Relations",
        "description": "Credit reports and scores: incorrect information on a report, "
        "disputes with credit bureaus, identity theft entries, credit monitoring.",
    },
}

CATEGORY_IDS: list[str] = list(CATEGORIES)

# CFPB has renamed products over the years, so mapping by prefix keeps old and new
# names together (e.g. "Credit card" and "Credit card or prepaid card").
# Products with no clear banking owner ("Debt or credit management", "Other financial
# service") are deliberately left out.
PRODUCT_PREFIXES: list[tuple[str, str]] = [
    ("checking or savings account", "bank_account"),
    ("bank account or service", "bank_account"),
    ("credit card", "card"),
    ("prepaid card", "card"),
    ("mortgage", "mortgage"),
    ("vehicle loan", "consumer_loan"),
    ("payday loan", "consumer_loan"),
    ("student loan", "consumer_loan"),
    ("consumer loan", "consumer_loan"),
    ("money transfer", "money_transfer"),
    ("virtual currency", "money_transfer"),
    ("debt collection", "debt_collection"),
    ("credit reporting", "credit_reporting"),
]


def map_product(product: object) -> str | None:
    """Return the project category for a CFPB product name, or None if it is out of scope."""
    if not isinstance(product, str):
        return None
    name = product.strip().lower()
    for prefix, category in PRODUCT_PREFIXES:
        if name.startswith(prefix):
            return category
    return None
