"""Generate a small SYNTHETIC complaint set so the pipeline can run without the CFPB download.

These sentences are invented from templates. They are only for smoke-testing the code;
scores measured on them say nothing about real-world accuracy.

Usage:
    python scripts/make_demo_data.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import SAMPLE_PATH  # noqa: E402

PER_CLASS = 40
COMPANIES = ["Northwind Bank", "Contoso Financial", "Fabrikam Credit", "Woodgrove Bank", "Tailspin Lending"]
OPENINGS = [
    "I am writing to complain about {company}.",
    "On XX/XX/2024 I contacted {company} about a problem.",
    "I have been a customer of {company} for years and this is unacceptable.",
    "{company} has ignored my requests for weeks.",
]
CLOSINGS = [
    "I want this resolved as soon as possible.",
    "Nobody at the call center could help me.",
    "I have already filed a dispute and received no answer.",
    "Please make them fix this and refund me.",
]
PROBLEMS = {
    "bank_account": [
        "My checking account was frozen and I cannot access my deposit.",
        "They charged me an overdraft fee even though my savings account had enough money.",
        "I tried to close my savings account but they keep charging a monthly maintenance fee.",
        "A deposit I made at the branch never showed up in my checking account.",
        "My debit card withdrawals were blocked and the account was closed without notice.",
    ],
    "card": [
        "There are charges on my credit card statement that I never made.",
        "The credit card company raised my interest rate without warning.",
        "I disputed a purchase on my card but the chargeback was denied.",
        "My prepaid card balance disappeared after I reloaded it.",
        "They never applied the rewards points promised when I opened the credit card.",
    ],
    "mortgage": [
        "My mortgage servicer misapplied my monthly payment and reported me late.",
        "The escrow account for my home loan was miscalculated and my payment doubled.",
        "I applied for a loan modification on my mortgage and they started foreclosure anyway.",
        "They lost my refinance documents three times.",
        "The servicer is charging property inspection fees on my house that I never agreed to.",
    ],
    "consumer_loan": [
        "The payday loan charged far more interest than the contract said.",
        "My car was repossessed even though I made the auto loan payment on time.",
        "My student loan servicer put me in forbearance without asking me.",
        "The personal loan payoff amount they gave me was wrong.",
        "The title loan company keeps adding fees to my balance.",
    ],
    "money_transfer": [
        "I sent an international wire transfer and the money never arrived.",
        "A payment app sent my money to the wrong person and refuses to reverse it.",
        "I was scammed through a peer-to-peer transfer and the company will not help.",
        "My remittance to my family abroad has been on hold for two weeks.",
        "The money order I bought was never delivered to the recipient.",
    ],
    "debt_collection": [
        "A collection agency keeps calling me about a debt that is not mine.",
        "The debt collector threatened to sue me and called my employer.",
        "I asked the collector to validate the debt and they never sent proof.",
        "They are trying to collect a medical debt I already paid.",
        "A collector calls me several times a day, even late at night.",
    ],
    "credit_reporting": [
        "My credit report shows an account that I never opened.",
        "The credit bureau did not remove incorrect late payments after my dispute.",
        "Someone stole my identity and the fraudulent accounts are still on my credit report.",
        "My credit score dropped because the bureau mixed my file with another person.",
        "The bureau verified wrong information without investigating my dispute.",
    ],
}
PRODUCT_NAMES = {
    "bank_account": "Checking or savings account",
    "card": "Credit card",
    "mortgage": "Mortgage",
    "consumer_loan": "Payday loan, title loan, personal loan, or advance loan",
    "money_transfer": "Money transfer, virtual currency, or money service",
    "debt_collection": "Debt collection",
    "credit_reporting": "Credit reporting or other personal consumer reports",
}


def main() -> None:
    rng = random.Random(7)
    rows = []
    for category, problems in PROBLEMS.items():
        for _ in range(PER_CLASS):
            company = rng.choice(COMPANIES)
            text = " ".join(
                [
                    rng.choice(OPENINGS).format(company=company),
                    *rng.sample(problems, k=2),
                    rng.choice(CLOSINGS),
                ]
            )
            rows.append(
                {
                    "complaint_id": f"DEMO-{len(rows) + 1:04d}",
                    "date_received": "2024-01-01",
                    "product": PRODUCT_NAMES[category],
                    "issue": "synthetic",
                    "company": company,
                    "text": text,
                    "category": category,
                }
            )
    df = pd.DataFrame(rows).drop_duplicates(subset="text")
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAMPLE_PATH, index=False)
    print(f"Wrote {len(df)} synthetic complaints to {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
