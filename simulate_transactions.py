"""
simulate_transactions.py
────────────────────────
Sends 10 pre-defined transactions to the /predict endpoint with a 2-second
delay between each. Prints tier, probability, and decision to the terminal.

This is the GIF recording script for the GitHub README demo.

Usage:
    python simulate_transactions.py

    # Or point at a deployed API:
    API_URL=https://your-app.onrender.com python simulate_transactions.py

The 10 transactions are:
    1.  Obvious fraud      — 3AM, international, 30× avg spend, high velocity
    2.  Obvious fraud      — night, cross-border, stolen card pattern
    3.  Borderline         — slightly elevated amount, international travel
    4.  Borderline         — dusk transaction, new account, medium distance
    5.  Borderline         — above-average spend, low-reputation merchant
    6.  Legitimate         — daytime, domestic, normal spend
    7.  Legitimate         — grocery, established account
    8.  Legitimate         — small domestic coffee shop purchase
    9.  Obvious fraud      — midnight, unknown country, extreme distance
    10. Legitimate         — regular weekday transaction, known merchant
"""

import os
import sys
import time
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Colour codes for terminal output ─────────────────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

TIER_COLOURS = {
    "HIGH":   RED,
    "MEDIUM": YELLOW,
    "LOW":    GREEN,
}

# ── 10 pre-defined transactions ───────────────────────────────────────────────
TRANSACTIONS = [
    {
        "label": "Transaction 1 — Obvious Fraud (3AM international, 30× spend)",
        "payload": {
            "amount": 1850.00,
            "merchant_category": "Electronics",
            "card_type": "Visa",
            "device_type": "Mobile",
            "country": "Russia",
            "user_age": 29,
            "account_age_days": 95,
            "transaction_count_24h": 7,
            "avg_transaction_amount": 62.00,
            "distance_from_last_transaction": 3800.0,
            "merchant_reputation_score": 22.0,
            "hour_of_day": 3,
            "day_of_week": 6,
            "month": 1,
            "quarter": 1,
            "is_weekend": 1,
            "is_international": 1,
        },
    },
    {
        "label": "Transaction 2 — Obvious Fraud (night, stolen-card velocity)",
        "payload": {
            "amount": 990.00,
            "merchant_category": "Jewelry",
            "card_type": "Mastercard",
            "device_type": "Mobile",
            "country": "Ukraine",
            "user_age": 45,
            "account_age_days": 210,
            "transaction_count_24h": 8,
            "avg_transaction_amount": 85.00,
            "distance_from_last_transaction": 2100.0,
            "merchant_reputation_score": 31.0,
            "hour_of_day": 2,
            "day_of_week": 5,
            "month": 11,
            "quarter": 4,
            "is_weekend": 1,
            "is_international": 1,
        },
    },
    {
        "label": "Transaction 3 — Borderline (international travel, elevated amount)",
        "payload": {
            "amount": 480.00,
            "merchant_category": "Travel",
            "card_type": "Visa",
            "device_type": "Desktop",
            "country": "Germany",
            "user_age": 41,
            "account_age_days": 840,
            "transaction_count_24h": 3,
            "avg_transaction_amount": 195.00,
            "distance_from_last_transaction": 920.0,
            "merchant_reputation_score": 65.0,
            "hour_of_day": 20,
            "day_of_week": 4,
            "month": 7,
            "quarter": 3,
            "is_weekend": 0,
            "is_international": 1,
        },
    },
    {
        "label": "Transaction 4 — Borderline (new account, dusk, medium distance)",
        "payload": {
            "amount": 310.00,
            "merchant_category": "Online Shopping",
            "card_type": "Mastercard",
            "device_type": "Mobile",
            "country": "Ghana",
            "user_age": 27,
            "account_age_days": 65,
            "transaction_count_24h": 4,
            "avg_transaction_amount": 120.00,
            "distance_from_last_transaction": 600.0,
            "merchant_reputation_score": 55.0,
            "hour_of_day": 21,
            "day_of_week": 3,
            "month": 4,
            "quarter": 2,
            "is_weekend": 0,
            "is_international": 0,
        },
    },
    {
        "label": "Transaction 5 — Borderline (low-rep merchant, above-avg spend)",
        "payload": {
            "amount": 390.00,
            "merchant_category": "Electronics",
            "card_type": "Visa",
            "device_type": "Desktop",
            "country": "Nigeria",
            "user_age": 33,
            "account_age_days": 490,
            "transaction_count_24h": 2,
            "avg_transaction_amount": 130.00,
            "distance_from_last_transaction": 12.0,
            "merchant_reputation_score": 38.0,
            "hour_of_day": 16,
            "day_of_week": 2,
            "month": 9,
            "quarter": 3,
            "is_weekend": 0,
            "is_international": 0,
        },
    },
    {
        "label": "Transaction 6 — Legitimate (daytime, domestic, normal spend)",
        "payload": {
            "amount": 58.50,
            "merchant_category": "Grocery",
            "card_type": "Visa",
            "device_type": "Desktop",
            "country": "Nigeria",
            "user_age": 38,
            "account_age_days": 1820,
            "transaction_count_24h": 1,
            "avg_transaction_amount": 65.00,
            "distance_from_last_transaction": 1.8,
            "merchant_reputation_score": 91.0,
            "hour_of_day": 11,
            "day_of_week": 1,
            "month": 3,
            "quarter": 1,
            "is_weekend": 0,
            "is_international": 0,
        },
    },
    {
        "label": "Transaction 7 — Legitimate (established account, regular grocery)",
        "payload": {
            "amount": 42.00,
            "merchant_category": "Grocery",
            "card_type": "Mastercard",
            "device_type": "Mobile",
            "country": "UK",
            "user_age": 52,
            "account_age_days": 3200,
            "transaction_count_24h": 1,
            "avg_transaction_amount": 48.00,
            "distance_from_last_transaction": 0.5,
            "merchant_reputation_score": 94.0,
            "hour_of_day": 9,
            "day_of_week": 0,
            "month": 5,
            "quarter": 2,
            "is_weekend": 0,
            "is_international": 0,
        },
    },
    {
        "label": "Transaction 8 — Legitimate (small coffee shop, lunch hour)",
        "payload": {
            "amount": 8.75,
            "merchant_category": "Restaurant",
            "card_type": "Visa",
            "device_type": "Mobile",
            "country": "Nigeria",
            "user_age": 26,
            "account_age_days": 720,
            "transaction_count_24h": 2,
            "avg_transaction_amount": 55.00,
            "distance_from_last_transaction": 0.3,
            "merchant_reputation_score": 87.0,
            "hour_of_day": 12,
            "day_of_week": 2,
            "month": 8,
            "quarter": 3,
            "is_weekend": 0,
            "is_international": 0,
        },
    },
    {
        "label": "Transaction 9 — Obvious Fraud (midnight, extreme distance, unknown)",
        "payload": {
            "amount": 2200.00,
            "merchant_category": "Electronics",
            "card_type": "Visa",
            "device_type": "Mobile",
            "country": "North Korea",
            "user_age": 31,
            "account_age_days": 44,
            "transaction_count_24h": 9,
            "avg_transaction_amount": 75.00,
            "distance_from_last_transaction": 9800.0,
            "merchant_reputation_score": 15.0,
            "hour_of_day": 0,
            "day_of_week": 6,
            "month": 12,
            "quarter": 4,
            "is_weekend": 1,
            "is_international": 1,
        },
    },
    {
        "label": "Transaction 10 — Legitimate (regular weekday, trusted merchant)",
        "payload": {
            "amount": 120.00,
            "merchant_category": "Supermarket",
            "card_type": "Mastercard",
            "device_type": "Desktop",
            "country": "Nigeria",
            "user_age": 44,
            "account_age_days": 2100,
            "transaction_count_24h": 1,
            "avg_transaction_amount": 115.00,
            "distance_from_last_transaction": 2.2,
            "merchant_reputation_score": 89.0,
            "hour_of_day": 15,
            "day_of_week": 3,
            "month": 6,
            "quarter": 2,
            "is_weekend": 0,
            "is_international": 0,
        },
    },
]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_simulation():
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  FRAUD DETECTION SYSTEM — LIVE SIMULATION{RESET}")
    print(f"{BOLD}  API: {API_URL}{RESET}")
    print(f"{BOLD}  Model: XGBoost [CW] (Tuned) | BOT: 0.10{RESET}")
    print(f"{BOLD}{'='*65}{RESET}\n")

    # Check API is reachable
    try:
        health = requests.get(f"{API_URL}/", timeout=10)
        health.raise_for_status()
        print(f"{GREEN}✅ API is online.{RESET}\n")
    except Exception as e:
        print(f"{RED}❌ Cannot reach API at {API_URL}{RESET}")
        print(f"   Error: {e}")
        print(f"   Make sure the FastAPI server is running: uvicorn api.main:app\n")
        sys.exit(1)

    results_summary = []

    for i, txn in enumerate(TRANSACTIONS, 1):
        label   = txn["label"]
        payload = txn["payload"]

        print(f"{BOLD}[{i:02d}/10]{RESET} {label}")
        print(f"       Amount: ${payload['amount']:,.2f} | "
              f"Country: {payload['country']} | "
              f"Hour: {payload['hour_of_day']:02d}:00 | "
              f"International: {'Yes' if payload['is_international'] else 'No'}")

        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()

            tier  = result["risk_tier"]
            prob  = result["fraud_probability"]
            dec   = result["decision"]
            top_r = (
                result["shap_explanations"][0]["description"]
                if result.get("shap_explanations") else "N/A"
            )

            colour = TIER_COLOURS.get(tier, RESET)
            print(f"       {colour}{BOLD}▶ {tier} RISK — {dec}{RESET}")
            print(f"         Fraud probability : {colour}{prob*100:.1f}%{RESET}")
            print(f"         Top reason        : {top_r}")
            print()

            results_summary.append({
                "txn":  i,
                "tier": tier,
                "prob": prob,
                "dec":  dec,
            })

        except requests.exceptions.HTTPError as e:
            print(f"       {RED}API error: {e.response.status_code} — {e.response.text}{RESET}\n")
        except Exception as e:
            print(f"       {RED}Error: {e}{RESET}\n")

        if i < len(TRANSACTIONS):
            time.sleep(2)

    # Summary
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  SIMULATION COMPLETE — SUMMARY{RESET}")
    print(f"{BOLD}{'='*65}{RESET}")
    print(f"  {'#':<4} {'Tier':<10} {'Probability':<14} {'Decision'}")
    print(f"  {'-'*44}")
    for r in results_summary:
        colour = TIER_COLOURS.get(r["tier"], RESET)
        print(
            f"  {r['txn']:<4} "
            f"{colour}{r['tier']:<10}{RESET} "
            f"{r['prob']*100:>6.1f}%        "
            f"{colour}{r['dec']}{RESET}"
        )
    print()

    high   = sum(1 for r in results_summary if r["tier"] == "HIGH")
    medium = sum(1 for r in results_summary if r["tier"] == "MEDIUM")
    low    = sum(1 for r in results_summary if r["tier"] == "LOW")
    print(f"  {RED}HIGH (auto-decline) : {high}{RESET}")
    print(f"  {YELLOW}MEDIUM (review)     : {medium}{RESET}")
    print(f"  {GREEN}LOW (approved)      : {low}{RESET}")
    print(f"\n{BOLD}  To record this as a GIF:{RESET}")
    print("  Install ttyrec or asciinema, then run:")
    print("    asciinema rec demo.cast && agg demo.cast assets/demo.gif")
    print()


if __name__ == "__main__":
    run_simulation()
