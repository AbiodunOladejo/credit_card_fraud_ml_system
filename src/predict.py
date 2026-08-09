"""
src/predict.py
──────────────
Core prediction module. Loaded once at import; all inference calls go through
predict_single().

Pipeline mirror of the notebook (Phases 3 and 5):
  • Feature engineering  — identical to Phase 3
  • ColumnTransformer    — loaded from model.joblib (fitted on training data)
  • XGBoost [CW] tuned   — loaded from model.joblib
  • BOT = 0.10           — loaded from model.joblib
  • Three-tier routing   — HIGH / MEDIUM / LOW
  • SHAP TreeExplainer   — plain-English explanation per prediction

Cost constants (from notebook Phase 5):
  FN_COST = $752.09   ← average fraud transaction value; missing one fraud = full loss
  FP_COST = $5.00     ← customer service friction per false alarm

model.joblib keys (from Phase 15):
  pipeline        — full sklearn Pipeline (preprocessor + clf)
  model_name      — "XGBoost [CW] (Tuned)"
  strategy        — "class_weight"
  bot             — 0.10
  fn_cost         — 752.09
  fp_cost         — 5.0
  features        — ordered list of 24 input feature names
  metrics_at_05   — dict of tuned metrics at default threshold
  metrics_at_bot  — dict of tuned metrics at BOT
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap

# ── Load deployment package once at import ────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.joblib")

_PKG = joblib.load(_MODEL_PATH)

PIPELINE     = _PKG["pipeline"]   # Full sklearn Pipeline (preprocessor + clf)
BOT          = float(_PKG["bot"]) # Business Optimal Threshold = 0.10
FN_COST      = float(_PKG["fn_cost"])  # 752.09
FP_COST      = float(_PKG["fp_cost"])  # 5.0
ALL_FEATURES = _PKG["features"]   # 24 input feature names in ColumnTransformer order

# ── Extract preprocessor and classifier from pipeline ─────────────────────────
_PREPROCESSOR = PIPELINE.named_steps["pre"]
_CLASSIFIER   = PIPELINE.named_steps["clf"]

# ── Build expanded feature names after OHE (for SHAP labels) ─────────────────
_OHE_COLS = _PREPROCESSOR.named_transformers_["cat"].get_feature_names_out(
    ["merchant_category", "card_type", "device_type", "country"]
)
FEATURE_NAMES = (
    [
        "amount", "user_age", "account_age_days", "transaction_count_24h",
        "avg_transaction_amount", "distance_from_last_transaction",
        "merchant_reputation_score", "log_amount", "amount_to_avg_ratio",
        "risk_score", "account_age_years", "hour_of_day", "day_of_week",
        "month", "quarter",
    ]
    + list(_OHE_COLS)
    + ["is_weekend", "is_international", "is_night", "is_high_velocity", "is_large_distance"]
)

# ── Initialise SHAP TreeExplainer once ────────────────────────────────────────
_EXPLAINER = shap.TreeExplainer(_CLASSIFIER)


# ── Feature engineering (mirrors notebook Phase 3 exactly) ───────────────────

def _engineer_features(row: dict) -> dict:
    """
    Apply the same seven feature engineering steps as Phase 3 of the notebook.

    is_weekend and is_international are raw input fields (present in the
    original dataset, passed directly by the API caller — not derived here).

    Derives:
        is_night, amount_to_avg_ratio, is_high_velocity, is_large_distance,
        log_amount, account_age_years, risk_score
    """
    r = dict(row)  # copy — do not mutate caller's dict

    # is_night: fraud attacks cluster in off-hours (22:00–05:00)
    hour = int(r.get("hour_of_day", 0))
    r["is_night"] = int(hour >= 22 or hour <= 5)

    # amount_to_avg_ratio: how anomalous is this spend vs cardholder history
    amount  = float(r.get("amount", 0))
    avg_amt = float(r.get("avg_transaction_amount", 0))
    r["amount_to_avg_ratio"] = amount / (avg_amt + 1)

    # is_high_velocity: >4 transactions in 24 hours (95th-percentile threshold)
    r["is_high_velocity"] = int(float(r.get("transaction_count_24h", 0)) > 4)

    # is_large_distance: >1,000 km from last transaction (impossible travel)
    r["is_large_distance"] = int(float(r.get("distance_from_last_transaction", 0)) > 1000)

    # log_amount: compress right-skewed amounts for StandardScaler
    r["log_amount"] = float(np.log1p(amount))

    # account_age_years: more interpretable than raw days
    r["account_age_years"] = float(r.get("account_age_days", 0)) / 365.0

    # risk_score: weighted composite of fraud signals (Phase 3 formula verbatim)
    #   is_international × 2      strongest categorical signal
    #   (1 − reputation/100) × 3  inverted merchant reputation
    #   log1p(amount_to_avg_ratio) log-compressed personal-spending anomaly
    #   is_night × 1              off-hours flag
    #   is_large_distance × 2     geographic anomaly
    rep    = float(r.get("merchant_reputation_score", 50))
    is_intl = int(r.get("is_international", 0))
    r["risk_score"] = (
        is_intl * 2
        + (1 - rep / 100) * 3
        + float(np.log1p(r["amount_to_avg_ratio"]))
        + r["is_night"]
        + r["is_large_distance"] * 2
    )

    return r


def _build_dataframe(engineered: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame with ALL_FEATURES in the exact column order
    the ColumnTransformer was fitted on. Missing fields default to 0.
    """
    row_data = {f: engineered.get(f, 0) for f in ALL_FEATURES}
    return pd.DataFrame([row_data])


# ── Plain-English SHAP explanations ──────────────────────────────────────────

def _plain_english(shap_row: np.ndarray,
                   preprocessed_row: np.ndarray,
                   engineered: dict) -> list:
    """
    Return top fraud-driving and legitimacy-supporting features
    as a list of plain-English dicts.

    Each dict: { feature, shap_value, description, direction }
    """
    contrib = pd.DataFrame({
        "feature":    FEATURE_NAMES,
        "shap_value": shap_row,
        "raw_value":  preprocessed_row,
    }).sort_values("shap_value", key=abs, ascending=False)

    top_fraud = contrib[contrib["shap_value"] > 0].head(3)
    top_legit = contrib[contrib["shap_value"] < 0].head(2)

    explanations = []

    for _, r in top_fraud.iterrows():
        fname = r["feature"]
        sv    = float(r["shap_value"])
        raw   = float(r["raw_value"])

        if fname == "risk_score":
            actual = round(engineered.get("risk_score", raw), 2)
            desc = f"Composite risk score is {actual} — high combined suspicion across multiple signals"
        elif fname == "amount_to_avg_ratio":
            ratio = round(engineered.get("amount_to_avg_ratio", raw), 1)
            desc = f"Spending {ratio}× above this cardholder's average transaction amount"
        elif fname == "is_night":
            desc = "Transaction occurred during night-time (22:00–05:00) — reduced oversight window"
        elif fname == "is_international":
            desc = "Cross-border transaction — international cards carry 2× the average fraud rate"
        elif fname == "is_large_distance":
            desc = "Geographic distance from last transaction exceeds 1,000 km — possible card cloning"
        elif fname == "is_high_velocity":
            desc = "High transaction velocity — multiple transactions within 24 hours (stolen card pattern)"
        elif fname == "log_amount":
            actual = round(float(engineered.get("amount", 0)), 2)
            desc = f"Transaction amount of ${actual:,.2f} is unusually large for this account"
        elif fname == "merchant_reputation_score":
            actual = round(float(engineered.get("merchant_reputation_score", raw)), 0)
            desc = f"Low merchant reputation score ({actual}/100) — elevated risk merchant"
        elif fname == "is_weekend":
            desc = "Weekend transaction — fraud monitoring coverage is reduced on weekends"
        elif fname == "account_age_years":
            yrs = round(engineered.get("account_age_years", raw), 1)
            desc = f"New account ({yrs} years old) — newer accounts carry higher fraud risk"
        elif fname == "amount_to_avg_ratio" or fname.startswith("amount"):
            desc = f"Unusually high spend relative to account history (value: {raw:.3f})"
        else:
            desc = f"Feature pushed fraud probability higher (value: {raw:.3f})"

        explanations.append({
            "feature":     fname,
            "shap_value":  round(sv, 4),
            "description": desc,
            "direction":   "fraud",
        })

    for _, r in top_legit.iterrows():
        fname = r["feature"]
        sv    = float(r["shap_value"])
        raw   = float(r["raw_value"])
        explanations.append({
            "feature":     fname,
            "shap_value":  round(sv, 4),
            "description": f"{fname.replace('_', ' ').title()} suggests lower fraud risk (value: {raw:.3f})",
            "direction":   "legit",
        })

    return explanations


# ── Tier routing ──────────────────────────────────────────────────────────────

def _assign_tier(prob: float) -> tuple:
    """
    Three-tier routing matching the project specification:
        prob >= 0.70  → HIGH RISK   — Auto-decline
        prob >= BOT   → MEDIUM RISK — Route to analyst queue
        prob < BOT    → LOW RISK    — Approve
    """
    if prob >= 0.70:
        return "HIGH",   "DECLINE"
    elif prob >= BOT:
        return "MEDIUM", "REVIEW"
    else:
        return "LOW",    "APPROVE"


# ── Public interface ──────────────────────────────────────────────────────────

def predict_single(transaction: dict) -> dict:
    """
    Score a single transaction end-to-end.

    Parameters
    ----------
    transaction : dict
        Raw transaction fields. Expected keys:
            amount, merchant_category, card_type, device_type, country,
            user_age, account_age_days, transaction_count_24h,
            avg_transaction_amount, distance_from_last_transaction,
            merchant_reputation_score, hour_of_day, day_of_week,
            month, quarter, is_weekend, is_international

    Returns
    -------
    dict:
        fraud_probability   : float  (0.0 – 1.0)
        risk_tier           : str    ("HIGH" | "MEDIUM" | "LOW")
        decision            : str    ("DECLINE" | "REVIEW" | "APPROVE")
        threshold_used      : float  (BOT = 0.10)
        fn_cost             : float  (752.09)
        fp_cost             : float  (5.0)
        engineered_features : dict   (7 engineered values for UI display)
        shap_explanations   : list   (plain-English feature contributions)
        cost_context        : dict   (expected cost figures for UI display)
    """
    # Step 1 — feature engineering (derives 7 new features)
    engineered = _engineer_features(transaction)

    # Step 2 — build single-row DataFrame in ColumnTransformer column order
    df_input = _build_dataframe(engineered)

    # Step 3 — predict probability through full pipeline
    prob = float(PIPELINE.predict_proba(df_input)[0, 1])

    # Step 4 — tier routing
    tier, decision = _assign_tier(prob)

    # Step 5 — SHAP (run preprocessor separately to get the transformed matrix)
    X_pre = _PREPROCESSOR.transform(df_input)
    shap_vals = _EXPLAINER.shap_values(X_pre)
    # XGBoost TreeExplainer may return a list [class0, class1] — take class 1
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_row = np.array(shap_vals).flatten()

    # Step 6 — plain-English explanations
    explanations = _plain_english(shap_row, X_pre.flatten(), engineered)

    # Step 7 — cost context for UI display
    cost_context = {
        "fn_cost_message":  f"Approving this transaction risks ${FN_COST:.2f} if it is fraud",
        "fp_cost_message":  f"Declining costs ${FP_COST:.2f} if this transaction is legitimate",
        "expected_loss":    round(prob * FN_COST, 2),
    }

    # Step 8 — engineered feature summary for UI
    engineered_summary = {
        "is_night":           int(engineered["is_night"]),
        "is_high_velocity":   int(engineered["is_high_velocity"]),
        "is_large_distance":  int(engineered["is_large_distance"]),
        "amount_to_avg_ratio": round(float(engineered["amount_to_avg_ratio"]), 4),
        "log_amount":          round(float(engineered["log_amount"]), 4),
        "account_age_years":   round(float(engineered["account_age_years"]), 2),
        "risk_score":          round(float(engineered["risk_score"]), 4),
    }

    return {
        "fraud_probability":   round(prob, 6),
        "risk_tier":           tier,
        "decision":            decision,
        "threshold_used":      BOT,
        "fn_cost":             FN_COST,
        "fp_cost":             FP_COST,
        "engineered_features": engineered_summary,
        "shap_explanations":   explanations,
        "cost_context":        cost_context,
    }
