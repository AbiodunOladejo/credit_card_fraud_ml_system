"""
api/main.py
───────────
FastAPI application for credit card fraud detection.

Endpoints:
    GET  /              — health check + model info
    GET  /threshold     — BOT value with cost context
    POST /predict       — single transaction scoring
    POST /predict_batch — CSV upload, batch scoring

Every prediction is written to MySQL (non-fatal if DB is down).
"""

import io
import uuid
import logging
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.predict import predict_single, BOT, FN_COST, FP_COST
from api.database import create_tables, get_db, write_prediction

logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Production credit card fraud detection. "
        "Model: XGBoost [CW] (Tuned). "
        f"Business Optimal Threshold (BOT): {BOT}. "
        f"FN cost: ${FN_COST} | FP cost: ${FP_COST}."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """Create DB tables on startup if DATABASE_URL is set."""
    create_tables()


# ── Pydantic input model ──────────────────────────────────────────────────────

class Transaction(BaseModel):
    """
    Single transaction input. All 17 raw fields the model expects.
    is_weekend and is_international are raw dataset features passed by the caller.
    hour_of_day, day_of_week, month, quarter are temporal fields parsed from
    the transaction timestamp by the caller before sending to the API.
    """
    transaction_id: Optional[str] = Field(
        default=None,
        description="Optional caller-provided transaction ID. Auto-generated if omitted."
    )
    amount: float = Field(
        ..., gt=0, description="Transaction amount in USD (must be positive)"
    )
    merchant_category: str = Field(
        ..., min_length=1, description="Merchant category (e.g. 'Electronics', 'Grocery')"
    )
    card_type: str = Field(
        ..., min_length=1, description="Card type (e.g. 'Visa', 'Mastercard')"
    )
    device_type: str = Field(
        ..., min_length=1, description="Device used (e.g. 'Mobile', 'Desktop')"
    )
    country: str = Field(
        ..., min_length=1, description="Country of transaction (e.g. 'Nigeria', 'UK')"
    )
    user_age: int = Field(
        ..., ge=18, le=100, description="Cardholder age in years (18–100)"
    )
    account_age_days: int = Field(
        ..., ge=0, description="Days since the account was opened"
    )
    transaction_count_24h: int = Field(
        ..., ge=0, description="Number of transactions by this card in the past 24 hours"
    )
    avg_transaction_amount: float = Field(
        ..., ge=0, description="Cardholder's average historical transaction amount in USD"
    )
    distance_from_last_transaction: float = Field(
        ..., ge=0, description="Distance from last transaction in km"
    )
    merchant_reputation_score: float = Field(
        ..., ge=0, le=100, description="Merchant reputation score (0 = worst, 100 = best)"
    )
    hour_of_day: int = Field(
        ..., ge=0, le=23, description="Hour of transaction (0–23, 24-hour clock)"
    )
    day_of_week: int = Field(
        ..., ge=0, le=6, description="Day of week (0 = Monday, 6 = Sunday)"
    )
    month: int = Field(
        ..., ge=1, le=12, description="Month of transaction (1–12)"
    )
    quarter: int = Field(
        ..., ge=1, le=4, description="Quarter of transaction (1–4)"
    )
    is_weekend: int = Field(
        ..., ge=0, le=1, description="1 if the transaction is on a weekend, 0 otherwise"
    )
    is_international: int = Field(
        ..., ge=0, le=1, description="1 if the transaction is cross-border, 0 if domestic"
    )

    @field_validator("amount", "avg_transaction_amount")
    @classmethod
    def round_currency(cls, v: float) -> float:
        return round(v, 2)

    model_config = {"json_schema_extra": {
        "example": {
            "transaction_id":                 "TXN-DEMO-001",
            "amount":                         1800.00,
            "merchant_category":              "Electronics",
            "card_type":                      "Visa",
            "device_type":                    "Mobile",
            "country":                        "Russia",
            "user_age":                       34,
            "account_age_days":               120,
            "transaction_count_24h":          6,
            "avg_transaction_amount":         60.00,
            "distance_from_last_transaction": 3200.0,
            "merchant_reputation_score":      35.0,
            "hour_of_day":                    3,
            "day_of_week":                    6,
            "month":                          1,
            "quarter":                        1,
            "is_weekend":                     1,
            "is_international":               1,
        }
    }}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    """Health check and model info."""
    return {
        "status":      "ok",
        "model":       "XGBoost [CW] (Tuned)",
        "strategy":    "class_weight=balanced",
        "bot":         BOT,
        "fn_cost":     FN_COST,
        "fp_cost":     FP_COST,
        "description": (
            "Credit card fraud detection API. "
            "POST /predict for single-transaction scoring. "
            "POST /predict_batch for CSV batch scoring."
        ),
    }


@app.get("/threshold", tags=["Model Info"])
def get_threshold():
    """
    Return the Business Optimal Threshold (BOT) and cost context.

    BOT = 0.10 was derived by sweeping thresholds 0.05–0.95 and selecting
    the value that minimises total cost (FN × $752.09 + FP × $5.00).
    """
    return {
        "bot":              BOT,
        "fn_cost":          FN_COST,
        "fp_cost":          FP_COST,
        "cost_at_bot":      74635.0,
        "cost_at_0_5":      82208.0,
        "savings_vs_0_5":   7573.0,
        "savings_pct":      "9.2%",
        "tier_routing": {
            "HIGH":   f"prob >= 0.70  → AUTO-DECLINE",
            "MEDIUM": f"prob >= {BOT} and < 0.70 → ROUTE TO ANALYST",
            "LOW":    f"prob < {BOT}  → APPROVE",
        },
        "monthly_projection": {
            "transactions_per_month": 200000,
            "cost_at_bot":            373936,
            "cost_at_0_5":            411791,
            "monthly_saving_vs_0_5":  37855,
            "annual_saving_vs_0_5":   454259,
        },
    }


@app.post("/predict", tags=["Prediction"])
def predict(txn: Transaction):
    """
    Score a single transaction.

    Returns fraud probability, risk tier (HIGH / MEDIUM / LOW),
    routing decision (DECLINE / REVIEW / APPROVE), top SHAP drivers
    in plain English, engineered feature values, and cost context.

    Every prediction is written to MySQL if DATABASE_URL is set.
    The response is returned regardless of DB availability.
    """
    try:
        txn_dict = txn.model_dump()
        txn_id   = txn_dict.pop("transaction_id") or f"auto-{uuid.uuid4().hex[:12]}"

        result = predict_single(txn_dict)

        # Write to DB — non-fatal
        with get_db() as db:
            write_prediction(db, txn_dict, result, transaction_id=txn_id)

        result["transaction_id"] = txn_id
        return result

    except Exception as exc:
        logger.error(f"Prediction failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}")


@app.post("/predict_batch", tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)):
    """
    Score a batch of transactions from a CSV upload.

    The CSV must have the same columns as the /predict endpoint
    (transaction_id is optional). Use GET /predict to download
    a one-row template.

    Returns a summary (total, flagged by tier) and a results list
    with one row per input transaction.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV (.csv)")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {str(exc)}")

    # Required columns (transaction_id is optional)
    required = [
        "amount", "merchant_category", "card_type", "device_type", "country",
        "user_age", "account_age_days", "transaction_count_24h",
        "avg_transaction_amount", "distance_from_last_transaction",
        "merchant_reputation_score", "hour_of_day", "day_of_week",
        "month", "quarter", "is_weekend", "is_international",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"CSV missing required columns: {missing_cols}. "
                   f"Download a template from GET /predict.",
        )

    results     = []
    errors      = []
    tier_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        txn_id   = str(row_dict.pop("transaction_id", f"batch-{idx}"))

        try:
            result = predict_single(row_dict)

            with get_db() as db:
                write_prediction(db, row_dict, result, transaction_id=txn_id)

            tier_counts[result["risk_tier"]] += 1
            results.append({
                "transaction_id":   txn_id,
                "row":              int(idx),
                "fraud_probability": result["fraud_probability"],
                "risk_tier":        result["risk_tier"],
                "decision":         result["decision"],
                "top_reason":       (
                    result["shap_explanations"][0]["description"]
                    if result["shap_explanations"] else ""
                ),
            })

        except Exception as exc:
            errors.append({"row": int(idx), "transaction_id": txn_id, "error": str(exc)})

    flagged = tier_counts["HIGH"] + tier_counts["MEDIUM"]

    return {
        "summary": {
            "total_processed": len(results),
            "total_errors":    len(errors),
            "flagged":         flagged,
            "approved":        tier_counts["LOW"],
            "by_tier": {
                "HIGH":   tier_counts["HIGH"],
                "MEDIUM": tier_counts["MEDIUM"],
                "LOW":    tier_counts["LOW"],
            },
        },
        "results": results,
        "errors":  errors,
    }
