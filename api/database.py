"""
api/database.py
───────────────
SQLAlchemy ORM for prediction persistence.

Two tables:
    raw_transactions      — the input transaction as received by the API
    ml_fraud_predictions  — the model's output for every scored transaction

DATABASE_URL is always read from the environment variable.
If DATABASE_URL is not set, all DB operations are silently skipped —
the API continues to serve predictions without persistence.
"""

import os
import logging
from datetime import datetime
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, SmallInteger, Text, DECIMAL, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

logger = logging.getLogger(__name__)

# ── Engine setup ──────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL")   # e.g. mysql+pymysql://user:pass@host/db

_engine       = None
_SessionLocal = None
_DB_AVAILABLE = False

if DATABASE_URL:
    try:
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,       # reconnect on stale connections
            pool_recycle=3600,        # recycle connections every hour
            connect_args={
                "connect_timeout": 10,    # stop waiting after 10 seconds
                "ssl": {
                    "check_hostname": False,  # required for Aiven/PyMySQL SSL connection
                },
            },
        )
        _SessionLocal = sessionmaker(
            bind=_engine,
            autocommit=False,
            autoflush=False
        )
        _DB_AVAILABLE = True
        logger.info("Database engine created from DATABASE_URL.")
    except Exception as exc:
        logger.warning(
            f"Failed to create database engine: {exc}. "
            "DB persistence disabled."
        )
else:
    logger.warning("DATABASE_URL not set. Database persistence is disabled.")

Base = declarative_base()


# ── ORM Models ────────────────────────────────────────────────────────────────

class RawTransaction(Base):
    """
    Stores the input transaction exactly as received by the API.
    One row per /predict call, regardless of model output.
    """
    __tablename__ = "raw_transactions"

    id                             = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id                 = Column(String(64), unique=True, nullable=True)
    amount                         = Column(DECIMAL(10, 2), nullable=False)
    merchant_category              = Column(String(100), nullable=True)
    card_type                      = Column(String(50),  nullable=True)
    device_type                    = Column(String(50),  nullable=True)
    country                        = Column(String(100), nullable=True)
    user_age                       = Column(Integer,     nullable=True)
    account_age_days               = Column(Integer,     nullable=True)
    transaction_count_24h          = Column(Integer,     nullable=True)
    avg_transaction_amount         = Column(DECIMAL(10, 2), nullable=True)
    distance_from_last_transaction = Column(DECIMAL(10, 2), nullable=True)
    merchant_reputation_score      = Column(DECIMAL(5, 2),  nullable=True)
    hour_of_day                    = Column(Integer,     nullable=True)
    is_weekend                     = Column(SmallInteger, nullable=True)
    is_international               = Column(SmallInteger, nullable=True)
    created_at                     = Column(DateTime, default=datetime.utcnow)


class MlFraudPrediction(Base):
    """
    Stores the model's output for every scored transaction.
    This table drives Power BI Page 4 — Model Performance Monitoring.
    """
    __tablename__ = "ml_fraud_predictions"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id     = Column(String(64),   nullable=True, index=True)
    fraud_probability  = Column(DECIMAL(6, 5), nullable=False)
    risk_tier          = Column(String(20),   nullable=False)   # HIGH / MEDIUM / LOW
    decision           = Column(String(20),   nullable=False)   # DECLINE / REVIEW / APPROVE
    threshold_used     = Column(DECIMAL(4, 3), nullable=False)  # 0.10
    top_shap_feature_1 = Column(String(100),  nullable=True)
    top_shap_feature_2 = Column(String(100),  nullable=True)
    top_shap_feature_3 = Column(String(100),  nullable=True)
    shap_value_1       = Column(DECIMAL(8, 5), nullable=True)
    shap_value_2       = Column(DECIMAL(8, 5), nullable=True)
    shap_value_3       = Column(DECIMAL(8, 5), nullable=True)
    fn_cost            = Column(DECIMAL(8, 2), nullable=True)   # 752.09
    fp_cost            = Column(DECIMAL(5, 2), nullable=True)   # 5.00
    predicted_at       = Column(DateTime, default=datetime.utcnow, index=True)


# ── Table creation ────────────────────────────────────────────────────────────

def create_tables() -> None:
    """Create all tables if they do not already exist."""
    if not _DB_AVAILABLE:
        logger.warning("create_tables() skipped — database not available.")
        return
    try:
        Base.metadata.create_all(bind=_engine)
        logger.info("Database tables verified / created.")
    except Exception as exc:
        logger.error(f"create_tables() failed: {exc}")


# ── Session dependency ────────────────────────────────────────────────────────

@contextmanager
def get_db():
    """
    Context-manager session dependency.
    Yields None silently if DATABASE_URL is not set or the DB is unreachable.

    Usage (in FastAPI route or anywhere):
        with get_db() as db:
            if db:
                write_prediction(db, ...)
    """
    if not _DB_AVAILABLE or _SessionLocal is None:
        yield None
        return

    session: Session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error(f"DB session error: {exc}")
    finally:
        session.close()

# ── Write helpers ─────────────────────────────────────────────────────────────

def write_prediction(
    db: Session,
    transaction_input: dict,
    prediction_result: dict,
    transaction_id: str | None = None,
) -> None:
    """
    Write one raw transaction + one prediction row atomically.

    Parameters
    ----------
    db                  : active SQLAlchemy Session (from get_db())
    transaction_input   : the dict passed to predict_single()
    prediction_result   : the dict returned by predict_single()
    transaction_id      : optional caller-provided transaction ID string
    """
    if db is None:
        return  # DB unavailable — silently skip; prediction already returned to caller

    try:
        # ── raw_transactions row ──────────────────────────────────────────────
        raw = RawTransaction(
            transaction_id                 = transaction_id,
            amount                         = float(transaction_input.get("amount", 0)),
            merchant_category              = str(transaction_input.get("merchant_category", "")),
            card_type                      = str(transaction_input.get("card_type", "")),
            device_type                    = str(transaction_input.get("device_type", "")),
            country                        = str(transaction_input.get("country", "")),
            user_age                       = int(transaction_input.get("user_age", 0)),
            account_age_days               = int(transaction_input.get("account_age_days", 0)),
            transaction_count_24h          = int(transaction_input.get("transaction_count_24h", 0)),
            avg_transaction_amount         = float(transaction_input.get("avg_transaction_amount", 0)),
            distance_from_last_transaction = float(transaction_input.get("distance_from_last_transaction", 0)),
            merchant_reputation_score      = float(transaction_input.get("merchant_reputation_score", 50)),
            hour_of_day                    = int(transaction_input.get("hour_of_day", 0)),
            is_weekend                     = int(transaction_input.get("is_weekend", 0)),
            is_international               = int(transaction_input.get("is_international", 0)),
        )
        db.add(raw)

        # ── ml_fraud_predictions row ──────────────────────────────────────────
        shap_list  = prediction_result.get("shap_explanations", [])
        fraud_shap = [s for s in shap_list if s["direction"] == "fraud"]

        def _shap_feat(idx: int) -> str | None:
            return fraud_shap[idx]["feature"] if len(fraud_shap) > idx else None

        def _shap_val(idx: int) -> float | None:
            return fraud_shap[idx]["shap_value"] if len(fraud_shap) > idx else None

        pred = MlFraudPrediction(
            transaction_id    = transaction_id,
            fraud_probability = float(prediction_result["fraud_probability"]),
            risk_tier         = prediction_result["risk_tier"],
            decision          = prediction_result["decision"],
            threshold_used    = float(prediction_result["threshold_used"]),
            top_shap_feature_1 = _shap_feat(0),
            top_shap_feature_2 = _shap_feat(1),
            top_shap_feature_3 = _shap_feat(2),
            shap_value_1       = _shap_val(0),
            shap_value_2       = _shap_val(1),
            shap_value_3       = _shap_val(2),
            fn_cost            = float(prediction_result["fn_cost"]),
            fp_cost            = float(prediction_result["fp_cost"]),
        )
        db.add(pred)
        # commit handled by the get_db() context manager

    except Exception as exc:
        logger.error(f"write_prediction() failed: {exc}")
        # Do NOT re-raise — a DB write failure must never crash the prediction response
