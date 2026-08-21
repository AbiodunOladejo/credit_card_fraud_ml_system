"""
app/streamlit_app.py
────────────────────
Recruiter-facing Streamlit dashboard for the Credit Card Fraud Detection System.

The application provides three main views:

Tab 1 — Single Transaction Analyser
    • Scenario-based demonstrations
    • Guided transaction input
    • Fraud probability
    • Business risk tier
    • Business decision
    • SHAP-based explanation
    • Expected-loss context
    • Engineered feature inspection

Tab 2 — Batch CSV Upload
    • Downloadable template
    • Batch scoring through FastAPI
    • KPI summary
    • Risk-tier distribution
    • Scored transaction table
    • Downloadable results

Tab 3 — Live Model Monitoring
    • Reads prediction history from MySQL
    • Clearly distinguishes live data from demo fallback
    • Fraud-risk KPIs
    • Risk-tier distribution
    • Fraud patterns over time
    • Recent predictions

Architecture:

    Streamlit
        ↓
    Render-hosted FastAPI
        ↓
    XGBoost Fraud Model
        ↓
    MySQL / Aiven

The Streamlit application is only the presentation layer.
The deployed FastAPI service remains the source of truth for
prediction, risk tier and business decision.
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import os
import io
import requests

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# API_URL should be supplied through Streamlit Secrets.
# The localhost fallback is intentionally kept for local development only.
API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000"
)

# DATABASE_URL should also be supplied through Streamlit Secrets.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    ""
)

# Business Optimal Threshold established during model development.
BUSINESS_OPTIMAL_THRESHOLD = 0.10

# Risk-tier boundaries used by the deployed model.
HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.10


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ══════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════════════════

# Financial-risk visual language:
#
# RED    → high risk / fraud
# AMBER  → human attention / review
# BLUE   → lower risk / approved
# TEAL   → positive business outcome

COLOUR_HIGH = "#E63946"
COLOUR_MEDIUM = "#F4A261"
COLOUR_LOW = "#457B9D"
COLOUR_TEAL = "#2A9D8F"
COLOUR_BG = "#0D1B2A"
COLOUR_PANEL = "#1B2A3B"
COLOUR_TEXT = "#E8E8E8"
COLOUR_MUTED = "#A0A8B0"


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO PRESETS
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {

    # --------------------------------------------------------------------------
    # Clearly suspicious transaction
    # --------------------------------------------------------------------------
    "🚨 High-Risk Example": {
        "amount": 1800.0,
        "merchant_category": "Electronics",
        "card_type": "Visa",
        "device_type": "Mobile",
        "country": "Russia",
        "user_age": 34,
        "account_age_days": 90,
        "transaction_count_24h": 7,
        "avg_transaction_amount": 60.0,
        "distance_from_last_transaction": 3500.0,
        "merchant_reputation_score": 28.0,
        "hour_of_day": 3,
        "day_of_week": 6,
        "month": 1,
        "quarter": 1,
        "is_weekend": 1,
        "is_international": 1,
    },

    # --------------------------------------------------------------------------
    # Deliberately ambiguous transaction.
    #
    # IMPORTANT:
    # The model still determines the actual probability.
    # This scenario is intended to produce an intermediate probability,
    # not to artificially force a 30%, 50% or 60% prediction.
    # --------------------------------------------------------------------------
    "🟡 Borderline Example": {
        "amount": 420.0,
        "merchant_category": "Travel",
        "card_type": "Mastercard",
        "device_type": "Desktop",
        "country": "Germany",
        "user_age": 42,
        "account_age_days": 730,
        "transaction_count_24h": 3,
        "avg_transaction_amount": 180.0,
        "distance_from_last_transaction": 850.0,
        "merchant_reputation_score": 62.0,
        "hour_of_day": 21,
        "day_of_week": 4,
        "month": 6,
        "quarter": 2,
        "is_weekend": 0,
        "is_international": 1,
    },

    # --------------------------------------------------------------------------
    # Clearly legitimate transaction
    # --------------------------------------------------------------------------
    "✅ Low-Risk Example": {
        "amount": 55.0,
        "merchant_category": "Grocery",
        "card_type": "Visa",
        "device_type": "Desktop",
        "country": "Nigeria",
        "user_age": 38,
        "account_age_days": 1800,
        "transaction_count_24h": 1,
        "avg_transaction_amount": 62.0,
        "distance_from_last_transaction": 2.5,
        "merchant_reputation_score": 88.0,
        "hour_of_day": 14,
        "day_of_week": 2,
        "month": 3,
        "quarter": 1,
        "is_weekend": 0,
        "is_international": 0,
    },
}


# Empty/default form values.
EMPTY_FORM = {
    "amount": 100.0,
    "merchant_category": "Grocery",
    "card_type": "Visa",
    "device_type": "Mobile",
    "country": "Nigeria",
    "user_age": 30,
    "account_age_days": 365,
    "transaction_count_24h": 1,
    "avg_transaction_amount": 100.0,
    "distance_from_last_transaction": 0.0,
    "merchant_reputation_score": 75.0,
    "hour_of_day": 12,
    "day_of_week": 1,
    "month": 1,
    "quarter": 1,
    "is_weekend": 0,
    "is_international": 0,
}


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "form_values" not in st.session_state:
    st.session_state["form_values"] = SCENARIOS["✅ Low-Risk Example"].copy()


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def tier_colour(tier: str) -> str:
    """Return the visual colour associated with a risk tier."""

    return {
        "HIGH": COLOUR_HIGH,
        "MEDIUM": COLOUR_MEDIUM,
        "LOW": COLOUR_LOW,
    }.get(tier, "#888888")


def tier_emoji(tier: str) -> str:
    """Return the visual icon associated with a risk tier."""

    return {
        "HIGH": "🔴",
        "MEDIUM": "🟡",
        "LOW": "🔵",
    }.get(tier, "⚪")


def tier_description(tier: str) -> str:
    """Return plain-English business interpretation of the risk tier."""

    return {
        "HIGH": (
            "High-risk transaction. The model estimates a substantial "
            "likelihood of fraud and the transaction should be considered "
            "for automatic decline."
        ),
        "MEDIUM": (
            "Intermediate-risk transaction. The model has identified "
            "meaningful fraud signals and the transaction may require "
            "human review."
        ),
        "LOW": (
            "Low-risk transaction. The model estimates a relatively low "
            "fraud probability and the transaction can generally proceed."
        ),
    }.get(
        tier,
        "The model returned an unrecognised risk category."
    )


def probability_category(probability: float) -> str:
    """
    Convert probability into the dashboard's explanatory probability band.

    This function is ONLY for presentation.
    The API remains the source of truth for the official risk tier.
    """

    if probability >= HIGH_RISK_THRESHOLD:
        return "HIGH"
    elif probability >= MEDIUM_RISK_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def call_predict(payload: dict) -> dict | None:
    """
    Send a transaction to the deployed FastAPI service.

    Streamlit does not perform the prediction itself.
    It sends the transaction to the production API.
    """

    try:

        response = requests.post(
            f"{API_URL.rstrip('/')}/predict",
            json=payload,
            timeout=30,
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.ConnectionError:

        st.error(
            f"""
            **Unable to connect to the fraud detection API.**

            Current API URL:

            `{API_URL}`

            The Streamlit application is running, but the prediction API
            could not be reached.
            """
        )

        return None

    except requests.exceptions.HTTPError as error:

        st.error(
            f"""
            **API returned an error:**

            `{error.response.status_code}`

            `{error.response.text}`
            """
        )

        return None

    except Exception as error:

        st.error(
            f"Unexpected prediction error: {error}"
        )

        return None


def probability_gauge(
    probability: float,
    tier: str,
) -> go.Figure:

    """
    Create the fraud probability gauge.

    The 10% Business Optimal Threshold is explicitly displayed.
    """

    colour = tier_colour(tier)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",

            value=round(probability * 100, 1),

            number={
                "suffix": "%",
                "font": {
                    "size": 36,
                    "color": colour,
                },
            },

            gauge={

                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#888",
                },

                "bar": {
                    "color": colour,
                },

                "bgcolor": "#2A2A3E",

                "steps": [

                    {
                        "range": [0, 10],
                        "color": "#1A3A5C",
                    },

                    {
                        "range": [10, 70],
                        "color": "#4A3A1A",
                    },

                    {
                        "range": [70, 100],
                        "color": "#4A1A1A",
                    },

                ],

                "threshold": {

                    "line": {
                        "color": "#FFFFFF",
                        "width": 3,
                    },

                    "thickness": 0.75,

                    "value": BUSINESS_OPTIMAL_THRESHOLD * 100,
                },
            },

            title={
                "text": "Estimated Fraud Probability",
                "font": {
                    "size": 16,
                    "color": "#CCCCCC",
                },
            },
        )
    )

    fig.update_layout(
        height=300,

        margin=dict(
            t=60,
            b=20,
            l=30,
            r=30,
        ),

        paper_bgcolor="rgba(0,0,0,0)",

        font_color="#CCCCCC",
    )

    return fig


def shap_chart(shap_explanations: list) -> go.Figure | None:

    """
    Build a horizontal SHAP contribution chart.

    Positive SHAP values push the prediction toward fraud.
    Negative SHAP values push the prediction toward legitimate.
    """

    if not shap_explanations:
        return None

    rows = []

    for explanation in shap_explanations:

        rows.append(
            {
                "Feature": explanation.get(
                    "description",
                    explanation.get("feature", "Feature"),
                ),

                "SHAP Value": float(
                    explanation.get(
                        "shap_value",
                        0,
                    )
                ),

                "Direction": explanation.get(
                    "direction",
                    "",
                ),
            }
        )

    shap_df = pd.DataFrame(rows)

    if shap_df.empty:
        return None

    # Keep the most influential signals.
    shap_df = (
        shap_df
        .sort_values(
            "SHAP Value",
            key=lambda x: x.abs(),
            ascending=True,
        )
        .tail(8)
    )

    shap_df["Colour"] = shap_df["SHAP Value"].apply(
        lambda value:
        COLOUR_HIGH if value > 0 else COLOUR_LOW
    )

    fig = go.Figure(
        go.Bar(

            x=shap_df["SHAP Value"],

            y=shap_df["Feature"],

            orientation="h",

            marker_color=shap_df["Colour"],

            hovertemplate=(
                "<b>%{y}</b><br>"
                "SHAP contribution: %{x:.4f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vline(
        x=0,
        line_width=1,
        line_color="#888888",
    )

    fig.update_layout(

        title="What influenced the model's prediction?",

        xaxis_title="SHAP contribution",

        yaxis_title="",

        height=max(
            320,
            len(shap_df) * 45,
        ),

        margin=dict(
            t=50,
            b=40,
            l=20,
            r=20,
        ),

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font_color="#CCCCCC",

    )

    return fig


def _demo_predictions() -> pd.DataFrame:

    """
    Generate clearly labelled demo data for the monitoring dashboard.

    Demo data is used only when the live MySQL database cannot be reached.
    """

    rng = np.random.default_rng(42)

    n = 60

    tiers = rng.choice(
        ["HIGH", "MEDIUM", "LOW"],
        size=n,
        p=[0.08, 0.12, 0.80],
    )

    probs = np.where(

        tiers == "HIGH",

        rng.uniform(
            0.70,
            0.99,
            n,
        ),

        np.where(

            tiers == "MEDIUM",

            rng.uniform(
                0.10,
                0.70,
                n,
            ),

            rng.uniform(
                0.00,
                0.10,
                n,
            ),
        ),
    )

    hours = rng.integers(
        0,
        24,
        n,
    )

    amounts = rng.uniform(
        10,
        2000,
        n,
    ).round(2)

    countries = rng.choice(
        [
            "Nigeria",
            "UK",
            "Russia",
            "Germany",
            "USA",
        ],
        size=n,
    )

    merchants = rng.choice(
        [
            "Electronics",
            "Grocery",
            "Travel",
            "Restaurant",
            "Online",
        ],
        size=n,
    )

    days_back = rng.integers(
        0,
        30,
        n,
    )

    timestamps = (
        pd.Timestamp("today")
        - pd.to_timedelta(
            days_back,
            unit="D",
        )
    )

    return pd.DataFrame(
        {

            "transaction_id": [
                f"DEMO-{i:04d}"
                for i in range(n)
            ],

            "fraud_probability": probs.round(5),

            "risk_tier": tiers,

            "decision": np.where(
                tiers == "HIGH",
                "DECLINE",

                np.where(
                    tiers == "MEDIUM",
                    "REVIEW",
                    "APPROVE",
                ),
            ),

            "predicted_at": timestamps,

            "hour_of_day": hours,

            "amount": amounts,

            "country": countries,

            "merchant_category": merchants,
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# LIVE DATABASE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=30)
def load_live_data() -> tuple[pd.DataFrame, bool]:

    """
    Read recent predictions from MySQL.

    Returns:
        dataframe
        boolean indicating whether the data is genuinely live
    """

    if not DATABASE_URL:

        return (
            _demo_predictions(),
            False,
        )

    try:

        from sqlalchemy import create_engine, text

        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
        )

        query = text(
            """
            SELECT
                p.transaction_id,
                p.fraud_probability,
                p.risk_tier,
                p.decision,
                p.predicted_at,
                rt.hour_of_day,
                rt.amount,
                rt.country,
                rt.merchant_category

            FROM ml_fraud_predictions p

            LEFT JOIN raw_transactions rt
                ON p.transaction_id = rt.transaction_id

            ORDER BY p.predicted_at DESC

            LIMIT 500
            """
        )

        with engine.connect() as connection:

            dataframe = pd.read_sql(
                query,
                connection,
            )

        if dataframe.empty:

            return (
                _demo_predictions(),
                False,
            )

        return (
            dataframe,
            True,
        )

    except Exception:

        return (
            _demo_predictions(),
            False,
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

st.title("🔍 Credit Card Fraud Detection System")

st.caption(
    "An end-to-end machine learning system for identifying suspicious "
    "card transactions and translating model predictions into operational "
    "risk decisions."
)


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM STATUS
# ══════════════════════════════════════════════════════════════════════════════

with st.expander(
    "⚙️ System Architecture & Deployment Status",
    expanded=False,
):

    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:

        st.markdown("**Prediction API**")

        st.code(
            API_URL,
            language="text",
        )

        if "localhost" in API_URL:

            st.warning(
                "Local development API URL detected."
            )

        else:

            st.success(
                "Production API endpoint configured."
            )

    with status_col2:

        st.markdown("**Prediction Engine**")

        st.write(
            "XGBoost fraud classification model"
        )

        st.write(
            "Business Optimal Threshold: "
            f"**{BUSINESS_OPTIMAL_THRESHOLD:.0%}**"
        )

    with status_col3:

        st.markdown("**Data Layer**")

        if DATABASE_URL:

            st.success(
                "MySQL connection configured."
            )

        else:

            st.warning(
                "MySQL connection not configured."
            )


# ══════════════════════════════════════════════════════════════════════════════
# RISK FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════

with st.expander(
    "📖 How the risk decision works",
    expanded=False,
):

    st.markdown(
        """
        The model produces a **fraud probability** for each transaction.

        The probability is then translated into an operational risk category.

        | Fraud probability | Risk tier | Typical action |
        |---|---|---|
        | Below 10% | 🔵 LOW | Approve |
        | 10% – below 70% | 🟡 MEDIUM | Review |
        | 70% or above | 🔴 HIGH | Decline |

        The **10% Business Optimal Threshold (BOT)** was selected during
        model evaluation by considering the different financial costs of
        false negatives and false positives.

        Therefore, the probability itself and the business decision are
        related but are not the same thing.
        """
    )


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(
    [
        "🔎 Transaction Analyser",
        "📂 Batch Scoring",
        "📊 Live Monitoring",
    ]
)


# ╔════════════════════════════════════════════════════════════════════════════╗
# TAB 1 — SINGLE TRANSACTION ANALYSER
# ╚════════════════════════════════════════════════════════════════════════════╝

with tab1:

    st.subheader(
        "Analyse a Single Transaction"
    )

    st.markdown(
        """
        Test how the deployed fraud detection system responds to different
        transaction profiles. Start with one of the examples below, or
        customise the transaction yourself.
        """
    )

    # --------------------------------------------------------------------------
    # Scenario buttons
    # --------------------------------------------------------------------------

    st.markdown(
        "#### Try a scenario"
    )

    scenario_col1, scenario_col2, scenario_col3, scenario_col4 = st.columns(4)

    with scenario_col1:

        if st.button(
            "🚨 High-Risk Example",
            use_container_width=True,
        ):

            st.session_state["form_values"] = (
                SCENARIOS["🚨 High-Risk Example"].copy()
            )

            st.rerun()

    with scenario_col2:

        if st.button(
            "🟡 Borderline Example",
            use_container_width=True,
        ):

            st.session_state["form_values"] = (
                SCENARIOS["🟡 Borderline Example"].copy()
            )

            st.rerun()

    with scenario_col3:

        if st.button(
            "🔵 Low-Risk Example",
            use_container_width=True,
        ):

            st.session_state["form_values"] = (
                SCENARIOS["✅ Low-Risk Example"].copy()
            )

            st.rerun()

    with scenario_col4:

        if st.button(
            "✏️ Custom Transaction",
            use_container_width=True,
        ):

            st.session_state["form_values"] = (
                EMPTY_FORM.copy()
            )

            st.rerun()


    st.info(
        """
        **Tip:** You do not need to know what values to enter manually.
        The three example scenarios are provided specifically so recruiters
        can explore the system without understanding the dataset first.
        """
    )


    st.divider()


    # --------------------------------------------------------------------------
    # Input form
    # --------------------------------------------------------------------------

    fv = st.session_state["form_values"]

    with st.form(
        "transaction_form"
    ):

        st.markdown(
            "#### Transaction Details"
        )

        c1, c2, c3 = st.columns(3)

        # ----------------------------------------------------------------------
        # Column 1
        # ----------------------------------------------------------------------

        with c1:

            amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.01,
                value=float(
                    fv.get(
                        "amount",
                        100.0,
                    )
                ),
                step=0.01,
                format="%.2f",
                help=(
                    "Value of the transaction."
                ),
            )

            merchant_category = st.text_input(
                "Merchant Category",
                value=fv.get(
                    "merchant_category",
                    "Grocery",
                ),
                help=(
                    "Category of the merchant."
                ),
            )

            card_type = st.text_input(
                "Card Type",
                value=fv.get(
                    "card_type",
                    "Visa",
                ),
            )

            device_type = st.text_input(
                "Device Type",
                value=fv.get(
                    "device_type",
                    "Mobile",
                ),
            )

            country = st.text_input(
                "Country",
                value=fv.get(
                    "country",
                    "Nigeria",
                ),
            )


        # ----------------------------------------------------------------------
        # Column 2
        # ----------------------------------------------------------------------

        with c2:

            user_age = st.number_input(
                "Cardholder Age",
                min_value=18,
                max_value=100,
                value=int(
                    fv.get(
                        "user_age",
                        35,
                    )
                ),
            )

            account_age_days = st.number_input(
                "Account Age (days)",
                min_value=0,
                value=int(
                    fv.get(
                        "account_age_days",
                        365,
                    )
                ),
            )

            transaction_count_24h = st.number_input(
                "Transactions in Last 24 Hours",
                min_value=0,
                value=int(
                    fv.get(
                        "transaction_count_24h",
                        1,
                    )
                ),
                help=(
                    "Number of transactions associated with the account "
                    "during the previous 24 hours."
                ),
            )

            avg_transaction_amount = st.number_input(
                "Average Transaction Amount ($)",
                min_value=0.0,
                value=float(
                    fv.get(
                        "avg_transaction_amount",
                        100.0,
                    )
                ),
                step=0.01,
                format="%.2f",
            )

            distance_from_last_transaction = st.number_input(
                "Distance from Last Transaction (km)",
                min_value=0.0,
                value=float(
                    fv.get(
                        "distance_from_last_transaction",
                        0.0,
                    )
                ),
                step=0.1,
            )


        # ----------------------------------------------------------------------
        # Column 3
        # ----------------------------------------------------------------------

        with c3:

            merchant_reputation_score = st.slider(
                "Merchant Reputation Score",
                min_value=0.0,
                max_value=100.0,
                value=float(
                    fv.get(
                        "merchant_reputation_score",
                        75.0,
                    )
                ),
                step=0.5,
                help=(
                    "Higher values represent stronger merchant reputation."
                ),
            )

            hour_of_day = st.number_input(
                "Hour of Day",
                min_value=0,
                max_value=23,
                value=int(
                    fv.get(
                        "hour_of_day",
                        14,
                    )
                ),
            )

            day_of_week = st.number_input(
                "Day of Week (0 = Mon, 6 = Sun)",
                min_value=0,
                max_value=6,
                value=int(
                    fv.get(
                        "day_of_week",
                        2,
                    )
                ),
            )

            month = st.number_input(
                "Month",
                min_value=1,
                max_value=12,
                value=int(
                    fv.get(
                        "month",
                        1,
                    )
                ),
            )

            quarter = st.number_input(
                "Quarter",
                min_value=1,
                max_value=4,
                value=int(
                    fv.get(
                        "quarter",
                        1,
                    )
                ),
            )


        # ----------------------------------------------------------------------
        # Binary flags
        # ----------------------------------------------------------------------

        st.markdown(
            "#### Transaction Context"
        )

        bc1, bc2 = st.columns(2)

        with bc1:

            is_weekend = st.selectbox(
                "Is Weekend?",
                options=[0, 1],
                format_func=lambda x:
                    "Yes" if x else "No",
                index=int(
                    fv.get(
                        "is_weekend",
                        0,
                    )
                ),
            )

        with bc2:

            is_international = st.selectbox(
                "Is International?",
                options=[0, 1],
                format_func=lambda x:
                    "Yes" if x else "No",
                index=int(
                    fv.get(
                        "is_international",
                        0,
                    )
                ),
            )


        submitted = st.form_submit_button(
            "🔍 Analyse Transaction",
            use_container_width=True,
        )


    # ══════════════════════════════════════════════════════════════════════════
    # PREDICTION RESULT
    # ══════════════════════════════════════════════════════════════════════════

    if submitted:

        payload = {

            "amount": amount,

            "merchant_category": merchant_category,

            "card_type": card_type,

            "device_type": device_type,

            "country": country,

            "user_age": int(user_age),

            "account_age_days": int(account_age_days),

            "transaction_count_24h": int(
                transaction_count_24h
            ),

            "avg_transaction_amount":
                avg_transaction_amount,

            "distance_from_last_transaction":
                distance_from_last_transaction,

            "merchant_reputation_score":
                merchant_reputation_score,

            "hour_of_day":
                int(hour_of_day),

            "day_of_week":
                int(day_of_week),

            "month":
                int(month),

            "quarter":
                int(quarter),

            "is_weekend":
                int(is_weekend),

            "is_international":
                int(is_international),
        }


        with st.spinner(
            "Sending transaction to the deployed fraud detection API..."
        ):

            result = call_predict(
                payload
            )


        if result:

            # ------------------------------------------------------------------
            # Extract API response
            # ------------------------------------------------------------------

            tier = result.get(
                "risk_tier",
                "UNKNOWN",
            )

            probability = float(
                result.get(
                    "fraud_probability",
                    0,
                )
            )

            decision = result.get(
                "decision",
                "UNKNOWN",
            )

            threshold_used = float(
                result.get(
                    "threshold_used",
                    BUSINESS_OPTIMAL_THRESHOLD,
                )
            )

            shap_explanations = result.get(
                "shap_explanations",
                [],
            )

            engineered_features = result.get(
                "engineered_features",
                {},
            )

            cost_context = result.get(
                "cost_context",
                {},
            )


            colour = tier_colour(
                tier
            )

            emoji = tier_emoji(
                tier
            )


            # ------------------------------------------------------------------
            # Probability / decision consistency information
            # ------------------------------------------------------------------

            probability_band = probability_category(
                probability
            )

            st.divider()

            st.markdown(
                "### Model Result"
            )

            result_col1, result_col2 = st.columns(
                [3, 2]
            )


            # ------------------------------------------------------------------
            # Result card
            # ------------------------------------------------------------------

            with result_col1:

                st.markdown(
                    f"""
                    <div style="
                        border:2px solid {colour};
                        border-radius:12px;
                        padding:24px;
                        background:rgba(0,0,0,0.18);
                    ">

                        <h2 style="
                            color:{colour};
                            margin-bottom:8px;
                        ">
                            {emoji} {tier} RISK
                        </h2>

                        <p style="
                            font-size:22px;
                            margin:6px 0;
                        ">
                            Estimated fraud probability:
                            <strong>
                                {probability * 100:.2f}%
                            </strong>
                        </p>

                        <p style="
                            font-size:18px;
                            margin:6px 0;
                        ">
                            Business decision:
                            <strong>
                                {decision}
                            </strong>
                        </p>

                        <p style="
                            color:#AAAAAA;
                            margin-top:12px;
                        ">
                            Business Optimal Threshold:
                            <strong>
                                {threshold_used * 100:.0f}%
                            </strong>
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


                # --------------------------------------------------------------
                # Business interpretation
                # --------------------------------------------------------------

                st.markdown(
                    "#### What this means"
                )

                st.write(
                    tier_description(
                        tier
                    )
                )


                # --------------------------------------------------------------
                # Probability vs decision explanation
                # --------------------------------------------------------------

                if (
                    tier != probability_band
                    and probability_band != "UNKNOWN"
                ):

                    st.warning(
                        f"""
                        **Probability/tier difference detected**

                        The model returned a probability of
                        **{probability:.2%}**, which falls within the
                        **{probability_band} probability band**, while the API
                        returned the risk tier **{tier}**.

                        This is worth checking in the prediction API because
                        Streamlit is intentionally displaying the API's
                        official classification rather than changing it.
                        """
                    )


                # --------------------------------------------------------------
                # Business decision explanation
                # --------------------------------------------------------------

                st.markdown(
                    "#### Why the threshold matters"
                )

                st.write(
                    f"""
                    The system uses a **{threshold_used:.0%} Business Optimal
                    Threshold** rather than simply using 50%.

                    This means a transaction does not need to have a 50%
                    predicted probability of fraud before the system begins
                    treating it as operationally important.
                    """
                )


            # ------------------------------------------------------------------
            # Gauge
            # ------------------------------------------------------------------

            with result_col2:

                st.plotly_chart(
                    probability_gauge(
                        probability,
                        tier,
                    ),
                    use_container_width=True,
                )

                st.caption(
                    f"""
                    White marker = {threshold_used * 100:.0f}% Business
                    Optimal Threshold.
                    """
                )


            # ══════════════════════════════════════════════════════════════════
            # SHAP EXPLANATION
            # ══════════════════════════════════════════════════════════════════

            st.divider()

            st.markdown(
                "### 🧠 Why did the model make this prediction?"
            )

            st.markdown(
                """
                SHAP explains how individual features influenced this
                particular prediction.

                **Red bars push the prediction toward fraud.**
                **Blue bars push the prediction toward legitimate.**

                A SHAP value is not another fraud probability. It represents
                the contribution of a feature to this specific prediction.
                """
            )


            shap_fig = shap_chart(
                shap_explanations
            )


            if shap_fig is not None:

                st.plotly_chart(
                    shap_fig,
                    use_container_width=True,
                )


                st.caption(
                    "SHAP values show relative contribution to this prediction; "
                    "they should not be interpreted as percentages."
                )

            else:

                st.info(
                    "No SHAP explanation was returned by the API for this prediction."
                )


            # ══════════════════════════════════════════════════════════════════
            # COST CONTEXT
            # ══════════════════════════════════════════════════════════════════

            st.markdown(
                "### 💰 Business Cost Context"
            )

            cost1, cost2, cost3 = st.columns(3)


            with cost1:

                st.metric(
                    "Potential Fraud Loss",
                    f"${result.get('fn_cost', 0):,.2f}",
                )

                st.caption(
                    "Estimated cost if a fraudulent transaction is approved."
                )


            with cost2:

                st.metric(
                    "False Alarm Cost",
                    f"${result.get('fp_cost', 0):,.2f}",
                )

                st.caption(
                    "Estimated operational cost of incorrectly declining a legitimate transaction."
                )


            with cost3:

                expected_loss = cost_context.get(
                    "expected_loss",
                    0,
                )

                st.metric(
                    "Expected Loss",
                    f"${expected_loss:,.2f}",
                )

                st.caption(
                    "Probability-weighted expected loss for this transaction."
                )


            # ══════════════════════════════════════════════════════════════════
            # ENGINEERED FEATURES
            # ══════════════════════════════════════════════════════════════════

            with st.expander(
                "🔧 View engineered features used by the model"
            ):

                if engineered_features:

                    engineered_rows = []

                    for key, value in engineered_features.items():

                        description = {

                            "Is Night":
                                "1 = transaction occurred between 22:00 and 05:00.",

                            "Is High Velocity":
                                "1 = more than four transactions occurred in 24 hours.",

                            "Is Large Distance":
                                "1 = transaction occurred more than 1,000 km from the previous transaction.",

                            "Amount To Avg Ratio":
                                "Transaction amount divided by the cardholder's average transaction amount.",

                            "Log Amount":
                                "Log-transformed transaction amount used to reduce skew.",

                            "Account Age Years":
                                "Account age expressed in years.",

                            "Risk Score":
                                "Composite score representing multiple fraud-risk signals.",

                        }.get(
                            key.replace(
                                "_",
                                " "
                            ).title(),
                            "",
                        )


                        engineered_rows.append(
                            {
                                "Feature":
                                    key.replace(
                                        "_",
                                        " "
                                    ).title(),

                                "Value":
                                    value,

                                "Business Interpretation":
                                    description,
                            }
                        )


                    engineered_df = pd.DataFrame(
                        engineered_rows
                    )


                    st.dataframe(
                        engineered_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    st.info(
                        "No engineered features were returned by the API."
                    )


# ╔════════════════════════════════════════════════════════════════════════════╗
# TAB 2 — BATCH CSV SCORING
# ╚════════════════════════════════════════════════════════════════════════════╝

with tab2:

    st.subheader(
        "Batch Transaction Scoring"
    )

    st.markdown(
        """
        Upload multiple transactions and send them through the same deployed
        fraud detection API used by the single-transaction analyser.
        """
    )


    # --------------------------------------------------------------------------
    # CSV template
    # --------------------------------------------------------------------------

    st.markdown(
        "#### Step 1 — Download the CSV template"
    )


    template_columns = [

        "transaction_id",

        "amount",

        "merchant_category",

        "card_type",

        "device_type",

        "country",

        "user_age",

        "account_age_days",

        "transaction_count_24h",

        "avg_transaction_amount",

        "distance_from_last_transaction",

        "merchant_reputation_score",

        "hour_of_day",

        "day_of_week",

        "month",

        "quarter",

        "is_weekend",

        "is_international",
    ]


    template_row = {

        "transaction_id":
            "TXN-EXAMPLE-001",

        "amount":
            120.50,

        "merchant_category":
            "Grocery",

        "card_type":
            "Visa",

        "device_type":
            "Mobile",

        "country":
            "Nigeria",

        "user_age":
            35,

        "account_age_days":
            730,

        "transaction_count_24h":
            2,

        "avg_transaction_amount":
            95.00,

        "distance_from_last_transaction":
            5.0,

        "merchant_reputation_score":
            82.0,

        "hour_of_day":
            10,

        "day_of_week":
            1,

        "month":
            3,

        "quarter":
            1,

        "is_weekend":
            0,

        "is_international":
            0,
    }


    template_df = pd.DataFrame(
        [template_row],
        columns=template_columns,
    )


    template_csv = template_df.to_csv(
        index=False
    )


    st.download_button(
        label="⬇️ Download CSV Template",

        data=template_csv,

        file_name="fraud_detection_template.csv",

        mime="text/csv",
    )


    # --------------------------------------------------------------------------
    # Upload
    # --------------------------------------------------------------------------

    st.markdown(
        "#### Step 2 — Upload transactions"
    )


    uploaded_file = st.file_uploader(
        "Upload transactions CSV",
        type=["csv"],
        label_visibility="collapsed",
    )


    if uploaded_file:

        try:

            dataframe_upload = pd.read_csv(
                uploaded_file
            )


            st.success(
                f"Loaded **{len(dataframe_upload):,} transactions**."
            )


            with st.expander(
                "Preview uploaded data"
            ):

                st.dataframe(
                    dataframe_upload.head(10),
                    use_container_width=True,
                    hide_index=True,
                )


            with st.spinner(
                f"Scoring {len(dataframe_upload):,} transactions..."
            ):

                csv_bytes = (
                    dataframe_upload
                    .to_csv(
                        index=False
                    )
                    .encode()
                )


                response = requests.post(

                    f"{API_URL.rstrip('/')}/predict_batch",

                    files={
                        "file": (
                            "batch.csv",
                            csv_bytes,
                            "text/csv",
                        )
                    },

                    timeout=120,
                )


                response.raise_for_status()

                batch_result = response.json()


            if batch_result:

                summary = batch_result.get(
                    "summary",
                    {},
                )

                results = batch_result.get(
                    "results",
                    [],
                )


                # --------------------------------------------------------------
                # KPI summary
                # --------------------------------------------------------------

                st.markdown(
                    "### Results Summary"
                )


                k1, k2, k3, k4, k5 = st.columns(5)


                total_processed = summary.get(
                    "total_processed",
                    0,
                )

                flagged = summary.get(
                    "flagged",
                    0,
                )

                by_tier = summary.get(
                    "by_tier",
                    {
                        "HIGH": 0,
                        "MEDIUM": 0,
                        "LOW": 0,
                    },
                )


                k1.metric(
                    "Transactions",
                    f"{total_processed:,}",
                )


                k2.metric(
                    "Flagged",
                    f"{flagged:,}",
                )


                k3.metric(
                    "🔴 High Risk",
                    f"{by_tier.get('HIGH', 0):,}",
                )


                k4.metric(
                    "🟡 Review",
                    f"{by_tier.get('MEDIUM', 0):,}",
                )


                k5.metric(
                    "🔵 Approved",
                    f"{by_tier.get('LOW', 0):,}",
                )


                # --------------------------------------------------------------
                # Tier distribution
                # --------------------------------------------------------------

                if total_processed > 0:

                    tier_fig = go.Figure(
                        go.Pie(

                            labels=[
                                "HIGH",
                                "MEDIUM",
                                "LOW",
                            ],

                            values=[

                                by_tier.get(
                                    "HIGH",
                                    0,
                                ),

                                by_tier.get(
                                    "MEDIUM",
                                    0,
                                ),

                                by_tier.get(
                                    "LOW",
                                    0,
                                ),
                            ],

                            marker_colors=[

                                COLOUR_HIGH,

                                COLOUR_MEDIUM,

                                COLOUR_LOW,
                            ],

                            hole=0.45,
                        )
                    )


                    tier_fig.update_layout(

                        title="Operational Risk Distribution",

                        height=320,

                        margin=dict(
                            t=50,
                            b=0,
                            l=0,
                            r=0,
                        ),

                        paper_bgcolor="rgba(0,0,0,0)",

                        font_color="#CCCCCC",
                    )


                    st.plotly_chart(
                        tier_fig,
                        use_container_width=True,
                    )


                # --------------------------------------------------------------
                # Results table
                # --------------------------------------------------------------

                st.markdown(
                    "### Scored Transactions"
                )


                results_df = pd.DataFrame(
                    results
                )


                if not results_df.empty:

                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True,
                    )


                    results_csv = (
                        results_df
                        .to_csv(
                            index=False
                        )
                    )


                    st.download_button(

                        label="⬇️ Download Results CSV",

                        data=results_csv,

                        file_name=(
                            "fraud_detection_results.csv"
                        ),

                        mime="text/csv",
                    )


                # --------------------------------------------------------------
                # Errors
                # --------------------------------------------------------------

                errors = batch_result.get(
                    "errors",
                    [],
                )


                if errors:

                    with st.expander(
                        f"⚠️ {len(errors)} rows could not be processed"
                    ):

                        st.dataframe(
                            pd.DataFrame(errors),
                            use_container_width=True,
                            hide_index=True,
                        )


        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the deployed FastAPI service."
            )


        except requests.exceptions.HTTPError as error:

            st.error(
                f"""
                Batch API error:

                `{error.response.status_code}`

                `{error.response.text}`
                """
            )


        except Exception as error:

            st.error(
                f"Could not process the CSV: {error}"
            )


# ╔════════════════════════════════════════════════════════════════════════════╗
# TAB 3 — LIVE MODEL MONITORING
# ╚════════════════════════════════════════════════════════════════════════════╝

with tab3:

    st.subheader(
        "Live Model Monitoring"
    )

    st.markdown(
        """
        This view monitors predictions stored by the deployed fraud detection
        system. It is designed to show how the machine learning model can
        become part of an ongoing operational workflow rather than simply
        producing one-off predictions.
        """
    )


    dataframe_live, is_live = load_live_data()


    # --------------------------------------------------------------------------
    # Live / demo status
    # --------------------------------------------------------------------------

    if is_live:

        st.success(
            "🟢 Connected to the live MySQL prediction database."
        )

    else:

        st.warning(
            """
            ⚠️ The live MySQL database is currently unavailable.
            The dashboard is showing clearly labelled demo data so the
            interface remains usable.
            """
        )


    total = len(
        dataframe_live
    )


    flagged = len(
        dataframe_live[
            dataframe_live["risk_tier"].isin(
                [
                    "HIGH",
                    "MEDIUM",
                ]
            )
        ]
    )


    high = len(
        dataframe_live[
            dataframe_live["risk_tier"] == "HIGH"
        ]
    )


    medium = len(
        dataframe_live[
            dataframe_live["risk_tier"] == "MEDIUM"
        ]
    )


    low = len(
        dataframe_live[
            dataframe_live["risk_tier"] == "LOW"
        ]
    )


    # --------------------------------------------------------------------------
    # KPI row
    # --------------------------------------------------------------------------

    k1, k2, k3, k4, k5 = st.columns(5)


    k1.metric(
        "Total Predictions",
        f"{total:,}",
    )


    flagged_percentage = (
        flagged / total * 100
        if total > 0
        else 0
    )


    k2.metric(
        "Fraud Flagged",
        f"{flagged:,}",
        delta=f"{flagged_percentage:.1f}%",
    )


    k3.metric(
        "🔴 High Risk",
        f"{high:,}",
    )


    k4.metric(
        "🟡 Review",
        f"{medium:,}",
    )


    k5.metric(
        "🔵 Approved",
        f"{low:,}",
    )


    st.divider()


    # ══════════════════════════════════════════════════════════════════════════
    # FRAUD OVER TIME
    # ══════════════════════════════════════════════════════════════════════════

    st.markdown(
        "### Where is the operational risk appearing?"
    )


    row1_col1, row1_col2 = st.columns(
        [3, 2]
    )


    with row1_col1:

        if (
            "predicted_at" in dataframe_live.columns
            and not dataframe_live.empty
        ):

            dataframe_live["predicted_at"] = (
                pd.to_datetime(
                    dataframe_live["predicted_at"],
                    errors="coerce",
                )
            )


            dataframe_time = (

                dataframe_live

                .dropna(
                    subset=[
                        "predicted_at"
                    ]
                )

                .set_index(
                    "predicted_at"
                )

                .resample(
                    "D"
                )["risk_tier"]

                .apply(
                    lambda values:
                    values.isin(
                        [
                            "HIGH",
                            "MEDIUM",
                        ]
                    ).sum()
                )

                .reset_index()

                .rename(
                    columns={
                        "risk_tier":
                            "flagged_count"
                    }
                )
            )


            if not dataframe_time.empty:

                area_fig = go.Figure(

                    go.Scatter(

                        x=dataframe_time[
                            "predicted_at"
                        ],

                        y=dataframe_time[
                            "flagged_count"
                        ],

                        fill="tozeroy",

                        line_color=COLOUR_HIGH,

                        fillcolor=(
                            "rgba(230,57,70,0.20)"
                        ),

                        name="Flagged",
                    )
                )


                area_fig.update_layout(

                    title=(
                        "Fraud Flags Over Time"
                    ),

                    xaxis_title="Date",

                    yaxis_title=(
                        "Flagged Transactions"
                    ),

                    height=320,

                    margin=dict(
                        t=50,
                        b=40,
                        l=40,
                        r=20,
                    ),

                    paper_bgcolor=(
                        "rgba(0,0,0,0)"
                    ),

                    plot_bgcolor=(
                        "rgba(0,0,0,0)"
                    ),

                    font_color="#CCCCCC",
                )


                st.plotly_chart(
                    area_fig,
                    use_container_width=True,
                )


    # ══════════════════════════════════════════════════════════════════════════
    # RISK DISTRIBUTION
    # ══════════════════════════════════════════════════════════════════════════

    with row1_col2:

        pie_fig = go.Figure(

            go.Pie(

                labels=[
                    "HIGH",
                    "MEDIUM",
                    "LOW",
                ],

                values=[
                    high,
                    medium,
                    low,
                ],

                marker_colors=[
                    COLOUR_HIGH,
                    COLOUR_MEDIUM,
                    COLOUR_LOW,
                ],

                hole=0.45,
            )
        )


        pie_fig.update_layout(

            title=(
                "Operational Risk Distribution"
            ),

            height=320,

            margin=dict(
                t=50,
                b=0,
                l=0,
                r=0,
            ),

            paper_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            font_color="#CCCCCC",
        )


        st.plotly_chart(
            pie_fig,
            use_container_width=True,
        )


    # ══════════════════════════════════════════════════════════════════════════
    # FRAUD BY MERCHANT AND HOUR
    # ══════════════════════════════════════════════════════════════════════════

    row2_col1, row2_col2 = st.columns(2)


    with row2_col1:

        if (
            "merchant_category"
            in dataframe_live.columns
        ):

            merchant_df = (

                dataframe_live[

                    dataframe_live[
                        "risk_tier"
                    ].isin(
                        [
                            "HIGH",
                            "MEDIUM",
                        ]
                    )

                ]

                .groupby(
                    "merchant_category"
                )

                .size()

                .reset_index(
                    name="flagged"
                )

                .sort_values(
                    "flagged",
                    ascending=True,
                )
            )


            if not merchant_df.empty:

                merchant_fig = px.bar(

                    merchant_df,

                    x="flagged",

                    y="merchant_category",

                    orientation="h",

                    color_discrete_sequence=[
                        COLOUR_MEDIUM
                    ],

                    title=(
                        "Flagged Transactions by Merchant Category"
                    ),

                    labels={
                        "flagged":
                            "Flagged Transactions",

                        "merchant_category":
                            "Merchant Category",
                    },
                )


                merchant_fig.update_layout(

                    height=320,

                    margin=dict(
                        t=50,
                        b=40,
                        l=10,
                        r=20,
                    ),

                    paper_bgcolor=(
                        "rgba(0,0,0,0)"
                    ),

                    plot_bgcolor=(
                        "rgba(0,0,0,0)"
                    ),

                    font_color="#CCCCCC",
                )


                st.plotly_chart(
                    merchant_fig,
                    use_container_width=True,
                )


    with row2_col2:

        if (
            "hour_of_day"
            in dataframe_live.columns
        ):

            hour_df = (

                dataframe_live[

                    dataframe_live[
                        "risk_tier"
                    ].isin(
                        [
                            "HIGH",
                            "MEDIUM",
                        ]
                    )

                ]

                .groupby(
                    "hour_of_day"
                )

                .size()

                .reindex(
                    range(24),
                    fill_value=0,
                )

                .reset_index(
                    name="flagged"
                )
            )


            hour_df.columns = [
                "hour",
                "flagged",
            ]


            hour_df["colour"] = (
                hour_df["hour"]
                .apply(
                    lambda hour:
                    COLOUR_HIGH
                    if (
                        hour >= 22
                        or hour <= 5
                    )
                    else COLOUR_LOW
                )
            )


            hour_fig = go.Figure(

                go.Bar(

                    x=hour_df[
                        "hour"
                    ],

                    y=hour_df[
                        "flagged"
                    ],

                    marker_color=hour_df[
                        "colour"
                    ],
                )
            )


            hour_fig.update_layout(

                title=(
                    "Flagged Transactions by Hour"
                ),

                xaxis_title=(
                    "Hour of Day"
                ),

                yaxis_title=(
                    "Flagged Transactions"
                ),

                height=320,

                margin=dict(
                    t=50,
                    b=40,
                    l=40,
                    r=20,
                ),

                paper_bgcolor=(
                    "rgba(0,0,0,0)"
                ),

                plot_bgcolor=(
                    "rgba(0,0,0,0)"
                ),

                font_color="#CCCCCC",
            )


            st.plotly_chart(
                hour_fig,
                use_container_width=True,
            )


    # ══════════════════════════════════════════════════════════════════════════
    # RECENT PREDICTIONS
    # ══════════════════════════════════════════════════════════════════════════

    st.markdown(
        "### Recent Predictions"
    )


    display_columns = [

        column

        for column in [

            "transaction_id",

            "fraud_probability",

            "risk_tier",

            "decision",

            "predicted_at",

            "amount",

            "country",

        ]

        if column
        in dataframe_live.columns
    ]


    recent_df = (
        dataframe_live[
            display_columns
        ]
        .head(20)
        .copy()
    )


    if (
        "fraud_probability"
        in recent_df.columns
    ):

        recent_df[
            "fraud_probability"
        ] = recent_df[
            "fraud_probability"
        ].apply(
            lambda value:
            f"{value:.2%}"
        )


    if (
        "amount"
        in recent_df.columns
    ):

        recent_df[
            "amount"
        ] = recent_df[
            "amount"
        ].apply(
            lambda value:
            f"${value:,.2f}"
        )


    st.dataframe(
        recent_df,
        use_container_width=True,
        hide_index=True,
    )


    if is_live:

        st.caption(
            """
            Live predictions are read from the MySQL prediction store.
            Reload the application to retrieve the latest database state.
            """
        )

    else:

        st.caption(
            """
            Demo data is being displayed because the live database could not
            be reached. Demo records are prefixed with DEMO- and should not
            be interpreted as actual production predictions.
            """
        )


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.divider()

st.caption(
    """
    Credit Card Fraud Detection System · XGBoost · FastAPI · MySQL · Streamlit · PowerBI
    """
)