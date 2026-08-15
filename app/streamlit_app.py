"""
app/streamlit_app.py
────────────────────
Three-tab Streamlit analyst dashboard for the fraud detection system.

Tab 1 — Single Transaction Analyser
    Pre-loaded scenario buttons, full input form, risk profile card,
    SHAP explanation, probability gauge, engineered feature table.

Tab 2 — Batch CSV Upload
    Template download, CSV upload → batch scoring, KPI summary,
    colour-coded results table, tier pie chart, download results.

Tab 3 — Live Dashboard
    Reads from MySQL via DATABASE_URL (Hugging Face Space secret).
    Graceful fallback to demo data if DB is unavailable.
    Auto-refreshes every 30 seconds.
"""

import os
import io
import time
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000"))
DATABASE_URL = st.secrets.get("DATABASE_URL", os.getenv("DATABASE_URL", ""))

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Colour palette ────────────────────────────────────────────────────────────
COLOUR_HIGH   = "#E63946"   # red   — HIGH risk
COLOUR_MEDIUM = "#F4A261"   # amber — MEDIUM risk
COLOUR_LOW    = "#457B9D"   # blue  — LOW risk / approved
COLOUR_BG     = "#1E1E2E"   # dark background for gauges

# ── Scenario presets ─────────────────────────────────────────────────────────
SCENARIOS = {
    "🚨 Obvious Fraud": {
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
    "🟡 Borderline Case": {
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
    "✅ Legitimate Transaction": {
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

EMPTY_FORM = {
    "amount": 0.0,
    "merchant_category": "",
    "card_type": "",
    "device_type": "",
    "country": "",
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

# ── Session state init ────────────────────────────────────────────────────────
if "form_values" not in st.session_state:
    st.session_state["form_values"] = SCENARIOS["✅ Legitimate Transaction"].copy()

# ── Helper: call /predict ─────────────────────────────────────────────────────
def call_predict(payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to the API at {API_URL}. Is the FastAPI server running?")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None

# ── Helper: colour tier ───────────────────────────────────────────────────────
def tier_colour(tier: str) -> str:
    return {"HIGH": COLOUR_HIGH, "MEDIUM": COLOUR_MEDIUM, "LOW": COLOUR_LOW}.get(tier, "#888")

def tier_emoji(tier: str) -> str:
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(tier, "⚪")

# ── Probability gauge ─────────────────────────────────────────────────────────
def probability_gauge(prob: float, tier: str) -> go.Figure:
    colour = tier_colour(tier)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": colour}},
        gauge={
            "axis":      {"range": [0, 100], "tickcolor": "#888"},
            "bar":       {"color": colour},
            "bgcolor":   "#2A2A3E",
            "steps": [
                {"range": [0,  10],  "color": "#1A3A5C"},
                {"range": [10, 70],  "color": "#2E4A2E"},
                {"range": [70, 100], "color": "#4A1A1A"},
            ],
            "threshold": {
                "line":  {"color": "#FFFFFF", "width": 2},
                "thickness": 0.75,
                "value": 10,   # BOT line
            },
        },
        title={"text": "Fraud Probability", "font": {"size": 16, "color": "#CCCCCC"}},
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=60, b=20, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#CCCCCC",
    )
    return fig

# ── Demo data for Tab 3 fallback ──────────────────────────────────────────────
def _demo_predictions() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n   = 60
    tiers = rng.choice(["HIGH", "MEDIUM", "LOW"], size=n, p=[0.08, 0.12, 0.80])
    probs = np.where(
        tiers == "HIGH",   rng.uniform(0.70, 0.99, n),
        np.where(tiers == "MEDIUM", rng.uniform(0.10, 0.70, n),
                 rng.uniform(0.00, 0.10, n))
    )
    hours = rng.integers(0, 24, n)
    amounts = rng.uniform(10, 2000, n).round(2)
    countries = rng.choice(["Nigeria", "UK", "Russia", "Germany", "USA"], size=n)
    merchants = rng.choice(["Electronics", "Grocery", "Travel", "Restaurant", "Online"], size=n)
    days_back  = rng.integers(0, 30, n)
    timestamps = pd.Timestamp("today") - pd.to_timedelta(days_back, unit="D")
    return pd.DataFrame({
        "transaction_id":   [f"DEMO-{i:04d}" for i in range(n)],
        "fraud_probability": probs.round(5),
        "risk_tier":         tiers,
        "decision":          np.where(tiers == "HIGH", "DECLINE",
                             np.where(tiers == "MEDIUM", "REVIEW", "APPROVE")),
        "predicted_at":      timestamps,
        "hour_of_day":       hours,
        "amount":            amounts,
        "country":           countries,
        "merchant_category": merchants,
    })

# ── Load live data from MySQL ─────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_live_data() -> tuple[pd.DataFrame, bool]:
    """
    Returns (dataframe, is_live).
    is_live=True if we got real MySQL data, False if using demo fallback.
    """
    if not DATABASE_URL:
        return _demo_predictions(), False
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        query = text("""
            SELECT p.transaction_id, p.fraud_probability, p.risk_tier,
                   p.decision, p.predicted_at,
                   rt.hour_of_day, rt.amount, rt.country, rt.merchant_category
            FROM ml_fraud_predictions p
            LEFT JOIN raw_transactions rt ON p.transaction_id = rt.transaction_id
            ORDER BY p.predicted_at DESC
            LIMIT 500
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        if df.empty:
            return _demo_predictions(), False
        return df, True
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return _demo_predictions(), False

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

st.title("🔍 Fraud Detection Dashboard")
st.caption(
    "Model: XGBoost [CW] (Tuned) · Business Optimal Threshold: 0.10 · "
    "FN cost: $752.09 · FP cost: $5.00"
)

tab1, tab2, tab3 = st.tabs([
    "🔎 Single Transaction Analyser",
    "📂 Batch CSV Upload",
    "📊 Live Dashboard",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE TRANSACTION ANALYSER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Analyse a Single Transaction")
    st.markdown("Choose a scenario or fill in the form manually, then click **Analyse**.")

    # ── Scenario buttons ──────────────────────────────────────────────────────
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        if st.button("🚨 Obvious Fraud", use_container_width=True):
            st.session_state["form_values"] = SCENARIOS["🚨 Obvious Fraud"].copy()
    with col_s2:
        if st.button("🟡 Borderline Case", use_container_width=True):
            st.session_state["form_values"] = SCENARIOS["🟡 Borderline Case"].copy()
    with col_s3:
        if st.button("✅ Legitimate Transaction", use_container_width=True):
            st.session_state["form_values"] = SCENARIOS["✅ Legitimate Transaction"].copy()
    with col_s4:
        if st.button("✏️ Clear / Custom Values", use_container_width=True):
            st.session_state["form_values"] = EMPTY_FORM.copy()

    st.divider()
    fv = st.session_state["form_values"]

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("transaction_form"):
        st.markdown("#### Transaction Details")

        c1, c2, c3 = st.columns(3)
        with c1:
            amount = st.number_input(
                "Amount ($)", min_value=0.01, value=float(fv.get("amount", 100.0)),
                step=0.01, format="%.2f"
            )
            merchant_category = st.text_input(
                "Merchant Category", value=fv.get("merchant_category", "Grocery")
            )
            card_type = st.text_input(
                "Card Type", value=fv.get("card_type", "Visa")
            )
            device_type = st.text_input(
                "Device Type", value=fv.get("device_type", "Mobile")
            )
            country = st.text_input(
                "Country", value=fv.get("country", "Nigeria")
            )

        with c2:
            user_age = st.number_input(
                "Cardholder Age", min_value=18, max_value=100,
                value=int(fv.get("user_age", 35))
            )
            account_age_days = st.number_input(
                "Account Age (days)", min_value=0,
                value=int(fv.get("account_age_days", 365))
            )
            transaction_count_24h = st.number_input(
                "Transactions in last 24h", min_value=0,
                value=int(fv.get("transaction_count_24h", 1))
            )
            avg_transaction_amount = st.number_input(
                "Avg Transaction Amount ($)", min_value=0.0,
                value=float(fv.get("avg_transaction_amount", 100.0)),
                step=0.01, format="%.2f"
            )
            distance_from_last_transaction = st.number_input(
                "Distance from Last Transaction (km)", min_value=0.0,
                value=float(fv.get("distance_from_last_transaction", 0.0)),
                step=0.1
            )

        with c3:
            merchant_reputation_score = st.slider(
                "Merchant Reputation Score (0–100)",
                min_value=0.0, max_value=100.0,
                value=float(fv.get("merchant_reputation_score", 75.0)),
                step=0.5
            )
            hour_of_day = st.number_input(
                "Hour of Day (0–23)", min_value=0, max_value=23,
                value=int(fv.get("hour_of_day", 14))
            )
            day_of_week = st.number_input(
                "Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6,
                value=int(fv.get("day_of_week", 2))
            )
            month = st.number_input(
                "Month (1–12)", min_value=1, max_value=12,
                value=int(fv.get("month", 1))
            )
            quarter = st.number_input(
                "Quarter (1–4)", min_value=1, max_value=4,
                value=int(fv.get("quarter", 1))
            )

        st.markdown("#### Binary Flags")
        bc1, bc2 = st.columns(2)
        with bc1:
            is_weekend = st.selectbox(
                "Is Weekend?", options=[0, 1],
                format_func=lambda x: "Yes" if x else "No",
                index=int(fv.get("is_weekend", 0))
            )
        with bc2:
            is_international = st.selectbox(
                "Is International?", options=[0, 1],
                format_func=lambda x: "Yes" if x else "No",
                index=int(fv.get("is_international", 0))
            )

        submitted = st.form_submit_button("🔍 Analyse Transaction", use_container_width=True)

    # ── Results ───────────────────────────────────────────────────────────────
    if submitted:
        payload = {
            "amount":                         amount,
            "merchant_category":              merchant_category,
            "card_type":                      card_type,
            "device_type":                    device_type,
            "country":                        country,
            "user_age":                       int(user_age),
            "account_age_days":               int(account_age_days),
            "transaction_count_24h":          int(transaction_count_24h),
            "avg_transaction_amount":         avg_transaction_amount,
            "distance_from_last_transaction": distance_from_last_transaction,
            "merchant_reputation_score":      merchant_reputation_score,
            "hour_of_day":                    int(hour_of_day),
            "day_of_week":                    int(day_of_week),
            "month":                          int(month),
            "quarter":                        int(quarter),
            "is_weekend":                     int(is_weekend),
            "is_international":               int(is_international),
        }

        with st.spinner("Scoring transaction..."):
            result = call_predict(payload)

        if result:
            tier  = result["risk_tier"]
            prob  = result["fraud_probability"]
            dec   = result["decision"]
            bot   = result["threshold_used"]
            shaps = result.get("shap_explanations", [])
            eng   = result.get("engineered_features", {})
            cost  = result.get("cost_context", {})

            colour = tier_colour(tier)
            emoji  = tier_emoji(tier)

            st.divider()
            res_col, gauge_col = st.columns([3, 2])

            with res_col:
                # Risk profile card
                st.markdown(
                    f"""
                    <div style="border:2px solid {colour}; border-radius:10px; padding:20px; background:rgba(0,0,0,0.2)">
                        <h2 style="color:{colour}; margin:0">{emoji} {tier} RISK — {dec}</h2>
                        <p style="font-size:22px; margin:8px 0">
                            Fraud Probability: <strong>{prob*100:.1f}%</strong>
                        </p>
                        <p style="color:#aaa; margin:4px 0">
                            Threshold used: <strong>{bot}</strong> (Business Optimal Threshold)
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # SHAP explanations
                st.markdown("#### Why this decision was made")
                fraud_shaps = [s for s in shaps if s["direction"] == "fraud"]
                legit_shaps = [s for s in shaps if s["direction"] == "legit"]

                if fraud_shaps:
                    st.markdown("**Top fraud signals:**")
                    for i, s in enumerate(fraud_shaps[:3], 1):
                        st.markdown(
                            f"**{i}.** {s['description']} "
                            f"<span style='color:{COLOUR_HIGH}; font-size:12px'>"
                            f"(SHAP: +{s['shap_value']:.4f})</span>",
                            unsafe_allow_html=True,
                        )
                if legit_shaps:
                    st.markdown("**Factors reducing suspicion:**")
                    for s in legit_shaps[:2]:
                        st.markdown(
                            f"↓ {s['description']} "
                            f"<span style='color:{COLOUR_LOW}; font-size:12px'>"
                            f"(SHAP: {s['shap_value']:.4f})</span>",
                            unsafe_allow_html=True,
                        )

                # Cost context
                st.markdown("#### Cost Context")
                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    st.metric("If fraudulent & approved", f"${result['fn_cost']:,.2f} lost")
                with cc2:
                    st.metric("If legitimate & declined", f"${result['fp_cost']:,.2f} cost")
                with cc3:
                    st.metric("Expected loss (prob × FN)", f"${cost.get('expected_loss', 0):,.2f}")

            with gauge_col:
                st.plotly_chart(
                    probability_gauge(prob, tier),
                    use_container_width=True,
                )
                st.caption(
                    f"White line at {bot*100:.0f}% = Business Optimal Threshold (BOT). "
                    f"Above 70% = auto-decline."
                )

            # Engineered features table
            with st.expander("🔧 View Engineered Features"):
                eng_df = pd.DataFrame([{
                    "Feature":     k.replace("_", " ").title(),
                    "Value":       v,
                    "Description": {
                        "Is Night":           "1 = transaction between 22:00 and 05:00",
                        "Is High Velocity":   "1 = more than 4 transactions in 24 hours",
                        "Is Large Distance":  "1 = more than 1,000 km from last transaction",
                        "Amount To Avg Ratio":"Spend ÷ cardholder's average (higher = more anomalous)",
                        "Log Amount":         "Natural log of amount (compresses skew for model)",
                        "Account Age Years":  "Account age in years",
                        "Risk Score":         "Weighted composite of all fraud signals",
                    }.get(k.replace("_", " ").title(), "")
                } for k, v in eng.items()])
                st.dataframe(eng_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH CSV UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Batch Transaction Scoring")

    # ── Template download ─────────────────────────────────────────────────────
    st.markdown("#### Step 1 — Download the CSV template")
    template_cols = [
        "transaction_id", "amount", "merchant_category", "card_type",
        "device_type", "country", "user_age", "account_age_days",
        "transaction_count_24h", "avg_transaction_amount",
        "distance_from_last_transaction", "merchant_reputation_score",
        "hour_of_day", "day_of_week", "month", "quarter",
        "is_weekend", "is_international",
    ]
    template_row = {
        "transaction_id": "TXN-EXAMPLE-001",
        "amount": 120.50,
        "merchant_category": "Grocery",
        "card_type": "Visa",
        "device_type": "Mobile",
        "country": "Nigeria",
        "user_age": 35,
        "account_age_days": 730,
        "transaction_count_24h": 2,
        "avg_transaction_amount": 95.00,
        "distance_from_last_transaction": 5.0,
        "merchant_reputation_score": 82.0,
        "hour_of_day": 10,
        "day_of_week": 1,
        "month": 3,
        "quarter": 1,
        "is_weekend": 0,
        "is_international": 0,
    }
    template_df  = pd.DataFrame([template_row])
    template_csv = template_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download CSV Template",
        data=template_csv,
        file_name="fraud_detection_template.csv",
        mime="text/csv",
    )

    st.markdown("#### Step 2 — Upload your CSV")
    uploaded_file = st.file_uploader(
        "Upload transactions CSV", type=["csv"], label_visibility="collapsed"
    )

    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df_upload):,} rows. Sending to API...")

            with st.spinner(f"Scoring {len(df_upload):,} transactions..."):
                csv_bytes = df_upload.to_csv(index=False).encode()
                try:
                    resp = requests.post(
                        f"{API_URL}/predict_batch",
                        files={"file": ("batch.csv", csv_bytes, "text/csv")},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    batch_result = resp.json()
                except Exception as e:
                    st.error(f"Batch API call failed: {e}")
                    batch_result = None

            if batch_result:
                summary = batch_result["summary"]
                results = batch_result["results"]

                # KPI row
                st.markdown("#### Results Summary")
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Total Processed", f"{summary['total_processed']:,}")
                k2.metric("Flagged",          f"{summary['flagged']:,}")
                k3.metric("🔴 HIGH Risk",      f"{summary['by_tier']['HIGH']:,}")
                k4.metric("🟡 MEDIUM Risk",    f"{summary['by_tier']['MEDIUM']:,}")
                k5.metric("🟢 Approved",       f"{summary['by_tier']['LOW']:,}")

                results_df = pd.DataFrame(results)

                # Tier pie chart
                tier_counts_data = summary["by_tier"]
                pie_fig = go.Figure(go.Pie(
                    labels=["HIGH", "MEDIUM", "LOW"],
                    values=[
                        tier_counts_data["HIGH"],
                        tier_counts_data["MEDIUM"],
                        tier_counts_data["LOW"],
                    ],
                    marker_colors=[COLOUR_HIGH, COLOUR_MEDIUM, COLOUR_LOW],
                    hole=0.4,
                ))
                pie_fig.update_layout(
                    title="Tier Distribution",
                    height=300,
                    margin=dict(t=40, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#CCCCCC",
                )
                st.plotly_chart(pie_fig, use_container_width=True)

                # Colour-coded results table
                st.markdown("#### Scored Transactions")

                def colour_tier_cell(val):
                    c = {"HIGH": COLOUR_HIGH, "MEDIUM": COLOUR_MEDIUM, "LOW": COLOUR_LOW}.get(val, "")
                    return f"color: {c}; font-weight: bold" if c else ""

                styled = (
                    results_df.style
                    .applymap(colour_tier_cell, subset=["risk_tier"])
                    .format({"fraud_probability": "{:.2%}"})
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)

                # Download results
                results_csv = results_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Results CSV",
                    data=results_csv,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv",
                )

                if batch_result.get("errors"):
                    with st.expander(f"⚠️ {len(batch_result['errors'])} rows failed"):
                        st.dataframe(pd.DataFrame(batch_result["errors"]))

        except Exception as e:
            st.error(f"Could not read CSV: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    df_live, is_live = load_live_data()

    if is_live:
        st.success("🟢 Connected to live MySQL database. Refreshing every 30 seconds.")
    else:
        st.warning(
            "⚠️ Live database not connected. Showing demo data with the same layout. "
            "Set DATABASE_URL in your Hugging Face Space secrets to connect."
        )

    st.markdown(f"**{len(df_live):,} predictions loaded** · Last refresh: {pd.Timestamp.now().strftime('%H:%M:%S')}")

    # ── KPI row ───────────────────────────────────────────────────────────────
    total   = len(df_live)
    flagged = len(df_live[df_live["risk_tier"].isin(["HIGH", "MEDIUM"])])
    high    = len(df_live[df_live["risk_tier"] == "HIGH"])
    medium  = len(df_live[df_live["risk_tier"] == "MEDIUM"])
    low     = len(df_live[df_live["risk_tier"] == "LOW"])

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Predictions",  f"{total:,}")
    k2.metric("Fraud Flagged",       f"{flagged:,}", delta=f"{flagged/total*100:.1f}%")
    k3.metric("🔴 HIGH Risk",         f"{high:,}")
    k4.metric("🟡 MEDIUM Risk",        f"{medium:,}")
    k5.metric("🟢 Approved",           f"{low:,}")

    st.divider()

    # ── Row 1: Fraud over time + Tier breakdown ───────────────────────────────
    r1c1, r1c2 = st.columns([3, 2])

    with r1c1:
        df_live["predicted_at"] = pd.to_datetime(df_live["predicted_at"])
        df_time = (
            df_live.set_index("predicted_at")
            .resample("D")["risk_tier"]
            .apply(lambda x: (x.isin(["HIGH","MEDIUM"])).sum())
            .reset_index()
            .rename(columns={"risk_tier": "flagged_count"})
        )
        area_fig = go.Figure(go.Scatter(
            x=df_time["predicted_at"],
            y=df_time["flagged_count"],
            fill="tozeroy",
            line_color=COLOUR_HIGH,
            fillcolor="rgba(230,57,70,0.2)",
            name="Flagged",
        ))
        area_fig.update_layout(
            title="Fraud Flags Over Time",
            xaxis_title="Date",
            yaxis_title="Flagged Transactions",
            height=300,
            margin=dict(t=40, b=40, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CCCCCC",
        )
        st.plotly_chart(area_fig, use_container_width=True)

    with r1c2:
        pie2 = go.Figure(go.Pie(
            labels=["HIGH", "MEDIUM", "LOW"],
            values=[high, medium, low],
            marker_colors=[COLOUR_HIGH, COLOUR_MEDIUM, COLOUR_LOW],
            hole=0.45,
        ))
        pie2.update_layout(
            title="Tier Breakdown",
            height=300,
            margin=dict(t=40, b=0, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#CCCCCC",
        )
        st.plotly_chart(pie2, use_container_width=True)

    # ── Row 2: Fraud by merchant + Fraud by hour ──────────────────────────────
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        if "merchant_category" in df_live.columns:
            merch_df = (
                df_live[df_live["risk_tier"].isin(["HIGH","MEDIUM"])]
                .groupby("merchant_category")
                .size()
                .reset_index(name="flagged")
                .sort_values("flagged", ascending=True)
            )
            merch_fig = px.bar(
                merch_df, x="flagged", y="merchant_category",
                orientation="h",
                color_discrete_sequence=[COLOUR_MEDIUM],
                title="Fraud Flags by Merchant Category",
                labels={"flagged": "Flagged Count", "merchant_category": "Category"},
            )
            merch_fig.update_layout(
                height=320,
                margin=dict(t=40, b=40, l=10, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#CCCCCC",
            )
            st.plotly_chart(merch_fig, use_container_width=True)

    with r2c2:
        if "hour_of_day" in df_live.columns:
            hour_df = (
                df_live[df_live["risk_tier"].isin(["HIGH","MEDIUM"])]
                .groupby("hour_of_day")
                .size()
                .reindex(range(24), fill_value=0)
                .reset_index(name="flagged")
            )
            hour_df.columns = ["hour", "flagged"]
            # Night hours (22–5) highlighted in red
            hour_df["colour"] = hour_df["hour"].apply(
                lambda h: COLOUR_HIGH if (h >= 22 or h <= 5) else COLOUR_LOW
            )
            hour_fig = go.Figure(go.Bar(
                x=hour_df["hour"],
                y=hour_df["flagged"],
                marker_color=hour_df["colour"],
                name="Flagged",
            ))
            hour_fig.update_layout(
                title="Fraud Flags by Hour (red = night window 22:00–05:00)",
                xaxis_title="Hour of Day",
                yaxis_title="Flagged Count",
                height=320,
                margin=dict(t=40, b=40, l=40, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#CCCCCC",
            )
            st.plotly_chart(hour_fig, use_container_width=True)

    # ── Recent predictions table ──────────────────────────────────────────────
    st.markdown("#### Recent Predictions (last 20)")
    display_cols = [c for c in
        ["transaction_id","fraud_probability","risk_tier","decision","predicted_at","amount","country"]
        if c in df_live.columns]
    recent_df = df_live[display_cols].head(20).copy()

    def colour_row(row):
        c = tier_colour(row.get("risk_tier", ""))
        return [f"color: {c}" if col == "risk_tier" else "" for col in recent_df.columns]

    fmt = {}
    if "fraud_probability" in recent_df.columns:
        fmt["fraud_probability"] = "{:.2%}"
    if "amount" in recent_df.columns:
        fmt["amount"] = "${:,.2f}"

    st.dataframe(
        recent_df.style
            .apply(colour_row, axis=1)
            .format(fmt),
        use_container_width=True,
        hide_index=True,
    )

    # Auto-refresh notice
    st.caption("Dashboard auto-refreshes every 30 seconds via st.cache_data(ttl=30). Reload the page to force refresh.")
