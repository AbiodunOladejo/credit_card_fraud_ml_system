# Credit Card Fraud Detection — Production ML System

![Demo GIF](assets/demo.gif)
> *Live demo: Single transaction analysis with real-time SHAP explanation and risk routing. Recorded using `simulate_transactions.py`.*

---

## The Problem

Every year, billions of dollars are lost to credit card fraud. A bank processing **200,000 transactions per month** with no automated detection would absorb **$3,294,154 in fraud losses every month** — catching zero fraud, paying every cost in full.

The naive alternative — flag everything above a threshold — drowns analysts in false alarms and alienates legitimate customers. The real challenge is not whether to catch fraud. It is *which threshold* minimises total financial loss, not which threshold maximises a benchmark metric.

This project answers that question with a production-grade, end-to-end ML system.

---

## The Gap

Most fraud detection tutorials optimise for F1 score or accuracy. Neither of these is what a bank cares about. A missed fraud case costs **$752.09** — the full transaction value, gone. A false alarm costs **$5.00** in customer service friction. These costs are **150:1 asymmetric**.

At the default 0.5 classification threshold, the tuned model costs **$82,208 per evaluation period**. Simply by choosing the right threshold — the Business Optimal Threshold (BOT = 0.10) — that cost drops to **$74,635**: a saving of **$7,573 (9.2%)** with no retraining, no new data, no architecture changes.

A deployed system at a mid-sized bank using BOT would cost **$373,936/month** versus **$411,791/month** at the default threshold — an avoidable gap of **$37,855/month**, or **$454,259/year**.

---

## The Solution

A complete, end-to-end production-style fraud detection pipeline:

- **Automated model selection** across 10 models (5 algorithms × 2 imbalance strategies) using a composite Business Score (50% cost + 25% AUC + 15% F1 + 10% speed)
- **Optuna hyperparameter tuning** (15 trials, StratifiedKFold(3)) on the automated winner only
- **Business Optimal Threshold** derived by cost minimisation, not F1 maximisation
- **Three-tier risk routing** for analyst workflow integration
- **SHAP plain-English explanations** for every prediction
- **FastAPI backend** with MySQL persistence for monitoring and drift detection
- **Streamlit dashboard** for analyst use, batch processing, and live monitoring
- **Power BI dashboards** connecting historical data and live model predictions

---

## What Would Happen at a Real Bank?

If this model were deployed at a mid-sized bank processing 200,000 transactions per month, the numbers are concrete:

| Strategy | Fraud Caught | False Alarms/Month | Monthly Cost |
|---|---|---|---|
| No model — catch nothing | 0% | — | $3,294,154 |
| Industry default (0.5 threshold) | 87.5% | 230 | $411,791 |
| **This system — BOT (0.10)** | **88.8%** | **932** | **$373,936** |

At the Business Optimal Threshold, this model would **prevent $2,920,218 in fraud loss per month** versus no model. Compared to simply deploying the same model at the default 0.5 threshold, it saves an additional **$37,855/month — $454,259/year** — in avoidable losses. The cost of 932 false alarms ($4,660/month) is small compared to the 11 extra fraud cases caught at BOT versus 0.5 ($8,273 in recovered losses). The numbers justify the threshold choice unconditionally.

---

## Model Performance

### Pre-Tuning: All 10 Models Ranked by Business Score

The Business Score formula: **50% × (1 − normalised cost) + 25% × AUC + 15% × F1 + 10% × (1 − normalised speed)**

| Model | Train F1 | Test F1 | Precision | Recall | ROC-AUC | Total Cost | Train Time | Business Score |
|---|---|---|---|---|---|---|---|---|
| **🏆 XGBoost [CW]** | 0.9155 | 0.8612 | 0.8332 | 0.8913 | 0.9504 | $72,229 | 7.1s | **0.8538** |
| CatBoost [SMOTE] | 0.9089 | 0.9175 | 0.9612 | 0.8776 | 0.9529 | $80,629 | 19.1s | 0.8479 |
| XGBoost [SMOTE] | 0.9137 | 0.9159 | 0.9646 | 0.8719 | 0.9531 | $84,374 | 10.5s | 0.8440 |
| CatBoost [CW] | 0.7107 | 0.7032 | 0.5787 | 0.8959 | 0.9494 | $71,290 | 11.0s | 0.7919 |
| Random Forest [CW] | 0.8967 | 0.8727 | 0.8747 | 0.8707 | 0.9544 | $85,531 | 45.8s | 0.7892 |
| LightGBM [CW] | 0.9034 | 0.8656 | 0.8413 | 0.8913 | 0.9474 | $72,184 | 4.8s | 0.7860 |
| LightGBM [SMOTE] | 0.9186 | 0.9181 | 0.9695 | 0.8719 | 0.9488 | $84,354 | 5.4s | 0.7492 |
| Random Forest [SMOTE] | 0.9065 | 0.9002 | 0.9684 | 0.8410 | 0.9553 | $104,661 | 54.8s | 0.6712 |
| Logistic Regression [CW] | 0.2638 | 0.2610 | 0.1521 | 0.9165 | 0.9451 | $77,223 | 3.6s | 0.5600 |
| Logistic Regression [SMOTE] | 0.5735 | 0.5745 | 0.4531 | 0.7849 | 0.9464 | $145,533 | 6.1s | 0.1985 |

XGBoost [CW] won not because it had the highest F1 — CatBoost [SMOTE] had better F1 — but because it produced the lowest cost ($72,229) at a fast training time, and the composite Business Score rewards that combination.

### Post-Tuning: Deployed Model Performance

Optuna tuned XGBoost [CW] with 15 trials and StratifiedKFold(3), optimising for F1 (best CV F1: 0.9020). This improved F1 significantly but, because the objective was F1 not cost, raw cost at 0.5 increased. BOT analysis on the tuned model then recovered and surpassed the pre-tuning cost.

| Metric | At Default 0.5 | At BOT (0.10) |
|---|---|---|
| Precision | 0.9433 | 0.807 |
| Recall | 0.8753 | 0.888 |
| F1 | 0.908 | 0.845 |
| ROC-AUC | 0.9502 | 0.9502 |
| Fraud Caught | 87.5% (765/874) | 88.8% (776/874) |
| False Alarms | 46 | 186 |
| **Total Cost** | **$82,208** | **$74,635** |

**The deployed system uses BOT = 0.10.** This is not a quirk — it is the whole point. Threshold optimisation is the highest-leverage, lowest-cost intervention available after training.

### Why XGBoost [CW] Won Over SMOTE Models

SMOTE models (XGBoost [SMOTE], CatBoost [SMOTE]) had slightly better F1 scores but meaningfully higher costs: XGBoost [SMOTE] cost $84,374 vs $72,229 for the class-weight variant. The Business Score, which weights cost at 50%, correctly identified that a higher F1 that comes with $12,000 more in financial loss is not a better model for this use case.

---

## Three-Tier Risk Routing

Every prediction produces one of three decisions:

| Tier | Probability Range | Action | Rationale |
|---|---|---|---|
| 🔴 HIGH RISK | ≥ 0.70 | Auto-decline | Above 70% — cost of approval far exceeds $5 decline cost |
| 🟡 MEDIUM RISK | ≥ BOT (0.10) and < 0.70 | Route to analyst queue | Model suspects fraud but confidence is not absolute |
| 🟢 LOW RISK | < BOT (0.10) | Approve | Below cost-optimal threshold — expected loss is acceptable |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│   fraud_detection_dataset.csv  ←→  Jupyter Notebook (15 phases)│
└─────────────────────────┬───────────────────────────────────────┘
                          │ model.joblib
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION LAYER                             │
│                                                                 │
│   src/predict.py                                                │
│   ├── Feature engineering (matches notebook Phase 3 exactly)   │
│   ├── ColumnTransformer (15 numeric / 4 categorical / 5 binary) │
│   ├── XGBoost [CW] tuned inference                              │
│   ├── BOT threshold (0.10) applied                              │
│   ├── Three-tier routing                                        │
│   └── SHAP TreeExplainer → plain-English explanation            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API LAYER                                  │
│                                                                 │
│   api/main.py (FastAPI)                                         │
│   ├── GET  /              → health check                        │
│   ├── GET  /threshold     → BOT value + cost context            │
│   ├── POST /predict       → single transaction analysis         │
│   └── POST /predict_batch → CSV upload, batch scoring           │
│                                                                 │
│   api/database.py (SQLAlchemy ORM)                              │
│   ├── raw_transactions table                                    │
│   └── ml_fraud_predictions table                                │
└──────────┬──────────────────────────────┬───────────────────────┘
           │                              │
           ▼                              ▼
┌──────────────────────┐      ┌───────────────────────────────────┐
│    MySQL Database    │      │        FRONT-END LAYER            │
│    (Render / local)  │      │                                   │
│                      │      │  app/streamlit_app.py             │
│  raw_transactions    │      │  ├── Tab 1: Single Analyser       │
│  ml_fraud_pred...    │ ←──  │  ├── Tab 2: Batch CSV Upload      │
│                      │      │  └── Tab 3: Live Dashboard        │
│  4 analytics views   │      │      (reads MySQL, 30s refresh,   │
└──────────┬───────────┘      │       degrades gracefully)        │
           │                  └───────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ANALYTICS LAYER (Power BI)                     │
│                                                                 │
│  Page 1 — Executive Summary (KPIs + trend)                      │
│  Page 2 — Fraud Pattern Analysis (map + hour + merchant)        │
│  Page 3 — Transaction Deep Dive (scatter + high-risk table)     │
│  Page 4 — Model Performance Monitoring (drift via MySQL)        │
│                                                                 │
│  ← Historical data from fraud_clean.csv                         │
│  ← Live predictions from ml_fraud_predictions (MySQL)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Deployment Stack

| Component | Technology | Platform |
|---|---|---|
| ML Model | XGBoost + scikit-learn pipeline | Serialised as model.joblib |
| API | FastAPI + Uvicorn | Render (Docker) |
| Database | MySQL + SQLAlchemy | Render managed MySQL |
| Front-end | Streamlit | Hugging Face Spaces |
| Analytics | Power BI Desktop + Service | Power BI Service (embedded) |
| Container | Docker (python:3.10-slim) | Render |

---

## Project Structure

```
fraud-detection/
├── README.md
├── METHODOLOGY.md
├── MODEL_CARD.md
├── DATA_DICTIONARY.md
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
├── simulate_transactions.py
├── model.joblib                  ← not in repo (see .gitignore)
├── assets/
│   ├── demo.gif
│   └── model_comparison.png
├── src/
│   └── predict.py                ← feature engineering + inference + SHAP
├── api/
│   ├── main.py                   ← FastAPI endpoints
│   └── database.py               ← SQLAlchemy ORM + session management
├── config/
│   └── db_schema.sql             ← table definitions + analytics views
└── app/
    └── streamlit_app.py          ← three-tab analyst dashboard
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
```

### 2. Add your model

Copy `model.joblib` (generated by the notebook Phase 15) into the project root.

### 3. Set environment variables

```bash
cp .env.example .env
# Edit .env — add DATABASE_URL and API_URL
```

### 4. Run the API locally

```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Run the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

### 6. Record the demo GIF

```bash
python simulate_transactions.py
```

---

## Known Limitations

**Overfitting:** Train F1 (0.9155 pre-tuning, 0.9865 post-tuning) is meaningfully higher than Test F1 (0.8612 pre-tuning, 0.908 post-tuning). Tighter regularisation was tested during Optuna tuning and reduced F1 without improving cost. This is documented as a genuine finding, not hidden.

**Geographic bias:** Country is used as a model feature. Countries with structurally higher fraud rates in this training data will generate higher false positive rates for legitimate customers from those regions. A production system would require fairness auditing before deployment.

**SMOTE trade-off:** SMOTE with full sampling_strategy=1.0 produced marginally lower cost but required 40+ minutes of training. Reduced to 0.15 to cut training below 5 minutes. This trade-off is documented. The class_weight winner was unaffected by this constraint.

**Static model:** The current system has no automated retraining. Drift monitoring via Power BI Page 4 signals when performance degrades. Retraining is a human-supervised decision triggered by those signals.

---

## Power BI Dashboard

*Embedded link added after publishing to Power BI Service.*

---

## Author

**Abiodun Oladejo**  
ALX Afica Fellow  
[LinkedIn](https://linkedin.com/in/YOUR_HANDLE) · [GitHub](https://github.com/YOUR_USERNAME) · [Kaggle](https://kaggle.com/YOUR_HANDLE) · [Portfolio](https://YOUR_PORTFOLIO_URL) · [Medium](https://medium.com/@YOUR_HANDLE)
