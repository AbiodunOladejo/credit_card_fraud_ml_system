# Data Dictionary — Credit Card Fraud Detection

This document covers every data field used in this project: the original raw features, the data quality issues found, how each issue was resolved, the engineered features added, and the MySQL database schema used for prediction storage and monitoring.

---

## Raw Dataset — Original Features

The source dataset is `fraud_detection_dataset.csv`. It contains one row per transaction.

| Column | Data Type | Description | Range / Values |
|---|---|---|---|
| `transaction_id` | string | Unique identifier for the transaction | Alphanumeric |
| `user_id` | string | Unique identifier for the cardholder | Alphanumeric |
| `amount` | mixed (string/float) | Transaction value in USD | $0 to $2,749 (before capping) |
| `merchant` | string | Merchant name | 66 unique values |
| `merchant_category` | string | Category of the merchant | Multiple categories (mixed case in raw data) |
| `card_type` | string | Payment card type | Multiple types (whitespace issues in raw data) |
| `device_type` | string | Device used for transaction | Multiple types |
| `country` | string | Country where transaction occurred | Multiple countries |
| `user_age` | float | Age of the cardholder in years | 18–100 (error values of 999 observed) |
| `account_age_days` | float | How many days since the account was opened | 0–3,650 (negatives observed) |
| `transaction_count_24h` | float | Number of transactions by this card in the last 24 hours | 1–10+ |
| `avg_transaction_amount` | float | Cardholder's average transaction amount historically | Positive float |
| `distance_from_last_transaction` | float | Geographic distance from last transaction in km | 0–5,000+ |
| `merchant_reputation_score` | float | Third-party merchant trust score | 0–100 |
| `transaction_time` | string | Time of transaction | Mixed formats (see cleaning log) |
| `transaction_date` | string | Date of transaction | With bogus timestamp artifact appended |
| `ip_address` | string | IP address of the device | IPv4 format (malformed values observed) |
| `card_last_4` | string | Last 4 digits of the card | Excluded from model |
| `is_fraud` | integer | Ground truth label | 0 = Legitimate, 1 = Fraud |

**Class distribution:** Approximately 97.8% legitimate (0), 2.2% fraud (1) — a 44:1 imbalance.

---

## Data Quality Issues and Cleaning Log

Every issue was identified during Phase 1 (data audit) and resolved in Phase 2 (cleaning). The cleaning log below matches the exact operations performed in the notebook, in order.

| Step | Column | Issue Found | Resolution |
|---|---|---|---|
| [1] | All columns | Exact duplicate rows | Removed via `drop_duplicates()` |
| [2] | `transaction_id` | Duplicate transaction IDs (same ID appearing on multiple rows) | Kept first occurrence per ID; duplicates removed |
| [3] | `amount` | Mixed dtype — some values stored as strings, not floats. Non-positive amounts (≤ 0). Extreme outliers | Coerced to float. Removed rows where amount ≤ 0. Winsorised at 99th percentile to preserve extreme-but-valid transactions |
| [4] | `card_type` | Leading/trailing whitespace. Blank strings present | Stripped whitespace. Blank strings replaced with NaN |
| [5] | `merchant_category` | Mixed case (e.g., 'electronics' vs 'Electronics') | Standardised to Title Case via `.str.title()` |
| [6] | `user_age` | Impossible values — ages below 18 and above 100 (data entry errors, values like 999 observed) | Impossible ages set to NaN, then median imputed |
| [7] | `account_age_days` | Negative values (impossible — account cannot have negative age). Values above 10 years (3,650 days) | Negatives set to NaN then median imputed. Values above 3,650 capped at 3,650 |
| [8] | `transaction_time` | Mixed formats across the dataset — three formats found: `HH:MM`, `HH:MM:SS`, and `10h33m29s` | Custom parser extracted hour integer from all three formats. Parsed to `hour_of_day`. Remaining NaN filled with median |
| [9] | `transaction_date` | A bogus timestamp artifact `'10:58:51.700727'` was appended to every date — a dataset generation artefact, not a real time | Stripped to first 10 characters (date only). Extracted `day_of_week`, `month`, `quarter` |
| [10] | `ip_address` | Malformed IP addresses (incorrect format, non-numeric octets) | Malformed values set to NaN. IP excluded from model features entirely — fraudsters use VPNs, making IP unreliable |
| [11] | `avg_transaction_amount`, `distance_from_last_transaction`, `merchant_reputation_score`, `transaction_count_24h` | Remaining missing values after above steps | Median imputation |
| [11] | `merchant_category`, `card_type`, `country`, `device_type`, `merchant` | Remaining missing values | Mode imputation |
| [12] | `transaction_id`, `user_id`, `card_last_4`, `ip_address`, `transaction_date`, `transaction_time` | Not needed for modelling (identifiers, excluded by design) | Dropped from model dataset. `merchant` retained in cleaned CSV for Power BI analytics use |

**Outlier philosophy applied throughout:**

Two categories of outliers were treated differently. Error outliers (ages of 999, negative amounts) are data entry mistakes — they teach the model impossible patterns and were replaced or removed. Valid extreme values (very large transactions, very long distances) are real events that are often *the fraud signal* — they were winsorised at the 99th percentile to reduce StandardScaler distortion while keeping every row in the dataset.

---

## Engineered Features

Seven features were created from raw inputs during Phase 3. Engineering was performed before EDA so that charts could visualise the actual features the model uses.

| Feature | Formula | Type | Why It Was Created |
|---|---|---|---|
| `is_night` | `1 if hour_of_day ≥ 22 or hour_of_day ≤ 5, else 0` | Binary | Night-time transactions have a higher fraud rate than daytime. This converts the continuous hour into a binary risk signal |
| `amount_to_avg_ratio` | `amount / (avg_transaction_amount + 1)` | Numeric | Captures how anomalous this transaction is relative to the cardholder's own spending history. A ratio of 30 means the cardholder spent 30× their normal amount — a strong fraud signal |
| `is_high_velocity` | `1 if transaction_count_24h > 4, else 0` | Binary | Multiple rapid transactions in 24 hours. Threshold = 95th percentile of `transaction_count_24h` in the dataset (≈5) |
| `is_large_distance` | `1 if distance_from_last_transaction > 1,000, else 0` | Binary | Geographic distance above 1,000 km from the last transaction — the 'impossible travel' signal, consistent with card cloning or stolen credentials used in a different location |
| `log_amount` | `log(1 + amount)` | Numeric | Transaction amounts are right-skewed ($5 to $2,749). StandardScaler on a heavily skewed feature performs poorly. Log transformation compresses the distribution for the scaler |
| `account_age_years` | `account_age_days / 365.0` | Numeric | More human-interpretable than raw days. Newer accounts carry higher fraud risk |
| `risk_score` | `is_international × 2 + is_night × 1.5 + log(amount_to_avg_ratio) + is_high_velocity + is_large_distance` | Numeric | Weighted composite of the five binary/ratio fraud signals into a single feature. SHAP analysis confirmed this is the most important feature in the final model — combining weak signals into a composite amplifies their predictive power |

---

## Features Used in the Model

The ColumnTransformer applies three transformations, in this order:

**Numeric (15 features) → StandardScaler:**

`amount, user_age, account_age_days, transaction_count_24h, avg_transaction_amount, distance_from_last_transaction, merchant_reputation_score, log_amount, amount_to_avg_ratio, risk_score, account_age_years, hour_of_day, day_of_week, month, quarter`

**Categorical (4 features) → OneHotEncoder (handle_unknown='ignore', sparse_output=False):**

`merchant_category, card_type, device_type, country`

**Binary (5 features) → passthrough (no transformation):**

`is_weekend, is_international, is_night, is_high_velocity, is_large_distance`

**Excluded from model (retained in cleaned CSV for analytics):**

`merchant` — 66 unique values, too high cardinality for OHE without domain-specific grouping. Present in `fraud_clean.csv` for Power BI merchant-level analysis.

---

## MySQL Database Schema

Two tables store prediction data. The schema is defined in `config/db_schema.sql`.

### Table: `raw_transactions`

Stores the input transaction as received by the API, before any feature engineering.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | INT | PRIMARY KEY, AUTO_INCREMENT | Internal row identifier |
| `transaction_id` | VARCHAR(64) | UNIQUE | The transaction ID passed by the API caller |
| `amount` | DECIMAL(10,2) | NOT NULL | Transaction amount in USD |
| `merchant_category` | VARCHAR(100) | | Merchant category |
| `card_type` | VARCHAR(50) | | Card type |
| `device_type` | VARCHAR(50) | | Device type |
| `country` | VARCHAR(100) | | Transaction country |
| `user_age` | INT | | Cardholder age |
| `account_age_days` | INT | | Account age in days |
| `transaction_count_24h` | INT | | Transactions in last 24 hours |
| `avg_transaction_amount` | DECIMAL(10,2) | | Cardholder's average transaction amount |
| `distance_from_last_transaction` | DECIMAL(10,2) | | Distance from last transaction in km |
| `merchant_reputation_score` | DECIMAL(5,2) | | Merchant reputation score |
| `hour_of_day` | INT | | Hour of transaction (0–23) |
| `is_weekend` | TINYINT(1) | | 1 = weekend, 0 = weekday |
| `is_international` | TINYINT(1) | | 1 = international, 0 = domestic |
| `created_at` | DATETIME | DEFAULT NOW() | Record insertion timestamp |

### Table: `ml_fraud_predictions`

Stores the model's output for every transaction scored through the API. This table drives Power BI Page 4 (Model Performance Monitoring).

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | INT | PRIMARY KEY, AUTO_INCREMENT | Internal row identifier |
| `transaction_id` | VARCHAR(64) | | Foreign reference to raw_transactions |
| `fraud_probability` | DECIMAL(6,5) | NOT NULL | Raw model probability score (0–1) |
| `risk_tier` | VARCHAR(20) | NOT NULL | 'HIGH', 'MEDIUM', or 'LOW' |
| `decision` | VARCHAR(20) | NOT NULL | 'DECLINE', 'REVIEW', or 'APPROVE' |
| `threshold_used` | DECIMAL(4,3) | NOT NULL | The BOT value applied (0.10) |
| `top_shap_feature_1` | VARCHAR(100) | | Most important feature for this prediction |
| `top_shap_feature_2` | VARCHAR(100) | | Second most important feature |
| `top_shap_feature_3` | VARCHAR(100) | | Third most important feature |
| `shap_value_1` | DECIMAL(8,5) | | SHAP value for top feature |
| `shap_value_2` | DECIMAL(8,5) | | SHAP value for second feature |
| `shap_value_3` | DECIMAL(8,5) | | SHAP value for third feature |
| `fn_cost` | DECIMAL(8,2) | | FN cost used at prediction time ($752.09) |
| `fp_cost` | DECIMAL(5,2) | | FP cost used at prediction time ($5.00) |
| `predicted_at` | DATETIME | DEFAULT NOW() | Prediction timestamp |

### Analytics Views

Four views are defined to support Power BI connections and SQL-based monitoring:

| View | Description |
|---|---|
| `fraud_summary_by_hour` | Prediction counts, fraud flag rate, and average probability grouped by hour of day |
| `fraud_summary_by_merchant` | Prediction counts and fraud rate grouped by merchant category |
| `fraud_summary_by_country` | Prediction counts and fraud rate grouped by country |
| `model_performance_weekly` | Weekly rollup of prediction counts, tier distribution, average fraud probability, and implied precision/recall for drift monitoring |

---

## cleaned CSV Columns (fraud_clean.csv)

The cleaned CSV saved at the end of Phase 2 contains all columns needed for Power BI historical analysis. It includes `merchant` (excluded from model) and excludes raw identifiers dropped during cleaning.

Key columns present: all 15 numeric model features, all 4 categorical model features, all 5 binary features, `is_fraud` (target), `merchant` (for analytics), `day_of_week`, `month`, `quarter` (derived from `transaction_date`).

Columns absent (dropped during cleaning): `transaction_id`, `user_id`, `card_last_4`, `ip_address`, `transaction_date`, `transaction_time`
