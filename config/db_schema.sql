-- ============================================================
-- config/db_schema.sql
-- Credit Card Fraud Detection — MySQL Schema
-- ============================================================
-- Two tables:
--   raw_transactions      — input transaction as received by the API
--   ml_fraud_predictions  — model output for every scored transaction
--
-- Four analytics views (Power BI / monitoring):
--   fraud_summary_by_hour
--   fraud_summary_by_merchant
--   fraud_summary_by_country
--   model_performance_weekly
-- ============================================================

-- ------------------------------------------------------------
-- Table 1: raw_transactions
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_transactions (
    id                              INT           NOT NULL AUTO_INCREMENT,
    transaction_id                  VARCHAR(64)   NULL,
    amount                          DECIMAL(10,2) NOT NULL,
    merchant_category               VARCHAR(100)  NULL,
    card_type                       VARCHAR(50)   NULL,
    device_type                     VARCHAR(50)   NULL,
    country                         VARCHAR(100)  NULL,
    user_age                        INT           NULL,
    account_age_days                INT           NULL,
    transaction_count_24h           INT           NULL,
    avg_transaction_amount          DECIMAL(10,2) NULL,
    distance_from_last_transaction  DECIMAL(10,2) NULL,
    merchant_reputation_score       DECIMAL(5,2)  NULL,
    hour_of_day                     INT           NULL,
    is_weekend                      TINYINT(1)    NULL,
    is_international                TINYINT(1)    NULL,
    created_at                      DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE  INDEX uq_transaction_id  (transaction_id),
    INDEX   idx_created_at           (created_at),
    INDEX   idx_country              (country),
    INDEX   idx_merchant_category    (merchant_category),
    INDEX   idx_hour_of_day          (hour_of_day)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ------------------------------------------------------------
-- Table 2: ml_fraud_predictions
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ml_fraud_predictions (
    id                  INT            NOT NULL AUTO_INCREMENT,
    transaction_id      VARCHAR(64)    NULL,
    fraud_probability   DECIMAL(6,5)   NOT NULL,
    risk_tier           VARCHAR(20)    NOT NULL,
    decision            VARCHAR(20)    NOT NULL,
    threshold_used      DECIMAL(4,3)   NOT NULL,
    top_shap_feature_1  VARCHAR(100)   NULL,
    top_shap_feature_2  VARCHAR(100)   NULL,
    top_shap_feature_3  VARCHAR(100)   NULL,
    shap_value_1        DECIMAL(8,5)   NULL,
    shap_value_2        DECIMAL(8,5)   NULL,
    shap_value_3        DECIMAL(8,5)   NULL,
    fn_cost             DECIMAL(8,2)   NULL,
    fp_cost             DECIMAL(5,2)   NULL,
    predicted_at        DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    INDEX idx_transaction_id  (transaction_id),
    INDEX idx_predicted_at    (predicted_at),
    INDEX idx_risk_tier       (risk_tier),
    INDEX idx_decision        (decision),
    INDEX idx_tier_date       (risk_tier, predicted_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ============================================================
-- ANALYTICS VIEWS
-- ============================================================

-- View 1: fraud_summary_by_hour
CREATE OR REPLACE VIEW fraud_summary_by_hour AS
SELECT
    rt.hour_of_day,
    COUNT(p.id)                                                                   AS total_predictions,
    SUM(CASE WHEN p.risk_tier IN ('HIGH','MEDIUM') THEN 1 ELSE 0 END)            AS flagged_count,
    SUM(CASE WHEN p.risk_tier = 'HIGH'   THEN 1 ELSE 0 END)                      AS high_risk_count,
    SUM(CASE WHEN p.risk_tier = 'MEDIUM' THEN 1 ELSE 0 END)                      AS medium_risk_count,
    SUM(CASE WHEN p.risk_tier = 'LOW'    THEN 1 ELSE 0 END)                      AS low_risk_count,
    ROUND(SUM(CASE WHEN p.risk_tier IN ('HIGH','MEDIUM') THEN 1 ELSE 0 END)
          / NULLIF(COUNT(p.id),0) * 100, 2)                                       AS fraud_flag_rate_pct,
    ROUND(AVG(p.fraud_probability), 5)                                            AS avg_fraud_probability,
    CASE WHEN rt.hour_of_day >= 22 OR rt.hour_of_day <= 5 THEN 1 ELSE 0 END      AS is_night_window
FROM ml_fraud_predictions p
JOIN raw_transactions rt ON p.transaction_id = rt.transaction_id
GROUP BY rt.hour_of_day
ORDER BY rt.hour_of_day;


-- View 2: fraud_summary_by_merchant
CREATE OR REPLACE VIEW fraud_summary_by_merchant AS
SELECT
    rt.merchant_category,
    COUNT(p.id)                                                                   AS total_predictions,
    SUM(CASE WHEN p.risk_tier IN ('HIGH','MEDIUM') THEN 1 ELSE 0 END)            AS flagged_count,
    SUM(CASE WHEN p.risk_tier = 'HIGH'   THEN 1 ELSE 0 END)                      AS high_risk_count,
    SUM(CASE WHEN p.risk_tier = 'MEDIUM' THEN 1 ELSE 0 END)                      AS medium_risk_count,
    ROUND(SUM(CASE WHEN p.risk_tier IN ('HIGH','MEDIUM') THEN 1 ELSE 0 END)
          / NULLIF(COUNT(p.id),0) * 100, 2)                                       AS fraud_flag_rate_pct,
    ROUND(AVG(p.fraud_probability), 5)                                            AS avg_fraud_probability,
    ROUND(AVG(rt.amount), 2)                                                      AS avg_transaction_amount,
    ROUND(SUM(CASE WHEN p.risk_tier IN ('HIGH','MEDIUM') THEN 1 ELSE 0 END)
          * 752.09, 2)                                                            AS implied_fraud_value_at_risk
FROM ml_fraud_predictions p
JOIN raw_transactions rt ON p.transaction_id = rt.transaction_id
GROUP BY rt.merchant_category
ORDER BY fraud_flag_rate_pct DESC;


-- View 3: fraud_summary_by_country
CREATE OR REPLACE VIEW fraud_summary_by_country AS
SELECT
    rt.country,
    COUNT(p.id)                                                                   AS total_predictions,
    SUM(CASE WHEN p.risk_tier IN ('HIGH','MEDIUM') THEN 1 ELSE 0 END)            AS flagged_count,
    SUM(CASE WHEN p.risk_tier = 'LOW' THEN 1 ELSE 0 END)                         AS approved_count,
    ROUND(SUM(CASE WHEN p.risk_tier IN ('HIGH','MEDIUM') THEN 1 ELSE 0 END)
          / NULLIF(COUNT(p.id),0) * 100, 2)                                       AS fraud_flag_rate_pct,
    ROUND(AVG(p.fraud_probability), 5)                                            AS avg_fraud_probability,
    ROUND(AVG(rt.amount), 2)                                                      AS avg_transaction_amount,
    SUM(CASE WHEN rt.is_international = 1 THEN 1 ELSE 0 END)                     AS international_txn_count
FROM ml_fraud_predictions p
JOIN raw_transactions rt ON p.transaction_id = rt.transaction_id
GROUP BY rt.country
ORDER BY fraud_flag_rate_pct DESC;


-- View 4: model_performance_weekly
-- The drift monitoring view. Powers Power BI Page 4.
CREATE OR REPLACE VIEW model_performance_weekly AS
SELECT
    YEARWEEK(p.predicted_at, 1)                                                   AS year_week,
    DATE(DATE_SUB(p.predicted_at,
         INTERVAL WEEKDAY(p.predicted_at) DAY))                                   AS week_start_date,
    COUNT(p.id)                                                                   AS total_predictions,
    SUM(CASE WHEN p.risk_tier = 'HIGH'   THEN 1 ELSE 0 END)                      AS high_risk_count,
    SUM(CASE WHEN p.risk_tier = 'MEDIUM' THEN 1 ELSE 0 END)                      AS medium_risk_count,
    SUM(CASE WHEN p.risk_tier = 'LOW'    THEN 1 ELSE 0 END)                      AS low_risk_count,
    SUM(CASE WHEN p.risk_tier IN ('HIGH','MEDIUM') THEN 1 ELSE 0 END)            AS total_flagged,
    ROUND(SUM(CASE WHEN p.risk_tier IN ('HIGH','MEDIUM') THEN 1 ELSE 0 END)
          / NULLIF(COUNT(p.id),0) * 100, 2)                                       AS flag_rate_pct,
    ROUND(AVG(p.fraud_probability), 5)                                            AS avg_fraud_probability,
    ROUND(MAX(p.fraud_probability), 5)                                            AS max_fraud_probability,
    ROUND(MIN(p.fraud_probability), 5)                                            AS min_fraud_probability,
    ROUND(SUM(CASE WHEN p.risk_tier = 'HIGH' THEN 1 ELSE 0 END)
          / NULLIF(SUM(CASE WHEN p.risk_tier IN ('HIGH','MEDIUM') THEN 1 ELSE 0 END),0)
          * 100, 2)                                                               AS high_tier_concentration_pct,
    ROUND(SUM(CASE WHEN p.risk_tier IN ('HIGH','MEDIUM') THEN 1 ELSE 0 END)
          * 752.09 * (1 - AVG(p.fraud_probability)), 2)                           AS estimated_weekly_fn_cost,
    ROUND(SUM(CASE WHEN p.risk_tier = 'LOW' THEN 1 ELSE 0 END)
          * AVG(p.fraud_probability) * 752.09, 2)                                 AS estimated_weekly_missed_fraud_cost,
    MIN(p.predicted_at)                                                           AS week_first_prediction,
    MAX(p.predicted_at)                                                           AS week_last_prediction
FROM ml_fraud_predictions p
GROUP BY YEARWEEK(p.predicted_at, 1),
         DATE(DATE_SUB(p.predicted_at, INTERVAL WEEKDAY(p.predicted_at) DAY))
ORDER BY year_week DESC;
