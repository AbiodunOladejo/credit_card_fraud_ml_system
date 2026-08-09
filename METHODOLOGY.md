# Methodology — Credit Card Fraud Detection System

Every technical decision made in this project had a reason. This document explains those reasons in plain language. It is written for both technical reviewers (hiring managers, ML engineers) and non-technical stakeholders who want to understand why the system works the way it does.

---

## 1. Why Temporal Split Instead of Random Split

Most introductory ML tutorials use a random train/test split: rows are shuffled and 80% go to training, 20% to testing. This is wrong for fraud detection.

In production, a fraud model only ever sees transactions that occurred *before* the current moment. If you train on a random sample that includes future transactions, the model has effectively seen the future during training. This inflates performance metrics — the model looks better than it actually is.

This project uses a **temporal split**: the dataset is sorted chronologically, and the first 80% of transactions (by date) become the training set, the last 20% become the test set. This mirrors real deployment exactly. If the model performs well here, that performance is trustworthy.

---

## 2. Why Feature Engineering Before EDA

Feature engineering was performed in Phase 3, before EDA in Phase 4. This is a deliberate design choice.

Several EDA charts — the hourly fraud rate chart, the binary risk signal impact bars — require engineered features like `is_night` and `risk_score` to be meaningful. Engineering first means EDA visualises the actual features the model will use, not just raw inputs. It also means the training pipeline has no leakage risk from EDA-informed decisions about which features to create.

---

## 3. The Seven Engineered Features

Seven features were created from raw inputs:

| Feature | Formula | Purpose |
|---|---|---|
| `is_night` | 1 if hour ≥ 22 or hour ≤ 5, else 0 | Night-time fraud signal |
| `amount_to_avg_ratio` | amount / (avg_transaction_amount + 1) | How anomalous this amount is for this cardholder |
| `is_high_velocity` | 1 if transaction_count_24h > 4, else 0 | Multiple rapid transactions (95th percentile threshold) |
| `is_large_distance` | 1 if distance_from_last_transaction > 1,000 km | Impossible travel / card cloning signal |
| `log_amount` | log(1 + amount) | Compress right-skewed amounts for StandardScaler |
| `account_age_years` | account_age_days / 365 | More interpretable than raw days |
| `risk_score` | weighted composite of is_international, is_night, log(amount_to_avg_ratio), is_high_velocity, is_large_distance | Single combined fraud signal — confirmed by SHAP as the most important feature |

The `merchant` column (66 unique values) was excluded from model features due to high cardinality but retained in the cleaned CSV for Power BI analytics use.

---

## 4. Why Two Imbalance Strategies Were Compared

Credit card fraud datasets are severely imbalanced — in this dataset, 97.8% of transactions are legitimate and 2.2% are fraudulent (approximately 44:1 ratio). A model trained on this raw distribution tends to predict "legitimate" for almost everything, achieving high accuracy while catching little fraud.

Two standard strategies exist to address this:

**Class weight (`class_weight='balanced'`)** — tells the model to penalise mistakes on the minority class (fraud) more heavily during training. The class weights are set inversely proportional to class frequencies. This requires no data modification and adds no training time.

**SMOTE (Synthetic Minority Oversampling Technique)** — generates synthetic fraud examples by interpolating between existing fraud cases, bringing the training set closer to balance. This requires data modification and significantly increases training time (from seconds to minutes at full `sampling_strategy=1.0`).

Both strategies were tested on all five algorithms (XGBoost, CatBoost, LightGBM, Random Forest, Logistic Regression), producing 10 models total.

---

## 5. SMOTE Speed Fix — Why sampling_strategy=0.15

SMOTE at `sampling_strategy=1.0` (full 1:1 balance) produced slightly lower cost and marginally better F1 in some models, but required **40+ minutes** of training time — driven by the volume of synthetic samples being generated, not model complexity.

This trade-off was unacceptable for a system that must be retrained periodically. The parameter was reduced to `sampling_strategy=0.15`, which generates far fewer synthetic samples, cuts training to under 5 minutes, and sacrifices only a small amount of performance. This trade-off is an honest engineering decision, not a shortcut.

Note: this parameter only affects SMOTE-strategy models. The winning model (XGBoost [CW]) uses class weights and is entirely unaffected by SMOTE parameters.

---

## 6. Why MLP Was Not Included

Multi-layer Perceptrons (neural networks) were considered but excluded from the comparison. The reasons are practical:

- MLPs require substantially more hyperparameter tuning effort (architecture search, learning rate schedules, batch size, regularisation) than tree-based models
- On tabular data with ~24,000 training samples, gradient-boosted trees consistently outperform MLPs without the tuning overhead
- The goal of Phase 8 was a fair, automated comparison on a level playing field; including MLP would require a disproportionate amount of tuning effort to make it competitive
- The five algorithms included (XGBoost, CatBoost, LightGBM, Random Forest, Logistic Regression) provide a thorough coverage of the tree-based family and a linear baseline

---

## 7. Automated Winner Selection — The Business Score

The winner was not chosen by hand. An automated composite Business Score ranked all 10 models:

```
Business Score = 0.50 × (1 − normalised cost)
               + 0.25 × AUC
               + 0.15 × F1
               + 0.10 × (1 − normalised speed)
```

Cost carries 50% of the weight because this is a cost-minimisation problem, not a benchmark problem. AUC carries 25% because it measures discrimination ability across all thresholds — relevant because we will apply BOT, not the default 0.5. F1 carries 15% as a secondary metric. Speed carries 10% because a model that takes 54 seconds to train is harder to retrain frequently.

**XGBoost [CW] won with a Business Score of 0.8538.** Although CatBoost [SMOTE] had a higher Test F1 (0.9175 vs 0.8612), its cost was $80,629 vs $72,229 — an $8,400 difference. The Business Score correctly penalised that cost gap.

---

## 8. Why the Optuna Objective Was F1, Not Cost

Optuna was used to tune XGBoost [CW] across 15 trials with StratifiedKFold(3), optimising for cross-validated F1. A natural question is: if cost is what matters, why not optimise for cost directly?

Two reasons:

**Cross-validation cost is unreliable.** Computing cost in cross-validation requires knowing FP and FN counts from each fold, then summing them. This works but introduces fold-level variance — cost swings dramatically depending on how fraud cases are distributed across folds. F1 is more stable as a CV objective.

**BOT analysis handles threshold post-hoc.** Because the BOT step (Phase 11) finds the cost-minimising threshold after training, training to the best discrimination ability (F1, AUC) and then applying the right threshold produces the same end result. The threshold is the cost-aware step; training does not need to be.

The consequence: tuning improved Test F1 from 0.8612 to 0.908 (+0.0468) but raised cost at 0.5 from $72,229 to $82,208. This is expected — Optuna made the model better at separating fraud from legitimate cases but did not change the threshold. BOT analysis on the tuned model brought cost down to $74,635.

Best Optuna parameters found: `n_estimators=312, max_depth=8, learning_rate=0.108, subsample=0.839, colsample_bytree=0.662, min_child_weight=2, gamma=0.174, reg_alpha=0.866, reg_lambda=1.402`

---

## 9. Why 0.5 Was Rejected as the Deployment Threshold

The default 0.5 threshold treats fraud and legitimate predictions as equally costly to get wrong. They are not. Missing a fraud case costs $752.09; triggering a false alarm costs $5.00. The asymmetry is 150:1.

At 0.5, the tuned model costs $82,208. By sweeping thresholds from 0.05 to 0.95 in 0.05 steps and computing the total cost at each, the minimum was found at threshold 0.10 — a cost of $74,635. This is the Business Optimal Threshold (BOT).

Choosing a lower threshold means the model flags more transactions as potentially fraudulent. This increases false alarms (from 46 at 0.5 to 186 at BOT) but catches 11 more actual fraud cases (from 765 at 0.5 to 776 at BOT). The 11 additional fraud cases recovered ($8,272 in prevented loss) exceeds the cost of the extra 140 false alarms ($700). The net saving is $7,573.

---

## 10. How BOT Was Computed

For each threshold value t from 0.05 to 0.95:

1. Apply t to the tuned model's test-set probability scores
2. Compute the confusion matrix (TP, FP, FN, TN)
3. Compute Total Cost = FN × $752.09 + FP × $5.00
4. Record the result

The threshold with the minimum Total Cost is the Business Optimal Threshold. BOT = **0.10**, producing Total Cost = **$74,635**.

This approach is grounded in the actual cost structure of the business, not in any generic metric. If FN_COST or FP_COST change (e.g., at a higher-value bank), the BOT would shift accordingly — the method is general, the result is dataset-specific.

---

## 11. The Overfitting Finding — Why It Was Left As-Is

The tuned model shows a gap between Train F1 (0.9865) and Test F1 (0.908). This is moderate overfitting.

During Optuna tuning, the regularisation parameters (`reg_alpha`, `reg_lambda`, `gamma`, `min_child_weight`, `max_depth`) were part of the search space. The best parameters found were not maximally regularised — the model found a balance that produced the best CV F1, which included some degree of overfitting.

Attempts to force tighter regularisation (manually increasing `reg_alpha` and `reg_lambda` beyond the Optuna results) reduced Test F1 without improving cost. In other words, accepting the overfitting was the cost-minimising choice.

This finding is documented honestly. It is not hidden. In production, it would be monitored via the Power BI drift dashboard — if Test F1 began degrading over time relative to train metrics, retraining would be triggered.

---

## 12. Stacking Ensemble — Experimental Finding

Phase 14 tested a stacking ensemble: XGBoost, LightGBM, CatBoost, and Random Forest as base learners, with Logistic Regression as the meta-learner, using SMOTE-balanced inputs and 3-fold cross-validation.

The result was worse than the tuned single model on both F1 and cost, and significantly slower. This is informative: on this dataset, with this number of training samples, the diversity gain from combining multiple learners did not offset the variance introduced by the meta-learner or the latency overhead from running four base models at inference time.

In real-time fraud detection, inference latency is a hard constraint. Running four models at every transaction is not justifiable when a single tuned model produces better results. The deployed model is always the tuned single winner.

---

## 13. Why XGBoost [CW] Over SMOTE Models — Deeper Reasoning

The three top-scoring models in the Business Score ranking were all within 0.01 of each other (XGBoost [CW]: 0.8538, CatBoost [SMOTE]: 0.8479, XGBoost [SMOTE]: 0.8440). The cost differences were more meaningful:

- XGBoost [CW]: $72,229
- CatBoost [SMOTE]: $80,629 (+$8,400)
- XGBoost [SMOTE]: $84,374 (+$12,145)

The class-weight approach produced lower cost despite lower F1. This confirms the Business Score's design: a model that minimises financial loss is a better model, regardless of benchmark position.

---

## 14. Deployment Stack Justification

**FastAPI** was chosen over Flask for its automatic OpenAPI documentation, native async support, and Pydantic validation — production patterns, not tutorial patterns.

**Docker** ensures the API environment is reproducible. The python:3.10-slim base keeps the image small.

**Render** hosts the Docker container with managed MySQL. Free tier is sufficient for portfolio demonstration.

**Streamlit on Hugging Face Spaces** provides a shareable UI without infrastructure cost. The `DATABASE_URL` is injected via Hugging Face Space secrets — no credentials are ever in code.

**MySQL over SQLite** was a deliberate choice. SQLite would be simpler, but MySQL mirrors production database infrastructure and enables the Power BI live connection that is central to the model monitoring narrative.

---

## 15. MySQL Role — Persistence, Not Learning

MySQL stores every prediction made by the API: the input transaction, the model's probability score, the tier assignment, the threshold used, and a timestamp. It does not feed back into the model.

The purpose of storing predictions is **monitoring**. Over time, if fraud rates in predictions drift away from historical rates, or if precision/recall degrade week over week (visible on Power BI Page 4), that is a signal that the underlying data distribution has changed and the model needs retraining.

This architecture — predict → store → monitor → human decision to retrain — is how production ML systems operate. The MySQL table is the connective tissue between the model and the monitoring loop.

---

## 16. Retraining Trigger Logic via Power BI Monitoring

The system does not retrain automatically. Retraining is a human decision triggered by signals visible on Power BI Page 4:

- Weekly recall drops below 85% (the model is missing more fraud than expected)
- Weekly precision drops below 75% (false alarm rate is increasing)
- Cost at BOT drifts upward week over week
- The fraud-by-tier distribution shifts materially (e.g., more cases appearing in the HIGH tier than historical baseline)

When any of these signals appear, the analyst escalates to the ML engineer, who retrains the model using the accumulated `raw_transactions` data, re-runs the BOT analysis, and deploys the new `model.joblib`. The BOT may shift — that is expected and desirable.

---

## 16. Feature Preprocessing Pipeline

The ColumnTransformer applied in this exact order (order matters for predict.py feature reconstruction):

1. **Numeric (15 features → StandardScaler):** `amount, user_age, account_age_days, transaction_count_24h, avg_transaction_amount, distance_from_last_transaction, merchant_reputation_score, log_amount, amount_to_avg_ratio, risk_score, account_age_years, hour_of_day, day_of_week, month, quarter`

2. **Categorical (4 features → OneHotEncoder, handle_unknown='ignore', sparse_output=False):** `merchant_category, card_type, device_type, country`

3. **Binary (5 features → passthrough):** `is_weekend, is_international, is_night, is_high_velocity, is_large_distance`

The preprocessor is fitted on the training set only and applied to the test set — no data leakage. At inference time, the same ColumnTransformer (stored inside `model.joblib`) is applied to incoming single transactions before prediction.
