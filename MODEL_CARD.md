# Model Card — Credit Card Fraud Detection

**Model name:** XGBoost [CW] (Tuned)  
**Model type:** Gradient Boosted Trees (XGBoost) with class_weight='balanced'  
**Version:** 1.0  
**Trained:** Phase 9 of fraud_detection.ipynb  
**Serialised as:** model.joblib  
**Intended use:** Real-time and batch credit card transaction fraud scoring

---

## Model Selection

This model was selected by an automated pipeline that trained and evaluated 10 models (5 algorithms × 2 imbalance strategies) and ranked them by a composite Business Score:

```
Business Score = 0.50 × (1 − normalised cost)
               + 0.25 × AUC
               + 0.15 × F1
               + 0.10 × (1 − normalised speed)
```

XGBoost [CW] achieved a Business Score of **0.8538**, ranking first across all 10 models. It won not because it had the highest F1 score (CatBoost [SMOTE] had a higher F1 of 0.9175), but because it produced the lowest pre-tuning cost ($72,229) combined with a fast training time (7.1 seconds) and a strong ROC-AUC (0.9504).

The model was then tuned with Optuna (15 trials, StratifiedKFold(3), F1 objective), and a Business Optimal Threshold was derived by cost minimisation across thresholds 0.05–0.95.

---

## Performance — Side by Side

### At Default 0.5 Threshold

| Metric | Value |
|---|---|
| Train F1 | 0.9865 |
| Test F1 | 0.908 |
| Precision | 0.9433 |
| Recall | 0.8753 |
| ROC-AUC | 0.9502 |
| True Positives (fraud caught) | 765 of 874 (87.5%) |
| False Positives (false alarms) | 46 |
| False Negatives (missed fraud) | 109 |
| **Total Cost** | **$82,208** |

### At Business Optimal Threshold (BOT = 0.10) ← Deployed

| Metric | Value |
|---|---|
| Precision | 0.807 |
| Recall | 0.888 |
| F1 | 0.845 |
| ROC-AUC | 0.9502 |
| True Positives (fraud caught) | 776 of 874 (88.8%) |
| False Positives (false alarms) | 186 |
| False Negatives (missed fraud) | 98 |
| **Total Cost** | **$74,635** |

**BOT saves $7,573 (9.2%) compared to the default 0.5 threshold on the same tuned model.**

### Why the Deployed Metrics Look "Worse"

Precision drops from 0.9433 at 0.5 to 0.807 at BOT. F1 drops from 0.908 to 0.845. This is expected and correct. The BOT trades some precision (more false alarms) for higher recall (more fraud caught). The trade-off is justified by the cost structure: every additional fraud caught saves $752.09, while every additional false alarm costs $5.00. The net financial result is what matters, and the net result is $7,573 better at BOT.

---

## FP/FN Cost Asymmetry

This is the central insight of the entire project.

| Error Type | Description | Cost |
|---|---|---|
| **False Negative (FN)** | A fraudulent transaction was approved | **$752.09** (average fraud transaction value — the full amount is lost) |
| **False Positive (FP)** | A legitimate transaction was declined | **$5.00** (customer service friction, complaint handling) |
| **Asymmetry ratio** | How much worse a missed fraud is than a false alarm | **150:1** |

At the default 0.5 threshold, the model misses 109 fraud cases, costing $81,978 in direct losses, and triggers 46 false alarms, costing $230. Total: $82,208.

At BOT (0.10), the model misses 98 fraud cases, costing $73,705 in direct losses, and triggers 186 false alarms, costing $930. Total: $74,635.

The extra 140 false alarms at BOT cost $700. They enable catching 11 extra fraud cases worth $8,272. The net benefit is $7,573. This is why the threshold matters and why 0.5 was rejected.

---

## Three-Tier Risk Routing

The deployed model does not produce a binary Fraud/Not Fraud output. It produces a probability score and routes every transaction to one of three tiers:

| Tier | Probability Range | System Action | Business Logic |
|---|---|---|---|
| 🔴 HIGH RISK | prob ≥ 0.70 | **Auto-decline** | Above 70% fraud probability — no analyst review needed; the cost of approval ($752.09 expected loss) far exceeds the cost of decline ($5.00) |
| 🟡 MEDIUM RISK | prob ≥ 0.10 and < 0.70 | **Route to analyst queue** | Model suspects fraud but is not highly confident — a human analyst reviews within 2 minutes |
| 🟢 LOW RISK | prob < 0.10 | **Approve** | Below the cost-optimal threshold — expected loss if approved is acceptable |

The 0.70 boundary for HIGH RISK is a business rule, not derived from cost optimisation. It reflects the point at which the probability is high enough that manual review adds no value — the correct action is always decline.

---

## Full Threshold Sweep — All Results

This table shows every threshold evaluated, confirming BOT = 0.10 is the global minimum cost.

| Threshold | Precision | Recall | F1 | TP | FP | FN | Total Cost |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.605 | 0.890 | 0.720 | 778 | 508 | 96 | $74,741 |
| **0.10** | **0.807** | **0.888** | **0.845** | **776** | **186** | **98** | **$74,635** |
| 0.15 | 0.874 | 0.884 | 0.879 | 773 | 111 | 101 | $76,516 |
| 0.20 | 0.898 | 0.884 | 0.891 | 773 | 88 | 101 | $76,401 |
| 0.25 | 0.913 | 0.881 | 0.897 | 770 | 73 | 104 | $78,582 |
| 0.30 | 0.924 | 0.881 | 0.902 | 770 | 63 | 104 | $78,532 |
| 0.35 | 0.929 | 0.880 | 0.904 | 769 | 59 | 105 | $79,264 |
| 0.40 | 0.934 | 0.876 | 0.904 | 766 | 54 | 108 | $81,496 |
| 0.45 | 0.938 | 0.876 | 0.906 | 766 | 51 | 108 | $81,481 |
| 0.50 | 0.943 | 0.875 | 0.908 | 765 | 46 | 109 | $82,208 |
| 0.55 | 0.947 | 0.872 | 0.908 | 762 | 43 | 112 | $84,449 |
| 0.60 | 0.951 | 0.871 | 0.909 | 761 | 39 | 113 | $85,181 |
| 0.65 | 0.954 | 0.870 | 0.910 | 760 | 37 | 114 | $85,923 |
| 0.70 | 0.956 | 0.870 | 0.911 | 760 | 35 | 114 | $85,913 |
| 0.75 | 0.958 | 0.868 | 0.911 | 759 | 33 | 115 | $86,655 |
| 0.80 | 0.958 | 0.866 | 0.910 | 757 | 33 | 117 | $88,160 |
| 0.85 | 0.959 | 0.863 | 0.908 | 754 | 32 | 120 | $90,411 |
| 0.90 | 0.963 | 0.859 | 0.908 | 751 | 29 | 123 | $92,652 |
| 0.95 | 0.966 | 0.855 | 0.907 | 747 | 26 | 127 | $95,645 |

---

## Business Scale Projection

At a mid-sized bank processing 200,000 transactions/month with ~4,380 expected fraud transactions:

| Strategy | Fraud Caught | False Alarms/Month | Monthly Cost | vs No Model |
|---|---|---|---|---|
| No model | 0% | — | $3,294,154 | — |
| Default 0.5 threshold | 87.5% | 230 | $411,791 | -$2,882,363/mo |
| **BOT (0.10) — deployed** | **88.8%** | **932** | **$373,936** | **-$2,920,218/mo** |

Compared to deploying the same model at the default 0.5 threshold, the BOT approach saves **$37,855/month** — **$454,259/year** — in avoidable losses.

---

## Input Features

The model accepts 24 input features (before OHE expansion):

**Numeric (15) — StandardScaler applied:**
`amount, user_age, account_age_days, transaction_count_24h, avg_transaction_amount, distance_from_last_transaction, merchant_reputation_score, log_amount, amount_to_avg_ratio, risk_score, account_age_years, hour_of_day, day_of_week, month, quarter`

**Categorical (4) — OneHotEncoder applied:**
`merchant_category, card_type, device_type, country`

**Binary (5) — passthrough:**
`is_weekend, is_international, is_night, is_high_velocity, is_large_distance`

---

## Hyperparameters (Post-Tuning)

| Parameter | Value |
|---|---|
| n_estimators | 312 |
| max_depth | 8 |
| learning_rate | 0.108 |
| subsample | 0.839 |
| colsample_bytree | 0.662 |
| min_child_weight | 2 |
| gamma | 0.174 |
| reg_alpha | 0.866 |
| reg_lambda | 1.402 |
| scale_pos_weight | derived from class_weight='balanced' |
| eval_metric | logloss |

---

## Known Limitations

**Overfitting.** Train F1 (0.9865) is significantly higher than Test F1 (0.908). Attempts during Optuna tuning to force tighter regularisation reduced Test F1 without improving cost. The gap is documented as a genuine finding. In production, the drift monitoring dashboard would flag if this gap increased over time.

**Optimised for cost minimisation, not fairness.** Country is used as a model feature. Countries with higher fraud rates in this training data may produce higher false positive rates for legitimate customers from those regions. A production deployment would require demographic fairness auditing before launch.

**Static deployment.** The model does not retrain automatically. Performance drift is monitored via the Power BI dashboard (Page 4). Retraining is a human-supervised decision triggered by that monitoring.

**Synthetic training data.** The dataset is synthetic. Real-world fraud patterns are more complex, evolve faster, and contain adversarial dynamics (fraudsters adapt to detection systems). Performance on real bank data would require separate evaluation.

**Threshold is dataset-specific.** BOT = 0.10 was derived from this dataset's cost structure and fraud rate. A different dataset, a different fraud rate, or different FN/FP cost assumptions would produce a different BOT. The method is general; the number is not transferable without re-running the BOT analysis.

---

## Experimental Finding — Stacking Ensemble

A stacking ensemble (XGBoost + LightGBM + CatBoost + Random Forest base learners, Logistic Regression meta-learner) was tested in Phase 14. It performed worse than the tuned single model on both F1 and cost, and required significantly more training and inference time. The deployed model is the tuned single winner. Stacking is documented as an informative negative result.

---

## Intended Use and Misuse Warning

**Intended use:** Portfolio demonstration of end-to-end ML engineering skills. The system is designed to show production patterns — temporal splits, automated model selection, business-cost-aware threshold optimisation, SHAP interpretability, and monitoring.

**Not intended for:** Deployment in a real financial institution without fairness auditing, compliance review, regulatory approval, and evaluation on real (non-synthetic) transaction data.
