# Business Memo — Retail Intelligence Platform

**To:** Executive team, Online Retail II
**From:** Analytics
**Subject:** Customer value, retention, event impact and demand forecasting
**Data:** UCI Online Retail II — 1,067,371 real transactions, Dec 2009 – Dec 2011 (CC BY 4.0)

---

## 1. Executive summary

- The business is highly concentrated: the **top 5% of customers generate 52% of revenue** and the
  top 20% generate **77%** — close to the classic Pareto 80/20 pattern.
- **12.3% of customers are "Superstars"** (721 people) but deliver **58.5% of revenue (£10.2M)**.
  Gaussian Mixture analysis isolates a further **"whale" segment of 64 customers** (1.1% of the base)
  with an average basket of **£2,917** and **80 orders** each, contributing **£4.96M (28.6%)**.
- **Retention is deteriorating**: the 90-day repeat-purchase rate fell from **85.0%** for 2009-12 to
  2010-05 cohorts to **61.6%** for 2010-09 to 2011-02 cohorts (z = 16.3, p ≈ 8.7e-60, power = 1.00).
- **Black Friday 2010 produced a measurable +25.9% lift** in UK weekly revenue (≈ £39.6k/week,
  ≈ £158k cumulative over four weeks; 95% CI £77k–£239k).
- A **leakage-free predictive model** identifies future high-value customers with **ROC-AUC 0.905**
  and **PR-AUC 0.719** — enabling targeted retention spend.
- **Prophet demand forecast** beats a seasonal-naive baseline by **20% on MAE** (£15.1k vs £18.8k
  daily error); cross-validated tuning improved RMSE only 1.8% — the defaults were near-optimal.

## 2. Data quality

| Step | Rows |
|---|---:|
| Raw transactions (two Excel sheets) | 1,067,371 |
| Cancellations (`C` invoices) removed | −19,494 |
| Missing CustomerID removed | −236,121 |
| Duplicates removed | −26,124 |
| **Analysis-ready transactions** | **779,425 (73.0%)** |

Final scope: 5,878 identified customers · 41 countries · 4,631 products · 36,969 invoices ·
**£17,374,804** revenue over 2009-12-01 → 2011-12-09.

## 3. Customer value & segmentation

**RFM scoring** (quartile ranks; Recency reversed so recent = better):

| Segment | Customers | Avg recency (days) | Avg orders | Avg basket | Revenue | Revenue share |
|---|---:|---:|---:|---:|---:|---:|
| Superstar | 721 | 19.3 | 19.6 | £619.7 | £10,167,594 | **58.5%** |
| Future Champion | 2,244 | 81.4 | 7.4 | £453.2 | £5,722,171 | 32.9% |
| High Potential | 1,533 | 255.0 | 2.8 | £369.1 | £1,167,025 | 6.7% |
| Low Relevance | 1,380 | 431.8 | 1.4 | £169.9 | £318,015 | 1.8% |

**Concentration:** top 1% = 32.0% · top 5% = 52.0% · top 10% = 63.9% · top 20% = 77.2%.

**Gaussian Mixture (AIC/BIC-selected, k=8)** adds overlap-aware segments. The most striking is a
cluster of **64 "whale" customers** averaging **80 orders** and a **£2,917 basket**, producing
£4.96M — 28.6% of all revenue from 1.1% of customers. RFM alone would blend these into "Superstar".

**Recommendation:** ring-fence the Superstar + whale cohorts for a dedicated account-management
programme; do not apply blanket discounts. Nurture "High Potential" (recent but low frequency)
with a second-purchase incentive — it is the cheapest source of future Superstars.

## 4. Retention & experimentation

Monthly cohort curves show first-month retention drifting down from ~35% (2009-12) to ~9%
(2010-12). A censoring-safe 90-day repeat-rate comparison of first-purchase cohorts confirms a
real decline: **85.0% (n=2,703) → 61.6% (n=1,216)**, z = 16.31, p ≈ 8.7e-60, achieved power = 1.00.

**Recommendation:** treat the drop as a strategic issue, not noise. Investigate onboarding/product
mix changes between the two periods; target the "High Potential" segment with lifecycle messaging.

## 5. Event impact (synthetic control / difference-in-differences)

**Event:** Black Friday, 26 Nov 2010. **Treated:** UK weekly revenue. **Controls:** the five markets
with the highest pre-period correlation (Portugal, Germany, Spain, Channel Islands, Switzerland).
Parallel-trends diagnostics: pre-period correlation 0.63, ADF p = 0.039 (stationary after weekly
aggregation), counterfactual pre-period R² = 0.495.

| Metric | Value |
|---|---:|
| Actual mean weekly revenue | £192,180 |
| Counterfactual prediction | £152,619 |
| Average weekly effect | **+£39,561 (+25.9%)** |
| Cumulative 4-week effect | **£158,244** |
| 95% CI (cumulative) | £77,225 – £239,263 |

**Caveat:** R² = 0.495 is only moderate — parallel trends is plausible but not perfect. Treat the
magnitude as directional; replicate on more events before basing budget decisions on it.

## 6. Predictive model — future high-value customers

Leakage-free setup: features from history before **2011-06-01**, target = top-20% spend in the
following six months (threshold £956; 4,933 customers). Best model: Random Forest
(n_estimators=300, max_depth=5), tuned by 5-fold GridSearchCV on ROC-AUC.

- **ROC-AUC 0.905 · PR-AUC 0.719**
- High-value class: precision 0.535, recall 0.807, F1 0.644
- Top drivers: past total spend (37%), purchase frequency (19%), recency (13%)

**Recommendation:** score the active base monthly; contact the top decile of predicted future
high-value customers before they lapse. Expected precision ≈ 54% means roughly half of contacted
customers would genuinely become high-value.

## 7. Forecast / demand plan

Daily revenue (739 calendar days, including zero-sale days as £0) with weekly and yearly
seasonality. **31-day holdout:**

| Model | MAE | RMSE | MAPE (selling days) |
|---|---:|---:|---:|
| Seasonal naive (last week's same weekday) | £18,773 | £29,154 | 45.1% |
| Prophet (default parameters) | **£15,062** | £30,110 | **24.8%** |

Prophet cut average absolute error by **20%** versus the naive baseline; MAPE improved from
45.1% to 24.8% (its RMSE was marginally higher due to a few large spike days).

**Time-series cross-validation** (rolling cutoffs, 31-day horizon) gave a mean RMSE of £14,187 and
MAE of £10,698 for the default configuration. A 16-combination parameter grid tuned on RMSE found
a best configuration of *multiplicative seasonality, seasonality_prior_scale = 5,
holidays_prior_scale = 10, changepoint_prior_scale = 0.01*, reaching CV RMSE **£13,937 — only a
1.8% gain**, i.e. the defaults were already near-optimal. The learning here is that disciplined
cross-validation tells you when *not* to over-engineer.

**Recommendation:** use the Prophet model for monthly revenue planning with a ±£14k error band;
do not rely on point forecasts for individual days. Re-fit when more than two full years of history
are available — yearly seasonality is currently under-identified (Prophet issues a warning), so
treat the yearly component as directional.

## 8. Limitations

- Only identified customers are analysed (236k transactions without CustomerID excluded).
- Segmentation is descriptive; causal claims are limited to the event study in §5.
- Data ends Dec 2011 — patterns may not transfer to current market conditions.
