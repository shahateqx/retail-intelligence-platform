"""Workstream 4 — Event impact via synthetic control / difference-in-differences.

Usage: python src/causal_impact.py
Design: treated = daily revenue from the top market; controls = other markets' daily revenue.
        Pre-period OLS counterfactual predicts what treated revenue would have been without the
        event. Assumption: parallel trends (checked via pre-period correlation + stationarity).
"""
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "outputs")
EVENT = "2010-11-26"          # Black Friday 2010 — real retail event inside the data window
POST_DAYS = 28          # 4 weekly observations


def build_market_panel(df):
    # Weekly aggregation reduces zero-inflation/noise and strengthens parallel-trends fit
    p = df.groupby(["Date", "Country"])["Sales"].sum().unstack(fill_value=0)
    p.index = pd.to_datetime(p.index)
    return p.resample("W").sum().fillna(0)


def adf_p(series):
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    return adfuller(s, autolag="AIC")[1]


def main():
    df = pd.read_csv(os.path.join(PROC, "retail_clean.csv"), parse_dates=["InvoiceDate"])
    panel = build_market_panel(df)
    treated_country = panel.sum().idxmax()
    event = pd.Timestamp(EVENT)
    y = panel[treated_country]
    pre = y.index < event

    # Control selection by pre-period correlation (as in the course: correlation matrix,
    # not by market size). Ensures comparable counterfactual donors.
    cors = panel.drop(columns=treated_country)[pre].apply(lambda c: y[pre].corr(c))
    controls = list(cors.sort_values(ascending=False).head(5).index)
    print("Treated market:", treated_country, "\nControl markets (pre-period correlation):", controls)

    X = panel[controls]
    post = (y.index >= event) & (y.index < event + pd.Timedelta(days=POST_DAYS))

    # Assumption checks
    corr = y[pre].corr(X[pre].sum(axis=1))
    print(f"Pre-period correlation treated vs control basket: {corr:.3f}")
    print("ADF p-value (treated, levels):", round(adf_p(y[pre]), 4))
    pre_ret = y[pre].pct_change()
    print("ADF p-value (treated, pct change):", round(adf_p(pre_ret), 4))

    # Counterfactual model
    Xpre = sm.add_constant(X[pre])
    model = sm.OLS(y[pre], Xpre).fit()
    Xpost = sm.add_constant(X[post], has_constant="add")
    pred = model.get_prediction(Xpost)
    summary = pred.summary_frame(alpha=0.05)

    actual = y[post]
    lo_col = "mean_ci_lower" if "mean_ci_lower" in summary else "mean_lower"
    hi_col = "mean_ci_upper" if "mean_ci_upper" in summary else "mean_upper"
    effect = actual - summary["mean"]
    ci_lo = actual - summary[hi_col]
    ci_hi = actual - summary[lo_col]

    print("\n--- CAUSAL IMPACT (synthetic control) ---")
    print(f"Event: {EVENT} | post window: {POST_DAYS} days | pre-period R2: {model.rsquared:.3f}")
    print(f"Actual mean:    {actual.mean():,.0f}")
    print(f"Predicted mean: {summary['mean'].mean():,.0f}")
    print(f"Average effect: {effect.mean():,.0f}  ({100*effect.mean()/summary['mean'].mean():.1f}%)")
    print(f"Cumulative effect: {effect.sum():,.0f}")
    print(f"95% CI of cumulative effect: [{ci_lo.sum():,.0f}, {ci_hi.sum():,.0f}]")
    print("Significant (CI excludes 0):", "YES" if ci_lo.sum() > 0 or ci_hi.sum() < 0 else "NO")

    out = pd.DataFrame({
        "date": actual.index, "actual": actual.values,
        "predicted": summary["mean"].values, "effect": effect.values,
        "ci_low": ci_lo.values, "ci_high": ci_hi.values,
    })
    out.to_csv(os.path.join(OUT, "causal_impact_results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(y[pre].index, y[pre].values, color="black", label="Actual (pre)")
    ax.plot(actual.index, actual.values, color="black", lw=2, label="Actual (post)")
    ax.plot(actual.index, summary["mean"].values, color="orange", ls="--", label="Counterfactual")
    ax.fill_between(actual.index, summary[lo_col], summary[hi_col], color="orange", alpha=0.2)
    ax.axvline(event, color="grey", ls=":")
    ax.set_title(f"Impact of {EVENT} on {treated_country} daily revenue")
    ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT, "causal_impact.png"), dpi=120); plt.close()


if __name__ == "__main__":
    main()
