"""Workstream 5 — Daily revenue forecasting with Prophet, CV and tuning.

Usage: python src/forecasting.py
Notes: align with the course workflow — holiday/regressor handling, time-series split,
       cross_validation(), parameter grid, RMSE for tuning / MAE for communication.
"""
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "outputs")


def build_daily(df):
    d = df.groupby("Date", as_index=False)["Sales"].sum().rename(columns={"Date": "ds", "Sales": "y"})
    d["ds"] = pd.to_datetime(d["ds"])
    d = d.set_index("ds").sort_index()
    # Calendar days with no transactions are genuine zero-revenue days — include them.
    full = pd.date_range(d.index.min(), d.index.max(), freq="D")
    d = d.reindex(full, fill_value=0).rename_axis("ds").reset_index()
    return d


def make_model(params=None, holidays=None, regressors=None):
    p = params or {}
    m = Prophet(
        growth=p.get("growth", "linear"),
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode=p.get("seasonality_mode", "additive"),
        seasonality_prior_scale=p.get("seasonality_prior_scale", 10),
        holidays_prior_scale=p.get("holidays_prior_scale", 10),
        changepoint_prior_scale=p.get("changepoint_prior_scale", 0.05),
        holidays=holidays,
    )
    for r in (regressors or []):
        m.add_regressor(r)
    return m


def main():
    df = pd.read_csv(os.path.join(PROC, "retail_clean.csv"), parse_dates=["InvoiceDate"])
    daily = build_daily(df)
    daily.to_csv(os.path.join(PROC, "daily_revenue.csv"), index=False)
    print("Daily observations:", len(daily), "| date range:", daily.ds.min().date(), "to", daily.ds.max().date())

    test_days = 31
    train, test = daily.iloc[:-test_days].copy(), daily.iloc[-test_days:].copy()

    # Baseline 1: seasonal naive (same weekday last week)
    naive = train["y"].iloc[-7:].values
    yhat_naive = np.array([naive[i % 7] for i in range(len(test))])

    # Prophet baseline
    m = make_model()
    m.fit(train)
    future = m.make_future_dataframe(periods=test_days, freq="D")
    fc = m.predict(future)
    pred = fc.set_index("ds").loc[test["ds"], "yhat"].values

    def metrics(actual, pred):
        mae = round(mean_absolute_error(actual, pred), 1)
        rmse = round(float(np.sqrt(mean_squared_error(actual, pred))), 1)
        # MAPE is undefined on zero-revenue days — evaluate only on selling days.
        mask = np.asarray(actual) > 0
        mape = round(100 * np.mean(np.abs((np.asarray(actual)[mask] - np.asarray(pred)[mask])
                                          / np.asarray(actual)[mask])), 1)
        return mae, rmse, mape

    print("\n--- HOLDOUT FORECAST (last 31 days) ---")
    print(f"Seasonal naive      MAE/RMSE/MAPE: {metrics(test.y.values, yhat_naive)}")
    print(f"Prophet (baseline)  MAE/RMSE/MAPE: {metrics(test.y.values, pred)}")

    fig = m.plot(fc); fig.set_size_inches(12, 4); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "prophet_forecast.png"), dpi=120); plt.close()
    fig = m.plot_components(fc); fig.set_size_inches(10, 8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "prophet_components.png"), dpi=120); plt.close()

    # Cross-validation on training history
    print("\nRunning time-series cross-validation ...", flush=True)
    df_cv = cross_validation(
        m, horizon="31 days", initial=f"{max(300, len(train)-120)} days", parallel="processes"
    )
    pm = performance_metrics(df_cv)
    cols = [c for c in ["horizon", "rmse", "mae", "mape"] if c in pm.columns]
    print("CV metrics (RMSE, MAE, MAPE):")
    print(pm[cols].tail(5).to_string(index=False))
    try:
        fig = plot_cross_validation_metric(df_cv, metric="rmse"); plt.tight_layout()
        plt.savefig(os.path.join(OUT, "prophet_cv_rmse.png"), dpi=120); plt.close()
    except Exception as e:
        print("CV plot skipped:", e)

    # Small parameter grid (expand to 54 combinations for full study)
    grid = {
        "seasonality_mode": ["additive", "multiplicative"],
        "seasonality_prior_scale": [5, 10],
        "holidays_prior_scale": [5, 10],
        "changepoint_prior_scale": [0.01, 0.05],
    }
    import itertools
    results = []
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), combo))
        mm = make_model(params)
        mm.fit(train)
        cv = cross_validation(mm, horizon="31 days", initial=f"{max(300, len(train)-120)} days")
        err = performance_metrics(cv)["rmse"].mean()
        results.append({**params, "rmse": round(err, 1)})
        print("  tuned:", params, "RMSE", round(err, 1), flush=True)
    res = pd.DataFrame(results).sort_values("rmse")
    res.to_csv(os.path.join(OUT, "prophet_tuning_results.csv"), index=False)
    print("\n--- BEST PARAMS ---\n", res.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
