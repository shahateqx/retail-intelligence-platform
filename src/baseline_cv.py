"""Compute the default Prophet configuration's mean CV RMSE for comparison with tuning."""
import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from prophet.diagnostics import cross_validation, performance_metrics
from forecasting import build_daily, make_model, PROC

df = pd.read_csv(os.path.join(PROC, "retail_clean.csv"), parse_dates=["InvoiceDate"])
daily = build_daily(df)
train = daily.iloc[:-31]
m = make_model().fit(train)
df_cv = cross_validation(m, horizon="31 days", initial=f"{len(train)-120} days")
pm = performance_metrics(df_cv)
print("Baseline mean CV RMSE:", round(pm['rmse'].mean(), 1))
print("Baseline mean CV MAE :", round(pm['mae'].mean(), 1))
