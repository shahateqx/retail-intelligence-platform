"""Workstream 3 — Cohort retention & hypothesis testing with power analysis.

Usage: python src/retention_tests.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "outputs")


def cohort_retention(df):
    first = df.groupby("CustomerID")["InvoiceMonth"].min().rename("Cohort")
    d = df.join(first, on="CustomerID")
    d["MonthIndex"] = (
        (d["InvoiceDate"].dt.year - d["Cohort"].str[:4].astype(int)) * 12
        + (d["InvoiceDate"].dt.month - d["Cohort"].str[5:7].astype(int))
    )
    coh = d.groupby(["Cohort", "MonthIndex"])["CustomerID"].nunique().unstack(fill_value=0)
    size = coh[0]
    ret = coh.div(size, axis=0).clip(upper=1)
    return ret, size


def proportions_test(succ_a, n_a, succ_b, n_b, alpha=0.05):
    z, p = proportions_ztest([succ_a, succ_b], [n_a, n_b], alternative="two-sided")
    es = proportion_effectsize(succ_a / n_a, succ_b / n_b)
    power = NormalIndPower().power(effect_size=abs(es), nobs1=n_a, alpha=alpha, ratio=n_b / n_a)
    return z, p, power


def main():
    df = pd.read_csv(os.path.join(PROC, "retail_clean.csv"), parse_dates=["InvoiceDate"])
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    ret, size = cohort_retention(df)
    ret.iloc[:, :7].to_csv(os.path.join(OUT, "cohort_retention.csv"))

    plt.figure(figsize=(10, 5))
    for i in range(min(6, ret.shape[1])):
        plt.plot(ret.columns[i], ret.iloc[:, i].mean(), marker="o", label=f"M{i}")
    plt.xlabel("Months since first purchase"); plt.ylabel("Average retention rate")
    plt.title("Average cohort retention curve"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "cohort_retention.png"), dpi=120); plt.close()

    # Leakage/censoring-safe comparison: 90-day repeat rate by FIRST-PURCHASE cohort.
    # Only include cohorts with >= 90 days of observation after acquisition.
    first = df.groupby("CustomerID")["InvoiceDate"].min().rename("First")
    d = df.join(first, on="CustomerID")
    d["DaysSinceFirst"] = (d["InvoiceDate"] - d["First"]).dt.days
    repeated = d[d["DaysSinceFirst"] > 0].groupby("CustomerID").size().rename("Repeats")
    coh = first.to_frame().join(repeated).fillna({"Repeats": 0})
    coh["Cohort"] = coh["First"].dt.to_period("M").astype(str)
    end = df["InvoiceDate"].max()
    eligible = coh[end - coh["First"] >= pd.Timedelta(days=90)]
    early = eligible[eligible["Cohort"].between("2009-12", "2010-05")]
    late = eligible[eligible["Cohort"].between("2010-09", "2011-02")]
    a, na = int((early["Repeats"] > 0).sum()), len(early)
    b, nb = int((late["Repeats"] > 0).sum()), len(late)
    z, p, power = proportions_test(a, na, b, nb)
    print("--- 90-DAY REPEAT-PURCHASE RATE (first-purchase cohorts) ---")
    print(f"Early cohorts (2009-12..2010-05): {a}/{na} = {a/na:.1%}")
    print(f"Late  cohorts (2010-09..2011-02): {b}/{nb} = {b/nb:.1%}")
    print(f"z = {z:.3f}, p = {p:.4g}, achieved power = {power:.2f}")
    print("Decision:", "reject H0 (rates differ)" if p < 0.05 else "fail to reject H0 (no significant difference)")
    print("\nRetention matrix (first 6 months):\n", ret.iloc[:, :6].round(3).to_string())


if __name__ == "__main__":
    main()
