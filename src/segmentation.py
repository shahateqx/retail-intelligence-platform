"""Workstream 2 — Customer value & segmentation (RFM + Gaussian Mixture).

Usage: python src/segmentation.py
Inputs : data/processed/retail_clean.csv
Outputs: data/processed/customer_features.csv, outputs/*.png, outputs/segment_*.csv
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)


def rfm_score_frame(df):
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    agg = df.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("Invoice", "nunique"),
        TotalSpend=("Sales", "sum"),
    )
    agg["Recency"] = (snapshot - agg["LastPurchase"]).dt.days
    agg["Monetary"] = agg["TotalSpend"] / agg["Frequency"]  # average basket (course preference)
    agg["R"] = pd.qcut(agg["Recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    agg["F"] = pd.qcut(agg["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    agg["M"] = pd.qcut(agg["Monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    agg["RFM"] = agg[["R", "F", "M"]].sum(axis=1)

    def label(s):
        if s >= 11:
            return "Superstar"
        if s >= 8:
            return "Future Champion"
        if s >= 6:
            return "High Potential"
        return "Low Relevance"

    agg["Segment"] = agg["RFM"].apply(label)
    return agg.reset_index()


def pareto_table(agg):
    s = agg.sort_values("TotalSpend", ascending=False).reset_index(drop=True)
    s["CumRevenuePct"] = 100 * s["TotalSpend"].cumsum() / s["TotalSpend"].sum()
    s["CumCustomerPct"] = 100 * (s.index + 1) / len(s)
    rows = []
    for pct in [1, 5, 10, 20, 50]:
        n = int(np.ceil(len(s) * pct / 100))
        rows.append({"TopCustomerPct": pct, "RevenuePct": round(s.loc[n - 1, "CumRevenuePct"], 1)})
    return pd.DataFrame(rows)


def gmm_segmentation(agg):
    X = agg[["Recency", "Frequency", "Monetary"]].copy()
    X["Monetary"] = np.log1p(X["Monetary"].clip(lower=0))
    Xs = StandardScaler().fit_transform(X)

    ks = range(1, 9)
    aics, bics = [], []
    for k in ks:
        gm = GaussianMixture(n_components=k, random_state=1502, n_init=3).fit(Xs)
        aics.append(gm.aic(Xs))
        bics.append(gm.bic(Xs))

    plt.figure(figsize=(8, 4))
    plt.plot(list(ks), aics, marker="o", label="AIC")
    plt.plot(list(ks), bics, marker="o", label="BIC")
    plt.xlabel("Number of clusters"); plt.ylabel("Information criterion"); plt.legend()
    plt.title("Gaussian Mixture — model selection")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "gmm_aic_bic.png"), dpi=120); plt.close()

    # Favour simplicity: smallest k within 2% of the best BIC
    bics = np.array(bics)
    best = bics.min()
    k = next(k for k, b in zip(ks, bics) if b <= best * 1.02)
    gm = GaussianMixture(n_components=k, random_state=1502, n_init=5).fit(Xs)
    agg["GMM_Cluster"] = gm.predict(Xs)
    return agg, k, pd.DataFrame({"k": list(ks), "AIC": aics, "BIC": bics})


def main():
    df = pd.read_csv(os.path.join(PROC, "retail_clean.csv"), parse_dates=["InvoiceDate"])
    print("Transactions loaded:", len(df), flush=True)

    agg = rfm_score_frame(df)
    pareto = pareto_table(agg)
    agg, k, criteria = gmm_segmentation(agg)
    agg.to_csv(os.path.join(PROC, "customer_features.csv"), index=False)
    pareto.to_csv(os.path.join(OUT, "pareto.csv"), index=False)
    criteria.to_csv(os.path.join(OUT, "gmm_criteria.csv"), index=False)

    seg = agg.groupby("Segment").agg(
        Customers=("CustomerID", "count"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgBasket=("Monetary", "mean"),
        TotalRevenue=("TotalSpend", "sum"),
    ).round(1)
    seg["RevenueSharePct"] = (100 * seg["TotalRevenue"] / seg["TotalRevenue"].sum()).round(1)
    seg.to_csv(os.path.join(OUT, "rfm_segment_summary.csv"))

    clus = agg.groupby("GMM_Cluster").agg(
        Customers=("CustomerID", "count"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgBasket=("Monetary", "mean"),
        TotalRevenue=("TotalSpend", "sum"),
    ).round(1)
    clus.to_csv(os.path.join(OUT, "gmm_cluster_summary.csv"))

    print("\n--- PARETO ---\n", pareto.to_string(index=False))
    print("\n--- RFM SEGMENTS ---\n", seg.to_string())
    print(f"\n--- GMM (selected k={k}) ---\n", clus.to_string())
    print("\nSaved to outputs/ and data/processed/")


if __name__ == "__main__":
    main()
