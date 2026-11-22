"""Workstream 6 — Predict FUTURE high-value customers (Random Forest + GridSearchCV).

Leakage-free design: features are computed from history only (before CUT); the target is
high spend in the *subsequent* 6 months. Usage: python src/high_value_model.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "outputs")
CUT = pd.Timestamp("2011-06-01")       # features before this date, target after it
FEATURES = ["Recency", "Frequency", "Monetary", "TotalSpend", "R", "F", "M"]


def build_history_features(df):
    h = df[df["InvoiceDate"] < CUT]
    g = h.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("Invoice", "nunique"),
        TotalSpend=("Sales", "sum"),
    )
    g["Recency"] = (CUT - g["LastPurchase"]).dt.days
    g["Monetary"] = g["TotalSpend"] / g["Frequency"]
    g["R"] = pd.qcut(g["Recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    g["F"] = pd.qcut(g["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    g["M"] = pd.qcut(g["Monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    return g


def main():
    df = pd.read_csv(os.path.join(PROC, "retail_clean.csv"), parse_dates=["InvoiceDate"])
    hist = build_history_features(df)
    future_spend = (
        df[df["InvoiceDate"] >= CUT].groupby("CustomerID")["Sales"].sum()
        .reindex(hist.index).fillna(0)
    )
    threshold = future_spend.quantile(0.8)
    y = (future_spend >= threshold).astype(int)
    X = hist[FEATURES]
    print(f"Prediction window: after {CUT.date()} | customers: {len(hist)} | "
          f"high-value prevalence: {y.mean():.1%} | spend threshold: £{threshold:,.0f}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1502, stratify=y)

    grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 3],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1502)
    search = GridSearchCV(
        RandomForestClassifier(random_state=1502, class_weight="balanced"),
        grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0,
    ).fit(Xtr, ytr)

    best = search.best_estimator_
    proba = best.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print("--- FUTURE HIGH-VALUE CUSTOMER MODEL (no leakage) ---")
    print("Best parameters:", search.best_params_)
    print(f"ROC-AUC: {roc_auc_score(yte, proba):.3f}   PR-AUC: {average_precision_score(yte, proba):.3f}")
    print(classification_report(yte, pred, digits=3))

    imp = pd.Series(best.feature_importances_, index=FEATURES).sort_values()
    imp.to_csv(os.path.join(OUT, "high_value_feature_importance.csv"), header=["importance"])
    plt.figure(figsize=(7, 4)); imp.plot(kind="barh")
    plt.title("Feature importance — future high-value customers"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "high_value_importance.png"), dpi=120); plt.close()
    print("\nFeature importance:\n", imp.to_string())


if __name__ == "__main__":
    main()
