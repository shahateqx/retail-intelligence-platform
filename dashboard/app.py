"""Streamlit dashboard for the Retail Intelligence Platform.

Run: streamlit run dashboard/app.py
"""
import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "outputs")

st.set_page_config(page_title="Retail Intelligence Platform", layout="wide")
st.title("Retail Intelligence Platform")
st.caption("UCI Online Retail II — real transactions, Dec 2009 – Dec 2011")

feats = pd.read_csv(os.path.join(PROC, "customer_features.csv"))
daily = pd.read_csv(os.path.join(PROC, "daily_revenue.csv"), parse_dates=["ds"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{feats.CustomerID.nunique():,}")
c2.metric("Total revenue", f"£{feats.TotalSpend.sum():,.0f}")
c3.metric("Avg basket", f"£{feats.Monetary.mean():,.1f}")
c4.metric("Days of data", f"{len(daily):,}")

st.subheader("Customer segments")
seg = feats.groupby("Segment").agg(
    Customers=("CustomerID", "count"), Revenue=("TotalSpend", "sum"),
    AvgBasket=("Monetary", "mean"), AvgRecency=("Recency", "mean"),
).round(1)
st.dataframe(seg, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Revenue by segment")
    st.bar_chart(seg["Revenue"])
with right:
    st.subheader("Daily revenue")
    st.line_chart(daily.set_index("ds")["y"].resample("MS").sum())

st.subheader("Segment explorer")
choice = st.selectbox("Segment", sorted(feats.Segment.unique()))
st.dataframe(feats[feats.Segment == choice].sort_values("TotalSpend", ascending=False).head(20),
             use_container_width=True)

if os.path.exists(os.path.join(OUT, "causal_impact_results.csv")):
    st.subheader("Event impact")
    st.dataframe(pd.read_csv(os.path.join(OUT, "causal_impact_results.csv")).tail(10), use_container_width=True)

if os.path.exists(os.path.join(OUT, "prophet_tuning_results.csv")):
    st.subheader("Forecast tuning results")
    st.dataframe(pd.read_csv(os.path.join(OUT, "prophet_tuning_results.csv")), use_container_width=True)
