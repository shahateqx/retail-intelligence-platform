"""Clean UCI Online Retail II into an analysis-ready transaction table.

Usage: python src/clean_data.py
Output: data/processed/retail_clean.csv  +  console data-quality report
"""
import os
import glob
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROCESSED = os.path.join(ROOT, "data", "processed")
os.makedirs(PROCESSED, exist_ok=True)


def find_source():
    candidates = glob.glob(os.path.join(RAW, "*.xlsx")) + glob.glob(os.path.join(RAW, "*.xls"))
    if not candidates:
        raise FileNotFoundError("No .xlsx found in data/raw — run src/download_data.py first")
    return max(candidates, key=os.path.getsize)


def main():
    src = find_source()
    print("Reading:", os.path.basename(src), flush=True)
    sheets = pd.read_excel(src, sheet_name=None)
    print("Sheets:", list(sheets), flush=True)
    df = pd.concat(sheets.values(), ignore_index=True)

    # Standardise column names across UCI revisions
    df.columns = [c.strip().replace(" ", "") for c in df.columns]
    rename = {"UnitPrice": "Price", "CustomerID": "CustomerID", "InvoiceDate": "InvoiceDate"}
    df = df.rename(columns=rename)
    need = {"Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "CustomerID", "Country"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Got: {list(df.columns)}")

    n0 = len(df)
    report = {}
    report["raw_rows"] = n0

    # Parse dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Remove cancellations / non-product lines / invalid prices & quantities
    inv = df["Invoice"].astype(str)
    mask_cancel = inv.str.startswith("C", na=False)
    mask_bad = (
        df["Quantity"].isna() | (df["Quantity"] <= 0)
        | df["Price"].isna() | (df["Price"] <= 0)
        | df["InvoiceDate"].isna()
        | df["Description"].isna()
    )
    report["cancellations_removed"] = int(mask_cancel.sum())
    df = df[~mask_cancel & ~mask_bad].copy()

    # Customer-level work needs an identified customer
    report["missing_customer_removed"] = int(df["CustomerID"].isna().sum())
    df = df[df["CustomerID"].notna()].copy()
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)

    # Exact duplicates
    dups = df.duplicated()
    report["duplicates_removed"] = int(dups.sum())
    df = df[~dups].copy()

    # Trim descriptions and format stock codes
    df["Description"] = df["Description"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Country"] = df["Country"].astype(str).str.strip()

    # Derived fields
    df["Sales"] = (df["Quantity"] * df["Price"]).round(2)
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["Date"] = df["InvoiceDate"].dt.date

    df = df.sort_values("InvoiceDate").reset_index(drop=True)
    out = os.path.join(PROCESSED, "retail_clean.csv")
    df.to_csv(out, index=False)

    report["clean_rows"] = len(df)
    report["retained_pct"] = round(100 * len(df) / n0, 1)
    report["customers"] = df["CustomerID"].nunique()
    report["countries"] = df["Country"].nunique()
    report["products"] = df["StockCode"].nunique()
    report["invoices"] = df["Invoice"].nunique()
    report["date_min"] = str(df["InvoiceDate"].min().date())
    report["date_max"] = str(df["InvoiceDate"].max().date())
    report["total_revenue"] = round(df["Sales"].sum(), 2)

    print("\n--- DATA QUALITY REPORT ---")
    for k, v in report.items():
        print(f"{k:>26}: {v:,}" if isinstance(v, (int, float)) else f"{k:>26}: {v}")
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
