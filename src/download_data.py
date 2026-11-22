"""Download the UCI Online Retail II dataset (real transactions, 2009-2011).

Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii
Licence: CC BY 4.0 (UCI Machine Learning Repository)
"""
import os
import sys
import zipfile
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW, exist_ok=True)

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ZPATH = os.path.join(RAW, "online_retail_ii.zip")


def main():
    if not os.path.exists(ZPATH):
        print("Downloading", URL, flush=True)
        with requests.get(URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(ZPATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
    print("zip size MB:", round(os.path.getsize(ZPATH) / 1e6, 1), flush=True)

    with zipfile.ZipFile(ZPATH) as z:
        print("contents:", z.namelist(), flush=True)
        z.extractall(RAW)
    for name in os.listdir(RAW):
        p = os.path.join(RAW, name)
        print(name, round(os.path.getsize(p) / 1e6, 1), "MB", flush=True)


if __name__ == "__main__":
    main()
