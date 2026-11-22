# Deploy to GitHub + Streamlit Cloud

Everything in this repo is ready. You only need a **GitHub account** and a **Streamlit Cloud
account** (sign up with GitHub at https://share.streamlit.io). No API keys or secrets are required.

---

## 1. Push the project to GitHub (you do this)

Open a terminal **in the project folder**:

```bash
cd "D:\ZeroToMastery - Course Collection. 128 Courses (2025)\ZeroToMastery - Business Analytics Bootcamp (with Python) Zero to Mastery (4.2025)\_portfolio\retail-intelligence"

git init
git add .
git commit -m "Retail Intelligence Platform: end-to-end analytics portfolio project"
```

Then create a **new public repository** on https://github.com/new (name it e.g.
`retail-intelligence-platform`, **do not** initialise with README), and run:

```bash
git remote add origin https://github.com/YOUR_USERNAME/retail-intelligence-platform.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` and the repo name with yours.

> The large raw data is gitignored on purpose. The small processed tables in `data/processed/`
> are committed, so the deployed dashboard works immediately without downloading 43 MB of data.
> Check on GitHub that `customer_features.csv` and `daily_revenue.csv` are present.

## 2. Deploy on Streamlit Cloud (you do this)

1. Go to https://share.streamlit.io → **Sign in with GitHub**.
2. Click **New app**.
3. Fill in:
   - **Repository**: `YOUR_USERNAME/retail-intelligence-platform`
   - **Branch**: `main`
   - **Main file path**: `dashboard/app.py`
4. Click **Advanced settings** → Python version: **3.11 or 3.12** (Prophet has wheels for these;
   3.13 may fail). Leave secrets empty.
5. Click **Deploy**.

First build takes a few minutes because `requirements.txt` installs Prophet. Watch the logs — a
successful build ends with `Uvicorn server started`.

## 3. Put the live URL in the README (you do this)

Once deployed, copy the URL (e.g. `https://retail-intelligence-platform.streamlit.app`) and replace
the placeholder line near the top of `README.md`:

```
> **Live dashboard:** https://YOUR_APP.streamlit.app
```

Then commit and push:

```bash
git add README.md
git commit -m "Add live dashboard URL"
git push
```

---

## What I already did for you

- `git`-ready repo structure, `.gitignore`, `runtime.txt`, `.streamlit/config.toml` (theme + no usage stats)
- `requirements.txt` with every dependency
- `README.md` with what I did, what I found, and a couple of charts
- `reports/business_memo.md` with the full written analysis
- Small processed tables committed so the Cloud app runs without raw data

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` in logs | Confirm `requirements.txt` is in the repo root and redeploy via "Reboot app" |
| Build fails on Prophet | Set Python 3.11/3.12 in Advanced settings, then reboot |
| Dashboard shows empty charts | Confirm `data/processed/customer_features.csv` and `daily_revenue.csv` are in the repo |
| App is slow to start | Normal on first deploy (Prophet install); subsequent starts are cached |
| Repo size warning | Never commit `data/raw/` or `data/processed/retail_clean.csv` — they are regenerated locally |

## Cost

Streamlit Cloud's public-app tier is free. The repo must be public (or on a paid plan for private repos).
