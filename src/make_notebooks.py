"""Generate narrative Jupyter notebooks for each workstream (run once)."""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "notebooks")
os.makedirs(NB, exist_ok=True)

SPECS = [
    ("01_data_cleaning", "Data engineering & quality audit", "clean_data.py",
     ["Source: UCI Online Retail II (1,067,371 real transactions, Dec 2009 – Dec 2011).",
      "Handle multi-sheet Excel, cancellations, negative quantities, missing customer IDs and duplicates.",
      "Derive Sales = Quantity × Price and save an analysis-ready table."]),
    ("02_segmentation", "Customer value & segmentation", "segmentation.py",
     ["RFM scoring with quartile ranks (Recency reversed: lower = better).",
      "Segments: Superstar / Future Champion / High Potential / Low Relevance.",
      "Gaussian Mixture with AIC/BIC model selection; Pareto revenue concentration."]),
    ("03_retention_tests", "Cohort retention & hypothesis testing", "retention_tests.py",
     ["Build monthly acquisition cohorts and retention curves.",
      "Compare repeat-purchase rates with a two-proportion z-test.",
      "Report achieved power so the test result is defensible."]),
    ("04_causal_impact", "Event impact — synthetic control", "causal_impact.py",
     ["Treated = top market daily revenue; controls = other markets.",
      "Check parallel trends (pre-correlation) and stationarity (ADF).",
      "OLS counterfactual, average/cumulative effect with 95% CI."]),
    ("05_forecasting", "Demand forecasting with Prophet", "forecasting.py",
     ["Daily revenue series, 31-day holdout, seasonal-naive baseline.",
      "Prophet with yearly/weekly seasonality; time-series cross-validation.",
      "Parameter grid tuned on RMSE; MAE used for stakeholder communication."]),
    ("06_high_value_model", "Predicting high-value customers", "high_value_model.py",
     ["Target = top 20% of customers by total spend.",
      "Random Forest with GridSearchCV (5-fold stratified CV, ROC-AUC).",
      "Feature importance translated into targeting rules."]),
]

for name, title, module, bullets in SPECS:
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n" + "\n".join(f"- {b}" for b in bullets))]
    cells.append(nbf.v4.new_code_cell(
        "import os, sys\n"
        "ROOT = os.path.abspath('..')\n"
        "sys.path.insert(0, os.path.join(ROOT, 'src'))\n"
        "print('Project root:', ROOT)"))
    cells.append(nbf.v4.new_code_cell(f"%run ../src/{module}"))
    cells.append(nbf.v4.new_markdown_cell(
        "## Deliverable\n\nReview the printed metrics and the figures saved in `outputs/`."))
    nb["cells"] = cells
    path = os.path.join(NB, name + ".ipynb")
    nbf.write(nb, path)
    print("Created", path)
