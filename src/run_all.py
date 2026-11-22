"""Run the full reproducible pipeline in order: python src/run_all.py"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
STEPS = [
    "clean_data.py",
    "segmentation.py",
    "retention_tests.py",
    "causal_impact.py",
    "high_value_model.py",
    "forecasting.py",
]

for step in STEPS:
    print(f"\n{'='*70}\nRUNNING {step}\n{'='*70}", flush=True)
    subprocess.run([sys.executable, os.path.join(SRC, step)], check=True, cwd=ROOT)
print("\nPipeline complete.")
