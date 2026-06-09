"""
export_model.py — package the champion model into a portable folder
===================================================================

Why this exists:
The model registered in MLflow points at an ABSOLUTE path on THIS Windows
machine (file:C:/Users/.../mlruns/...). A Linux container has no such path,
so loading it by registry alias would fail inside the container.

The fix (and the normal production pattern): take the chosen model and
re-save it as a SELF-CONTAINED, relative-path MLflow model directory. That
folder has no absolute paths, so it works anywhere — we bake it into the image.

Run:
    .venv/Scripts/python.exe export_model.py
"""

import shutil

import mlflow

MODEL_NAME = "wine-classifier"
ALIAS = "champion"
OUT_DIR = "serving_model"

# 1. Load the champion model from the registry (works here on Windows).
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")

# 2. Re-save it to a clean, portable folder (overwrite if it exists).
shutil.rmtree(OUT_DIR, ignore_errors=True)
mlflow.sklearn.save_model(model, OUT_DIR)

print(f"Exported {MODEL_NAME}@{ALIAS} -> ./{OUT_DIR}/  (portable, no absolute paths)")
