"""
Module 2 — Serving the model as a REST API (FastAPI)
====================================================

This turns the registered model into an always-on network service.

Key MLOps ideas demonstrated:
  * Load the model ONCE at startup (not per request) — loading is slow,
    inference is fast. This is the single biggest serving performance rule.
  * Load it by registry ALIAS (champion), so deploying a new model = moving
    the alias, with no change to this file.
  * Validate incoming requests with a schema (Pydantic) so bad input is
    rejected with a clear 422 error instead of crashing the model.
  * Expose /health for load balancers and Kubernetes probes (Module 3).

Run:
    .venv/Scripts/uvicorn.exe app:app --host 0.0.0.0 --port 8000
Interactive docs auto-generated at:  http://localhost:8000/docs
"""

import os
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI
from mlflow import MlflowClient
from pydantic import BaseModel, Field

MODEL_NAME = "wine-classifier"
ALIAS = "champion"

# WHERE to load the model from, configurable via an environment variable.
#   * Locally (no env var set): default to the MLflow registry alias.
#   * In the container: the Dockerfile sets MODEL_URI=/app/serving_model,
#     so it loads the portable, baked-in copy instead.
# This is "config via environment" — the same image runs in any environment
# just by changing env vars, with no code changes.
MODEL_URI = os.getenv("MODEL_URI", f"models:/{MODEL_NAME}@{ALIAS}")

# Human-readable names for the 3 wine cultivars (the target classes).
CLASS_NAMES = ["class_0", "class_1", "class_2"]

# A place to stash things loaded at startup (the model + its metadata).
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs ONCE when the server boots: load the model from MODEL_URI."""
    state["model"] = mlflow.sklearn.load_model(MODEL_URI)
    # The exact column names/order the model was trained on. We rebuild
    # incoming requests into this shape so sklearn is happy.
    state["features"] = list(state["model"].feature_names_in_)

    # Record which version we're serving (for traceability in responses).
    # Prefer an explicit env var (set in the container); otherwise, if we
    # loaded via a registry alias, look the version up from the registry.
    version = os.getenv("MODEL_VERSION")
    if version is None and MODEL_URI.startswith("models:/") and "@" in MODEL_URI:
        version = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS).version
    state["version"] = version

    print(f"Loaded model from {MODEL_URI} (version {version})")
    yield
    state.clear()  # runs on shutdown


app = FastAPI(title="Wine Classifier API", version="1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request schema. Field names are clean identifiers; the example values are a
# real wine sample so the /docs "Try it out" button works out of the box.
# Order here MUST match the training feature order.
# ---------------------------------------------------------------------------
class WineFeatures(BaseModel):
    alcohol: float = Field(..., example=14.23)
    malic_acid: float = Field(..., example=1.71)
    ash: float = Field(..., example=2.43)
    alcalinity_of_ash: float = Field(..., example=15.6)
    magnesium: float = Field(..., example=127.0)
    total_phenols: float = Field(..., example=2.80)
    flavanoids: float = Field(..., example=3.06)
    nonflavanoid_phenols: float = Field(..., example=0.28)
    proanthocyanins: float = Field(..., example=2.29)
    color_intensity: float = Field(..., example=5.64)
    hue: float = Field(..., example=1.04)
    od280_od315: float = Field(..., example=3.92)
    proline: float = Field(..., example=1065.0)


@app.get("/health")
def health():
    """Liveness/readiness check. K8s and load balancers hit this."""
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "alias": ALIAS,
        "version": state.get("version"),
    }


@app.post("/predict")
def predict(features: WineFeatures):
    """Classify one wine sample and return the class + probabilities."""
    # Pydantic gives us a clean dict; reorder the VALUES into the exact
    # column order/names the model expects, then make a 1-row DataFrame.
    values = list(features.model_dump().values())
    row = pd.DataFrame([values], columns=state["features"])

    model = state["model"]
    pred_idx = int(model.predict(row)[0])
    probs = model.predict_proba(row)[0]

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "predicted_index": pred_idx,
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": {
            CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)
        },
        "model_version": state["version"],
    }
