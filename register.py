"""
Module 1.5 — The Model Registry
===============================

Experiment tracking (Module 1) answers "which run was best?".
The Model Registry answers "which model is the OFFICIAL one we serve?".

It is a versioned catalog of models, separate from the experiments. You
register a run's model into the catalog, it gets a version number (1, 2,
3...), and you attach an ALIAS like "champion" to point at whichever
version is currently the blessed one. Serving code then loads
"models:/wine-classifier@champion" and never has to know run IDs.

This decouples experimentation from serving — promoting a new model is
just moving the alias, with zero code changes downstream.

Run:
    .venv/Scripts/python.exe register.py
"""

import mlflow
from mlflow import MlflowClient

EXPERIMENT = "wine-classifier"
MODEL_NAME = "wine-classifier"   # the name it will have IN the registry
ALIAS = "champion"               # the pointer serving code will use

client = MlflowClient()

# ----------------------------------------------------------------------
# 1. Find the best run by accuracy (tie-break on f1). search_runs returns
#    a pandas DataFrame, already sorted by the order_by clause.
# ----------------------------------------------------------------------
runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT],
    order_by=["metrics.accuracy DESC", "metrics.f1_macro DESC"],
)
best = runs.iloc[0]
best_run_id = best["run_id"]
print(f"Best run: {best_run_id}")
print(f"  accuracy = {best['metrics.accuracy']:.4f}")
print(f"  f1_macro = {best['metrics.f1_macro']:.4f}")

# ----------------------------------------------------------------------
# 2. Register that run's model artifact into the registry. Each call
#    creates a NEW version under MODEL_NAME (v1 the first time, v2 next...).
# ----------------------------------------------------------------------
model_uri = f"runs:/{best_run_id}/model"
mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
print(f"Registered as '{MODEL_NAME}' version {mv.version}")

# ----------------------------------------------------------------------
# 3. Point the 'champion' alias at this new version. Aliases are the
#    modern MLflow way (they replaced the old Staging/Production stages).
# ----------------------------------------------------------------------
client.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)
print(f"Alias '{ALIAS}' -> version {mv.version}")

# ----------------------------------------------------------------------
# 4. Prove it: load the model purely by its alias and run a prediction.
#    This is exactly how the serving API (Module 2) will load the model.
# ----------------------------------------------------------------------
from sklearn.datasets import load_wine

loaded = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
X, y = load_wine(return_X_y=True, as_frame=True)
sample = X.iloc[[0]]
pred = loaded.predict(sample)[0]
print(f"\nLoaded 'models:/{MODEL_NAME}@{ALIAS}' and predicted class {pred} "
      f"(actual {y.iloc[0]}) for the first wine sample.")
