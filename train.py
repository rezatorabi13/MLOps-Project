"""
Module 1 — Experiment Tracking with MLflow
===========================================

Goal: train a simple classifier and record EVERYTHING about the run
(parameters, metrics, and the trained model itself) into MLflow, so the
experiment is reproducible and comparable against future runs.

Run it like this (from the project folder):
    .venv/Scripts/python.exe train.py
    .venv/Scripts/python.exe train.py --n-estimators 300 --max-depth 5

Then open the MLflow UI to compare runs:
    .venv/Scripts/mlflow.exe ui
"""

import argparse

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def main():
    # ----------------------------------------------------------------------
    # 1. Hyperparameters as command-line args.
    #    In real MLOps you rarely hard-code these — you sweep over them and
    #    let MLflow record which combination performed best.
    # ----------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="Number of trees in the random forest.")
    parser.add_argument("--max-depth", type=int, default=3,
                        help="Max depth of each tree.")
    args = parser.parse_args()

    # ----------------------------------------------------------------------
    # 2. Load data and split into train/test.
    #    The Wine dataset: 178 samples, 13 chemical features, 3 wine classes.
    #    A small, fast, classic dataset — perfect for learning the mechanics.
    # ----------------------------------------------------------------------
    X, y = load_wine(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ----------------------------------------------------------------------
    # 3. Tell MLflow which "experiment" (a named group of runs) to log to.
    #    A run = one execution of this script. Each run lives under an
    #    experiment so you can compare related runs side by side.
    # ----------------------------------------------------------------------
    mlflow.set_experiment("wine-classifier")

    with mlflow.start_run() as run:
        # --- log the inputs (parameters) ---
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        # --- train ---
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # --- evaluate ---
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        # --- log the outputs (metrics) ---
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        # --- log the trained model itself as an "artifact" ---
        #     This is what makes the run reproducible: the exact model that
        #     produced these metrics is saved and versioned alongside them.
        mlflow.sklearn.log_model(model, name="model")

        print(f"Run ID:    {run.info.run_id}")
        print(f"accuracy:  {acc:.4f}")
        print(f"f1_macro:  {f1:.4f}")
        print("Logged to MLflow experiment 'wine-classifier'.")


if __name__ == "__main__":
    main()
