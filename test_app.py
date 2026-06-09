"""
test_app.py — automated tests for the Wine Classifier API
=========================================================

These are the "quality gate": code that checks the API works, runnable in
one command. The CI robot (GitHub Actions) will run this on every push.

Run locally:
    .venv/Scripts/python.exe -m pytest -v

IMPORTANT design choice — tests must be SELF-CONTAINED:
A test should not depend on external services (the MLflow registry, a
database, the network). So BEFORE importing the app, we point it at the
portable `serving_model/` folder via the MODEL_URI env var. That folder is
committed to git, so these tests run identically on your laptop and on the
CI server, which has no registry.
"""

import os

# Must be set BEFORE importing app, because app.py reads MODEL_URI at import.
os.environ.setdefault("MODEL_URI", "serving_model")
os.environ.setdefault("MODEL_VERSION", "1")

import pytest
from fastapi.testclient import TestClient

from app import app

# A valid wine sample (the same one used in the API docs example).
VALID_WINE = {
    "alcohol": 14.23, "malic_acid": 1.71, "ash": 2.43,
    "alcalinity_of_ash": 15.6, "magnesium": 127.0, "total_phenols": 2.80,
    "flavanoids": 3.06, "nonflavanoid_phenols": 0.28, "proanthocyanins": 2.29,
    "color_intensity": 5.64, "hue": 1.04, "od280_od315": 3.92, "proline": 1065.0,
}


@pytest.fixture(scope="module")
def client():
    """A test client. The 'with' block triggers the app's startup (lifespan),
    so the model is loaded — exactly like a real server boot."""
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    """The health endpoint should report the service is alive."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_input(client):
    """A well-formed request should return a prediction with sane values."""
    response = client.post("/predict", json=VALID_WINE)
    assert response.status_code == 200

    body = response.json()
    # The predicted class must be one of the three known wine classes.
    assert body["predicted_class"] in {"class_0", "class_1", "class_2"}
    # Confidence is a probability, so it must be between 0 and 1.
    assert 0.0 <= body["confidence"] <= 1.0
    # The three class probabilities should sum to ~1.0 (allow rounding error).
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01


def test_predict_rejects_bad_input(client):
    """Missing/invalid fields must be rejected with 422 (the input guard),
    NOT crash the model."""
    bad_request = {"alcohol": "not_a_number"}  # wrong type + missing fields
    response = client.post("/predict", json=bad_request)
    assert response.status_code == 422
