# Wine Classifier — End-to-End MLOps Pipeline

A hands-on MLOps project that takes a single scikit-learn model (wine
classification) and wraps it in a complete production lifecycle: experiment
tracking, a model registry, REST + streaming serving, containerization,
CI/CD, observability, alerting, and Kubernetes deployment.

The point isn't the model — it's everything **around** the model that turns it
into a reliable, deployable, observable service.

---

## What's included (the MLOps stack)

| Area | Tooling | What it does |
|------|---------|--------------|
| **Experiment tracking** | MLflow | Logs runs, params, metrics for the model (`train.py`) |
| **Model registry** | MLflow Registry | Versions the model + a `champion` alias (`register.py`) |
| **Serving — sync** | FastAPI + Uvicorn | REST API with `/predict` + `/health` (`app.py`) |
| **Serving — async** | Kafka | Streaming inference via a consumer (`consumer.py`) |
| **Containerization** | Docker | One image, portable model baked in (`Dockerfile`) |
| **Orchestration (local)** | Docker Compose | Runs the whole stack together (`docker-compose.yml`) |
| **CI/CD** | GitHub Actions | Runs tests on every push (`.github/workflows/ci.yml`) |
| **Observability** | Prometheus + Grafana | Scrapes `/metrics`, dashboards (`prometheus.yml`, `grafana/`) |
| **SRE / alerting** | Prometheus rules | SLO-based alerts: availability, error rate, latency (`alert.rules.yml`) |
| **Deployment** | Kubernetes | Deployment + Service with health probes & scaling (`k8s/`) |

---

## Architecture

```
                 ┌──────────── sync ────────────┐
   client ──HTTP──▶  FastAPI API  ──┐            │
                                    ├─ serving_model/ (same model)
   producer ──▶ Kafka ──▶ consumer ─┘            │
                 └──────────── async ───────────┘
                          │
                  /metrics │ scraped
                          ▼
                   Prometheus ──▶ Grafana (dashboards)
                       │
                   alert rules (SLOs)

   Deployable to Kubernetes: Deployment (2 replicas) + Service + /health probes
```

The **same model** is served two ways: **synchronous** REST (wait for the
answer) and **asynchronous** Kafka streaming (fire-and-forget, result on a topic).

---

## Repository layout

```
train.py            Train the model, log to MLflow
register.py         Register the best run + set the champion alias
export_model.py     Export the model to a portable serving_model/ folder
app.py              FastAPI REST API (sync serving) + Prometheus /metrics
consumer.py         Kafka consumer (async streaming inference)
producer.py         Test driver: publishes wine samples to Kafka
test_app.py         Pytest suite (run by CI)
inspect_db.py       Peek into the MLflow registry SQLite DB

Dockerfile          Builds the serving image
docker-compose.yml  api + consumer + kafka + prometheus + grafana
prometheus.yml      Prometheus scrape config
alert.rules.yml     SRE alerting rules (SLOs)
grafana/            Auto-provisioned Prometheus datasource
k8s/                Kubernetes Deployment + Service
.github/workflows/  GitHub Actions CI
requirements.txt        Runtime dependencies
requirements-dev.txt    Test/dev dependencies (pytest, httpx)
```

---

## Quick start

### 1. Run the full stack (Docker Compose)
```bash
docker compose up -d --build
```
| Service | URL |
|---------|-----|
| API docs | http://localhost:8000/docs |
| Metrics | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

### 2. Try the async (Kafka) path
```bash
python producer.py                       # send wine samples
docker compose logs -f consumer          # watch predictions
```

### 3. Run the tests
```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -v
```

### 4. Deploy to Kubernetes
```bash
kubectl apply -f k8s/
kubectl get pods
kubectl port-forward service/wine-api 8080:8000   # then open http://localhost:8080/docs
```

---

## Key design choices

- **Load the model once at startup**, not per request — serving performance 101.
- **Config via environment variables** (`MODEL_URI`, etc.) — the same image runs
  anywhere with no code change.
- **Portable `serving_model/` folder** committed to the repo, so tests, the
  container, and Kubernetes all load the model with no external registry or
  network dependency.
- **Split runtime vs dev dependencies** — the production image stays lean.
- **Health probes on `/health`** — reused by both the load balancer and
  Kubernetes liveness/readiness checks.

---

## Tech stack

Python · scikit-learn · MLflow · FastAPI · Uvicorn · Pydantic · Kafka ·
Prometheus · Grafana · Docker · Docker Compose · Kubernetes · GitHub Actions ·
pytest
