# =============================================================================
# Dockerfile — the recipe to package the Wine Classifier API into an image.
# Each line is a STEP; Docker caches each step as a "layer" so unchanged steps
# are reused on the next build (this is why we install deps before copying code).
# =============================================================================

# 1. Base image: an official, slim Python 3.13 on Linux. Everything builds
#    on top of this. "slim" = smaller (no extra OS tools we don't need).
FROM python:3.13-slim

# 2. Set the working directory inside the container. All later paths are
#    relative to /app, and we'll end up there when the container starts.
WORKDIR /app

# 3. Copy ONLY requirements first, then install. Because this layer only
#    changes when requirements.txt changes, Docker reuses the cached install
#    on future builds where you edited code but not dependencies. Big speedup.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the application code and the portable model into the image.
#    Both the API (app.py) and the Kafka consumer (consumer.py) ship in the
#    same image; the docker-compose `command:` decides which one runs.
COPY app.py .
COPY consumer.py .
COPY serving_model/ ./serving_model/

# 5. Configuration via environment variables (12-factor style). This tells
#    app.py to load the baked-in model folder instead of the MLflow registry.
ENV MODEL_URI=/app/serving_model
ENV MODEL_VERSION=1

# 6. Document that the app listens on port 8000 (informational).
EXPOSE 8000

# 7. The command run when the container starts. Note host 0.0.0.0 (NOT
#    127.0.0.1) so the server is reachable from OUTSIDE the container.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
