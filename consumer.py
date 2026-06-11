"""
consumer.py — Module 6: streaming inference with Kafka
======================================================

This is the ASYNCHRONOUS counterpart to app.py's synchronous REST /predict.

Flow:
    wine-requests topic  ──▶  [this consumer: load model, predict]  ──▶  wine-predictions topic

It runs as its own long-lived container (the `consumer` service in
docker-compose). Like the API, it loads the model ONCE at startup, then
loops forever processing messages as they arrive.

Key streaming ideas demonstrated:
  * Decoupling — producers don't wait for us; they drop messages on the
    topic and move on. We process at our own pace.
  * Consumer group — `group_id` lets Kafka split a topic's partitions across
    multiple copies of this consumer for parallel throughput.
  * Durability — if this service crashes and restarts, Kafka remembers our
    position (offset) so we resume without losing or re-processing messages.
"""

import json
import os
import time

import mlflow
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable

# --- Config via environment (12-factor), with sensible in-container defaults --
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
REQUESTS_TOPIC = os.getenv("REQUESTS_TOPIC", "wine-requests")
PREDICTIONS_TOPIC = os.getenv("PREDICTIONS_TOPIC", "wine-predictions")
MODEL_URI = os.getenv("MODEL_URI", "serving_model")

CLASS_NAMES = ["class_0", "class_1", "class_2"]


def connect(bootstrap: str):
    """Kafka may still be booting when we start (depends_on only waits for the
    container to START, not to be READY). Retry until the broker answers."""
    for attempt in range(1, 31):
        try:
            consumer = KafkaConsumer(
                REQUESTS_TOPIC,
                bootstrap_servers=bootstrap,
                group_id="wine-classifier",          # the consumer group
                auto_offset_reset="earliest",        # read from the start if no saved offset
                value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            )
            producer = KafkaProducer(
                bootstrap_servers=bootstrap,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            return consumer, producer
        except NoBrokersAvailable:
            print(f"[{attempt}/30] Kafka not ready at {bootstrap}, retrying in 2s...")
            time.sleep(2)
    raise RuntimeError(f"Could not reach Kafka at {bootstrap} after many retries")


def main():
    # 1. Load the model once (same performance rule as the API).
    print(f"Loading model from {MODEL_URI} ...", flush=True)
    model = mlflow.sklearn.load_model(MODEL_URI)
    features = list(model.feature_names_in_)  # exact training column order
    print("Model loaded.", flush=True)

    # 2. Connect to Kafka (with retry).
    consumer, producer = connect(KAFKA_BOOTSTRAP)
    print(f"Listening on '{REQUESTS_TOPIC}' ... (Ctrl+C to stop)", flush=True)

    # 3. Loop forever: each iteration blocks until a new message arrives.
    for message in consumer:
        wine = message.value  # a dict of the 13 features, in training order
        # Reorder VALUES into the model's expected columns (mirrors app.py).
        row = pd.DataFrame([list(wine.values())], columns=features)

        idx = int(model.predict(row)[0])
        probs = model.predict_proba(row)[0]
        result = {
            "predicted_class": CLASS_NAMES[idx],
            "confidence": round(float(probs[idx]), 4),
            "input": wine,
        }

        # 4. Publish the prediction to the output topic.
        producer.send(PREDICTIONS_TOPIC, result)
        producer.flush()
        print(f"  -> {result['predicted_class']} (conf {result['confidence']})", flush=True)


if __name__ == "__main__":
    main()
