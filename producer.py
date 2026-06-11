"""
producer.py — Module 6: the test driver for the streaming pipeline
==================================================================

Publishes wine samples to the 'wine-requests' topic. The consumer service
picks them up, runs the model, and writes results to 'wine-predictions'.

Runs on your LAPTOP (not in Docker), so it reaches the broker through the
HOST listener at localhost:29092 (the door we published with -p 29092:29092).

Run:
    .venv\\Scripts\\python.exe producer.py
"""

import json
import os

from kafka import KafkaProducer

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:29092")
REQUESTS_TOPIC = os.getenv("REQUESTS_TOPIC", "wine-requests")

# Three wine samples, one roughly from each class. The dict VALUES are in the
# model's training feature order (the consumer maps them by position).
SAMPLES = [
    # ~class_0
    {"alcohol": 14.23, "malic_acid": 1.71, "ash": 2.43, "alcalinity_of_ash": 15.6,
     "magnesium": 127.0, "total_phenols": 2.80, "flavanoids": 3.06,
     "nonflavanoid_phenols": 0.28, "proanthocyanins": 2.29, "color_intensity": 5.64,
     "hue": 1.04, "od280_od315": 3.92, "proline": 1065.0},
    # ~class_1
    {"alcohol": 12.37, "malic_acid": 0.94, "ash": 1.36, "alcalinity_of_ash": 10.6,
     "magnesium": 88.0, "total_phenols": 1.98, "flavanoids": 0.57,
     "nonflavanoid_phenols": 0.28, "proanthocyanins": 0.42, "color_intensity": 1.95,
     "hue": 1.05, "od280_od315": 1.82, "proline": 520.0},
    # ~class_2
    {"alcohol": 13.71, "malic_acid": 5.65, "ash": 2.45, "alcalinity_of_ash": 20.5,
     "magnesium": 95.0, "total_phenols": 1.68, "flavanoids": 0.61,
     "nonflavanoid_phenols": 0.52, "proanthocyanins": 1.06, "color_intensity": 7.70,
     "hue": 0.64, "od280_od315": 1.74, "proline": 740.0},
]


def main():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    for i, wine in enumerate(SAMPLES, 1):
        producer.send(REQUESTS_TOPIC, wine)
        print(f"sent sample {i} (alcohol={wine['alcohol']}) -> {REQUESTS_TOPIC}")
    producer.flush()  # make sure everything is actually delivered before exiting
    print(f"Done. Sent {len(SAMPLES)} wine samples.")


if __name__ == "__main__":
    main()
