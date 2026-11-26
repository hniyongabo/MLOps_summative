"""Flood-request simulation for the Bean Leaf Doctor API.

Each simulated user keeps sending leaf photos from demo_samples/ to /api/predict
(the expensive call) and occasionally checks /api/status (cheap call).

Web UI:    locust -f locustfile.py --host http://localhost:8080
Headless:  locust -f locustfile.py --host http://localhost:8080 --headless \
                  -u 50 -r 5 -t 2m --csv loadtest/results/3_containers
"""
import glob
import os
import random

from locust import HttpUser, between, task

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = []
for path in sorted(glob.glob(os.path.join(ROOT, "demo_samples", "*.jpg"))):
    with open(path, "rb") as f:
        SAMPLES.append((os.path.basename(path), f.read()))
if not SAMPLES:
    raise SystemExit("No .jpg files found in demo_samples/")


class LeafUser(HttpUser):
    wait_time = between(1, 3)  # seconds between requests per user

    @task(5)
    def predict(self):
        name, data = random.choice(SAMPLES)
        with self.client.post("/api/predict", files={"file": (name, data, "image/jpeg")},
                              name="/api/predict", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
            elif "predicted_class" not in r.text:
                r.failure("no prediction in response")

    @task(1)
    def status(self):
        self.client.get("/api/status", name="/api/status")
