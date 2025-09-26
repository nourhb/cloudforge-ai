from locust import HttpUser, task, between
import json
import os
import requests

AI_URL = os.getenv("AI_URL", "http://localhost:5001")

class CloudforgeUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task(3)
    def backend_list_marketplace(self):
        # Uses locust host (set via --host or LOCUST_HOST env)
        self.client.get("/api/marketplace/list", name="GET /api/marketplace/list")

    @task(2)
    def backend_health(self):
        self.client.get("/health", name="GET /health")

    @task(2)
    def iac_generate_via_backend(self):
        payload = {"prompt": "Expose backend as ClusterIP on port 4000"}
        headers = {"Content-Type": "application/json"}
        self.client.post("/api/iac/generate", data=json.dumps(payload), headers=headers, name="POST /api/iac/generate")

    @task(1)
    def ai_metrics_direct(self):
        # Direct AI metrics scrape to ensure AI responsiveness under load
        try:
            r = requests.get(f"{AI_URL}/metrics", timeout=1.5)
            _ = r.status_code
        except Exception:
            pass

# Usage:
# locust -f tests/perf/locustfile.py --host http://localhost:4000 --users 100 --spawn-rate 10
