# controller.py
import os
import numpy as np
import requests
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
from prometheus_client import start_http_server, Gauge

# Map model version → worker base URL
WORKERS = {
    1: "http://0.0.0.0:5001",
    2: "http://0.0.0.0:5002",
}

# Controller exports metrics on this port
CONTROLLER_METRICS_PORT = int(os.getenv("CONTROLLER_METRICS_PORT", "9100"))
# ───────────────────────────────────────────────────────────────────────────────

# 1) Define a Gauge to record “latency returned by worker”
WORKER_LATENCY = Gauge(
    "controller_worker_reported_latency_ms",
    "Latency (ms) as reported by each worker",
    labelnames=["worker_id"]
)

app = FastAPI(title="MNIST-Controller")

class MNISTRequest(BaseModel):
    data: list  # flattened 784‐length list or nested list of shape [1,1,28,28]

@app.on_event("startup")
def startup_metric_server():
    # Expose /metrics on CONTROLLER_METRICS_PORT
    start_http_server(CONTROLLER_METRICS_PORT)

def _fetch_worker_metrics(url: str) -> dict:
    """
    Hits GET {url}/metrics and returns its JSON payload.
    Expected payload: {"version": int, "val_accuracy": float}
    """
    resp = requests.get(f"{url}/metrics", timeout=1)
    resp.raise_for_status()
    return resp.json()

def _choose_best_version() -> int:
    best_ver, best_acc = None, -1.0
    for ver, base in WORKERS.items():
        try:
            m = _fetch_worker_metrics(base)
            if m["val_accuracy"] is not None and m["val_accuracy"] > best_acc:
                best_acc, best_ver = m["val_accuracy"], ver
        except Exception:
            continue
    if best_ver is None:
        raise RuntimeError("No worker metrics available")
    return best_ver

@app.post("/predict")
def predict(req: MNISTRequest,
            worker: Optional[int] = Query(None, description="force a specific model version")):
    # 1) Decide which version to call
    if worker is None:
        worker = _choose_best_version()
    if worker not in WORKERS:
        return {"error": f"Unknown version {version}"}

    base_url = WORKERS[worker]
    
    # 2) Forward the payload to the worker’s /invocations
    invoc = {"inputs": np.array(req.data, dtype=np.float32).reshape(1,1,28,28).tolist()}
    r = requests.post(f"{base_url}/invocations", json=invoc, timeout=2)
    r.raise_for_status()
    resp_json = r.json()
    logits = np.array(resp_json.get("predictions"))

    # 3) Fetch val_accuracy from that worker again for the response
    metrics = _fetch_worker_metrics(base_url)

    # 4) Extract the worker’s reported latency
    worker_latency_ms = resp_json.get("latency_ms")
    #print('worker_latency_ms: ', worker_latency_ms)
    if worker_latency_ms is not None:
        WORKER_LATENCY.labels(worker_id=str(worker)).set(worker_latency_ms)

    return {
        "WORKER":      worker,
        "Sample Prediction":   int(logits.argmax()),
        "Model Name:": metrics.get("name", None),
        "Model Version:": metrics.get("version", None),
        "Model val_accuracy": float(metrics.get("val_accuracy", None)),
        "latency_ms":   worker_latency_ms
    }

# --- optional: standalone demo ----------------------------------------------
if __name__ == "__main__":
    import uvicorn
    os.environ.setdefault("CONTROLLER_METRICS_PORT", "9100")
    uvicorn.run("controller:app", host="0.0.0.0", port=9000, log_level="info")
