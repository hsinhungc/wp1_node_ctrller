"""
Minimal MLflow worker.

Expected env-vars
-----------------
MODEL_NAME           e.g. "mnist_model"
MODEL_VERSION        e.g. "1"
PORT                 defaults to 5001
MLFLOW_TRACKING_URI  defaults to "sqlite:///mlruns.db"
"""

import os, time, numpy as np, uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pytorch
import torch

import mlflow
from mlflow.tracking import MlflowClient

# --------------------------------------------------------------------------- #
# 1) configuration & MLflow housekeeping
# --------------------------------------------------------------------------- #
MODEL_NAME    = os.getenv("MODEL_NAME", "MNIST_PyTorch_Model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
PORT          = int(os.getenv("PORT", "5001"))
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")

mlflow.set_tracking_uri(MLFLOW_URI)
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model     = mlflow.pytorch.load_model(model_uri)

client    = MlflowClient(tracking_uri=MLFLOW_URI)
run_id    = client.get_model_version(MODEL_NAME, MODEL_VERSION).run_id
VAL_ACC   = client.get_run(run_id).data.metrics.get("val_accuracy")

# --------------------------------------------------------------------------- #
# 2) FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(title=f"{MODEL_NAME}_v{MODEL_VERSION}")

class InvocationRequest(BaseModel):
    # Mirrors mlflow's default "pandas-split" JSON format:
    # {"inputs": [[[...]]]}
    inputs: list

@app.post("/invocations")
def invocations(req: InvocationRequest):
    #arr   = np.asarray(req.inputs, dtype=np.float32)
    tensor_in = torch.tensor(req.inputs, dtype=torch.float32)
    #preds = model.predict(arr)            # returns logits

    with torch.no_grad():
        start_time = time.perf_counter()
        preds = model(tensor_in)
        end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1e3
    #preds = model(tensor_in)

    return {
        "predictions": preds.detach().cpu().numpy().tolist(),    
        "latency_ms":  latency_ms
        }

@app.get("/metrics")
def metrics():
    return {"name": MODEL_NAME,
            "version": int(MODEL_VERSION),
            "val_accuracy": VAL_ACC}
            #"latency_ms":  latency_ms}

@app.get("/ping")
def ping():
    # Same readiness probe mlflow serve uses
    return "ready"

# --------------------------------------------------------------------------- #
# 3) entry-point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    uvicorn.run("worker:app",
                host="0.0.0.0",
                port=PORT,
                log_level="info")
