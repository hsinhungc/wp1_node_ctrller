# node_ctrller_api
This repo demonstrate a simple prototype of using FastAPI and MLflow to build node(worker)/controller software architecture.

```
┌─────────────┐    /metrics   ┌─────────────┐     
│  Worker v1  │◀─────────────▶│ Controller  │◀───────┐     
│ (port 5001) │               │  (port 9000)│        │     
└─────────────┘               └─────────────┘        │     
                                  ▲  │  ▲            │     
                                  │  │  │            │     
                                  │  │  │            │     
                                  │  │  │            │     
┌─────────────┐                   │  │  │            │    
│  Worker v2  │◀──────────────────┘  │  └───┐        │    
│ (port 5002) │    /metrics          │      │        │    
└─────────────┘                      ▼      ▼        ▼    
                                    Client script  (your app)
```

## Node and Controller Architecture
Here we use MNIST Model Serving as the prototype.

1. Workers (nodes): each hosts one MLflow model version

Expose /invocations (for inference) and /metrics (name, version, val_accuracy). Report per‐call latency in the JSON response.

2. Controller:

Polls each worker’s /metrics to pick the worker with highest val_accuracy. For each client /predict request, forwards the payload to the chosen worker’s /invocations, records its latency as a Prometheus Gauge, and returns prediction + metadata.

3. Client script:

Generates a dummy MNIST sample (1×1×28×28), posts to /predict (optionally forcing a specific worker via ?worker=<n>), prints and saves the response in worker_benchmark_records.json.

### Setting
Here are the example setting.

Setup worker 1
```
MODEL_VERSION=1 PORT=5001 python worker.py
```
Setup worker 2
```
MODEL_NAME=MNIST_PyTorch_2 MODEL_VERSION=1 PORT=5002 python worker.py
```
Setup controller
```
uvicorn controller:app --host 0.0.0.0 --port 9000
```
Setup prometheus
```
prometheus --config.file=./prometheus.yml
```
### Run software
Here provides an example code to run the interaction of controller and nodes.
```
python api_manolo.py
```
