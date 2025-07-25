import requests
import numpy as np
import os
import json

# 1) Generate a dummy MNIST sample (shape: 1×1×28×28)
sample_image = np.random.rand(1, 1, 28, 28).astype("float32").tolist()

# 2) Controller URL
url = "http://0.0.0.0:9000/predict"  # your controller’s API

# 3) Build payload
payload = {"data": sample_image}

# 4) Send request
resp = requests.post(url, params={"worker": 2}, json=payload) # to worker 2
resp.raise_for_status()
result = resp.json()
# 5) Print the controller’s response
#    e.g. {"version": 2, "prediction": 7, "val_accuracy": 0.9843}
print(resp.json())

# 6) Save to a JSON file (append as a new record)
db_file = "worker_benchmark_records.json"

# if file exists, load existing list; otherwise start fresh
if os.path.exists(db_file):
    with open(db_file, "r") as f:
        records = json.load(f)
else:
    records = []

# add a timestamp and the record
import datetime
record = {
    "timestamp": datetime.datetime.now().isoformat(),
    **result
}
records.append(record)

# write back
with open(db_file, "w") as f:
    json.dump(records, f, indent=2)

print(f"Saved prediction to {db_file}")