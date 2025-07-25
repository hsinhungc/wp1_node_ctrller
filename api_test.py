import requests
import numpy as np

# Generate a dummy 28x28 grayscale image (reshape to correct format)
sample_image = np.random.rand(1, 1, 28, 28).tolist()  # Shape (1, 1, 28, 28)

# API Endpoint
url = "http://127.0.0.1:5002/invocations"

# Create JSON request
input_data = {"instances": sample_image}

# Send request
response = requests.post(url, json=input_data)

# Print response
print(response.json())
