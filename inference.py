import mlflow.pytorch
import torch

# Use the same tracking URI so MLflow knows where to find run metadata
mlflow.set_tracking_uri("sqlite:///mlruns.db")

# Get the latest run's model
ID = 'a4387d1ce94d4851b33e2ee045ea88a1'
latest_model_path = f"runs:/{ID}/mnist_model"  # Replace <YOUR_RUN_ID> with actual ID

# Load the model
model = mlflow.pytorch.load_model(model_uri=latest_model_path)

# Run inference
input_tensor = torch.randn(1, 1, 28, 28)  # Dummy input
output = model(input_tensor)

print("Model loaded successfully! Output:", output)


# registered_model = mlflow.register_model(
#     model_uri=f"runs:/{ID}/mnist_model",
#     name="MNIST_PyTorch_Model"
# )
# print(f"Model registered with name: {registered_model.name}")

model = mlflow.pytorch.load_model("models:/MNIST_PyTorch_Model/2")  # MNIST_PyTorch_Model/2 = Version 2
# Run inference
input_tensor = torch.randn(1, 1, 28, 28)  # Dummy input
output = model(input_tensor)
print("Model loaded successfully! Output:", output)

