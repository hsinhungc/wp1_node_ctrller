import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

# Define a simple model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Function to evaluate model on validation set
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.view(-1, 28 * 28)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Accuracy calculation
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

# Training function
def train_model():
    # Set up MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlruns.db")  # Local storage
    mlflow.set_experiment("mnist_classification_1")  # Create/use experiment

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Model, loss, optimizer
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Sample input for signature inference
    sample_input = torch.randn(1, 1, 28, 28)

    # Convert model output to numpy to infer signature
    sample_output = model(sample_input).detach().numpy()

    # Infer the model signature automatically
    signature = infer_signature(sample_input.numpy(), sample_output)

    # Start MLflow run
    with mlflow.start_run() as run:
        mlflow.log_param("epochs", 5)
        mlflow.log_param("learning_rate", 0.001)

        for epoch in range(5):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images = images.view(-1, 28 * 28)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Calculate train loss
            train_loss = running_loss / len(train_loader)

            # Evaluate on validation set
            val_loss, val_acc = evaluate(model, val_loader, criterion)

            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Log the final trained model
        mlflow.pytorch.log_model(
            model, 
            "mnist_model", 
            signature=signature, 
            input_example=sample_input.numpy()
        )

        # Register the model in MLflow Model Registry with a tag
        mlflow.set_tag("model_type", "pytorch_mnist")
        mlflow.set_tag("dataset", "MNIST")
        mlflow.set_tag("task", "classification")
        mlflow.set_tag("best_val_acc", val_acc)

        print(f"Model logged successfully in MLflow! Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train_model()
