import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Define the GNN model
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels=1, out_channels=16)
        self.conv2 = GCNConv(in_channels=16, out_channels=2)  # 2 classes: Benign, DDoS

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Load and prepare the dataset
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    node_features = torch.tensor(df[['PacketSize']].values.astype(float), dtype=torch.float)
    labels = torch.tensor([1 if label == 'DDoS' else 0 for label in df['Label']], dtype=torch.long)
    num_nodes = len(node_features)
    edge_index = torch.tensor([[i, i] for i in range(num_nodes)]).t().contiguous()  # Self-loops
    return Data(x=node_features, edge_index=edge_index, y=labels)

# Train the GNN model
def train_gnn_model(data, model, epochs=150):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    # Lists to store loss and accuracy values for visualization
    losses = []
    accuracies = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, pred = torch.max(out, dim=1)
        correct = (pred == data.y).sum().item()
        accuracy = correct / len(data.y)

        # Store loss and accuracy
        losses.append(loss.item())
        accuracies.append(accuracy)

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Accuracy: {accuracy * 100:.2f}%')

    return model, losses, accuracies

# Validate the GNN model
def validate_gnn_model(model, validation_data):
    model.eval()
    out = model(validation_data)
    _, pred = torch.max(out, dim=1)
    correct = (pred == validation_data.y).sum().item()
    accuracy = correct / len(validation_data.y)
    print(f'Validation Accuracy: {accuracy}')
    
    # Generate classification report
    print("Classification Report:")
    print(classification_report(validation_data.y.cpu(), pred.cpu(), target_names=["Benign", "DDoS"]))
    
    # Generate confusion matrix
    cm = confusion_matrix(validation_data.y.cpu(), pred.cpu())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "DDoS"], yticklabels=["Benign", "DDoS"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.show()

# Visualization function
def plot_training_metrics(losses, accuracies):
    epochs = range(1, len(losses) + 1)
    
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, 'b', label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, 'g', label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# Main function
def main():
    # Paths to the training and validation datasets
    training_csv_path = "C:/Users/Shaurya/Downloads/FuzzAIoT/data/CombinedTraffic.csv"
    validation_csv_path = "C:/Users/Shaurya/Downloads/FuzzAIoT/data/DDoSim_AttackSim.csv"

    # Load training and validation data
    training_data = load_data(training_csv_path)
    validation_data = load_data(validation_csv_path)

    # Initialize and train the model
    model = GCN()
    trained_model, losses, accuracies = train_gnn_model(training_data, model, epochs=150)

    # Save the trained model
    model_save_path = "C:/Users/Shaurya/Downloads/FuzzAIoT/models/gnn_model.pth"
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot training loss and accuracy
    plot_training_metrics(losses, accuracies)

    # Validate the model
    validate_gnn_model(trained_model, validation_data)

if __name__ == "__main__":
    main()
