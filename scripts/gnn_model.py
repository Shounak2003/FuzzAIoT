import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import os
import numpy as np
from scipy.spatial import cKDTree

# Step 1: Load the Dataset
data_path = "C:/Users/Shaurya/Downloads/FuzzAIoT/data/IoTAttackSimulation.csv"  # Update path if needed
df = pd.read_csv(data_path)

# Step 2: Prepare the Data
# Convert node features to PyTorch tensors
node_features = torch.tensor(df[['PacketSize']].values.astype(float), dtype=torch.float)
labels = torch.tensor([1 if label == 'DDoS' else 0 for label in df['Label']], dtype=torch.long)

# Step 3: Create Sparse Graph Connectivity using k-nearest neighbors (k-NN)
num_nodes = len(node_features)
k = 5  # Number of neighbors to connect each node to
tree = cKDTree(node_features)  # Use k-d tree for efficient neighbor search
_, indices = tree.query(node_features, k=k)

edge_index = []
for i in range(num_nodes):
    for j in indices[i]:
        if i != j:
            edge_index.append([i, j])

edge_index = torch.tensor(edge_index).t().contiguous()

# Step 4: Define the Graph Neural Network Model
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels=node_features.size(1), out_channels=16)
        self.conv2 = GCNConv(in_channels=16, out_channels=2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Step 5: Initialize the Optimizer and Loss Function
model = GCN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Step 6: Prepare the data for PyTorch Geometric
data = Data(x=node_features, edge_index=edge_index, y=labels)

# Step 7: Train the Model
def train_model():
    model.train()
    save_directory = "C:/Users/Shaurya/Downloads/FuzzAIoT/outputs/"
    os.makedirs(save_directory, exist_ok=True)

    for epoch in range(3):  # Increase the number of epochs as needed
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        with open(os.path.join(save_directory, 'gnn_training_logs.txt'), 'a') as f:
            f.write(f'Epoch {epoch+1}, Loss: {loss.item()}\n')

# Step 8: Evaluate the Model
def evaluate_model():
    model.eval()
    with torch.no_grad():
        out = model(data)
        _, preds = torch.max(out, dim=1)
        accuracy = torch.sum(preds == data.y).item() / len(data.y)

        save_directory = "C:/Users/Shaurya/Downloads/FuzzAIoT/outputs/"
        os.makedirs(save_directory, exist_ok=True)

        with open(os.path.join(save_directory, 'gnn_evaluation_metrics.txt'), 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
        print(f'Accuracy: {accuracy}')

if __name__ == "__main__":
    print("Training the model...")
    train_model()
    print("Evaluating the model...")
    evaluate_model()

    # Save the model
    model_save_directory = "C:/Users/Shaurya/Downloads/FuzzAIoT/models/"
    os.makedirs(model_save_directory, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(model_save_directory, 'gnn_model.pth'))
    print(f"Model saved to {os.path.join(model_save_directory, 'gnn_model.pth')}")
