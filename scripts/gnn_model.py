import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.nn import GCNConv
import pandas as pd
import os

# Step 1: Load the Dataset
data_path = "C:/Users/Shaurya/Downloads/FuzzAIoT/data/IoTAttackSimulation.csv"  # Update path if needed
df = pd.read_csv(data_path)

# Step 2: Prepare the Data
# Converting nodes and edges from the CSV
node_features = df[['PacketSize']].values.astype(float)
labels = torch.tensor([1 if label == 'DDoS' else 0 for label in df['Label']])

# Assuming the nodes are arranged in a grid. You need to define the grid's width.
height = 100  # Adjust height based on your dataset
width = len(node_features) // height

# Creating edge indices for the grid
edge_index = torch_geometric.utils.grid(height, width).to(torch.long).t().contiguous()

# Convert node features to a PyTorch tensor
x = torch.tensor(node_features, dtype=torch.float)

# Step 3: Define the Graph Neural Network Model
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels=x.size(1), out_channels=16)
        self.conv2 = GCNConv(in_channels=16, out_channels=2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Step 4: Initialize the Optimizer and Loss Function
model = GCN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Step 5: Prepare the data for PyTorch Geometric
data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=labels)

# Step 6: Train the Model
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

# Step 7: Evaluate the Model
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
