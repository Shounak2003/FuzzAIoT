import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import pandas as pd
import os

print("Starting script...")  # Debug statement

# Step 1: Load the Dataset
data_path = "C:/Users/Shaurya/Downloads/FuzzAIoT/data/IoTSecurityLowRateTCPDoS.csv"
df = pd.read_csv(data_path)

# Step 2: Prepare the Data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and create input tensors
df['PacketSize'] = df['PacketSize'].astype(str)
inputs = tokenizer(list(df['PacketSize']), return_tensors='pt', padding=True, truncation=True)

# Assuming binary classification (e.g., LowRateTCPDoS vs. Normal)
labels = torch.tensor([1 if label == 'LowRateTCPDoS' else 0 for label in df['Label']])

# Step 3: Define the Transformer-Based Model
class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, 2)  # Binary classification (change 2 to num_classes if multiclass)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

model = TransformerModel()
print("Model created...")  # Debug statement

# Step 4: Initialize the Optimizer and Loss Function
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Step 5: Train the Model
def train_model():
    model.train()
    save_directory = "C:/Users/Shaurya/Downloads/FuzzAIoT/outputs/"
    os.makedirs(save_directory, exist_ok=True)
    
    for epoch in range(3):  # You can increase the number of epochs
        optimizer.zero_grad()
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')  # Debug statement
        with open(os.path.join(save_directory, 'training_logs.txt'), 'a') as f:
            f.write(f'Epoch {epoch+1}, Loss: {loss.item()}\n')

# Step 6: Evaluate the Model
def evaluate_model():
    model.eval()
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        _, preds = torch.max(outputs, dim=1)
        accuracy = torch.sum(preds == labels).item() / len(labels)
        
        save_directory = "C:/Users/Shaurya/Downloads/FuzzAIoT/outputs/"
        os.makedirs(save_directory, exist_ok=True)
        
        with open(os.path.join(save_directory, 'evaluation_metrics.txt'), 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
        print(f'Accuracy: {accuracy}')  # Debug statement

if __name__ == "__main__":
    print("Training the model...")  # Debug statement
    train_model()
    print("Evaluating the model...")  # Debug statement
    evaluate_model()
    
    # Save the model
    model_save_directory = "C:/Users/Shaurya/Downloads/FuzzAIoT/models/"
    os.makedirs(model_save_directory, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(model_save_directory, 'transformer_model.pth'))
    print(f"Model saved to {os.path.join(model_save_directory, 'transformer_model.pth')}")  # Debug statement
