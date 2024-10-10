"""
Script to test only IR intensity values for now
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv

print("Done with imports")

def get_data():
    data = {
        'IR_intensity': [],  # 7 IR readings for each sample
        'distance': []  # Distance values
    }
    
    with open("browncardboard.csv", "r") as file:
        reader = csv.DictReader(file)  # Use DictReader to read the file as a dictionary for each row
        for row in reader:
            # Collect IR intensity values (left3 to right3)
            ir_values = [
                int(row['left3']),
                int(row['left2']),
                int(row['left1']),
                int(row['M']),
                int(row['right1']),
                int(row['right2']),
                int(row['right3'])
            ]
            
            # Append the IR values to 'IR_intensity'
            data['IR_intensity'].append(ir_values)

            # Append the corresponding distance
            data['distance'].append(int(row['distance']))
    return data
        
    
data = get_data()

# Format data into a DataFrame
df = pd.DataFrame(data)

# Fill missing distances with 0 (for training purposes)
df_train = df.copy()
df_train['distance'] = df_train['distance'].fillna(0)

# Create a DataFrame with only the IR readings for input
df_train_ir = pd.DataFrame(df_train['IR_intensity'].tolist())

# Dataset class to handle loading IR data and corresponding distance bins
class IRDistanceDataset(Dataset):
    def __init__(self, dataframe, targets):
        self.inputs = dataframe.values  # Only IR readings
        self.targets = targets.values   # Corresponding distances
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)  # IR readings
        y = torch.tensor(self.targets[idx], dtype=torch.float32)  # Distance
        return x, y

# Create dataset
dataset = IRDistanceDataset(df_train_ir, df_train['distance'])
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model class for a simple feedforward network using only IR readings
class IRDistanceModel(nn.Module):
    def __init__(self):
        super(IRDistanceModel, self).__init__()
        self.fc1_ir_only = nn.Linear(in_features=7, out_features=64)  # Only IR inputs
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 13)  # Output probabilities for 13 distance bins (0 to 12 inches)
    
    def forward(self, x):
        x = torch.relu(self.fc1_ir_only(x))  # Use IR-only input
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer: 13 distance bins
        return F.softmax(x, dim=1)  # Softmax for probabilities

# Instantiate the model
model = IRDistanceModel()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 2
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, distances in dataloader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        bin_indices = distances.round().long()  # Convert distances to nearest bin
        
        # Compute loss
        loss = criterion(outputs, bin_indices)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")
print("Training completed.")

# Test the model on new IR intensity data
new_ir_values = torch.tensor([[6,85,171,410,275,16,0]], dtype=torch.float32)

# Set model to evaluation mode
model.eval()

# Predict the probabilities for each distance bin
with torch.no_grad():
    predicted_probs = model(new_ir_values)
    print(predicted_probs)
    
# Find the most probable distance bin
predicted_bin = torch.argmax(predicted_probs, dim=1)
print(f"Predicted distance bin: {predicted_bin.item()} inches")
# Save the model
torch.save(model, 'ir_distance_model_full.pth')
print("Model saved")