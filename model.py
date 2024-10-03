import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print("Done with imports")

# Example data
data = {
    'IR_intensity': [
        [0.3, 0.5, 0.7, 0.2, 0.9, 0.6, 0.4], 
        [0.2, 0.1, 0.8, 0.7, 0.3, 0.5, 0.9], 
        [0.6, 0.4, 0.3, 0.7, 0.5, 0.2, 0.8]], 
    'object_present': [1, 1, 0],              
    'color': ['red', 'blue', 'none'],         
    'shininess': [0.8, 0.2, 0.0],             
    'distance': [1.5, 2.0, None]              
}

# Format the other data points as One Hot Encoded Values (0 or 1)
df = pd.DataFrame(data)

encoder = OneHotEncoder(drop='first', sparse_output=False)
color_encoded = encoder.fit_transform(df[['color']])
color_columns = encoder.get_feature_names_out(['color'])

scaler = MinMaxScaler()
df['shininess'] = scaler.fit_transform(df[['shininess']])

df_train = df.copy()
df_train['distance'] = df_train['distance'].fillna(0) 
df_train = pd.concat([
    pd.DataFrame(df_train['IR_intensity'].tolist()), 
    df_train[['object_present', 'shininess', 'distance']], 
    pd.DataFrame(color_encoded, columns=color_columns)
], axis=1)

print(df_train)

# Class for the dataset that will be plugged into the model for training
class IRDistanceDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.inputs = self.data.drop(columns=['distance']).values 
        self.targets = self.data['distance'].values 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32) 
        y = torch.tensor(self.targets[idx], dtype=torch.float32) 
        return x, y

# Create dataset
dataset = IRDistanceDataset(df_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Class for model, create a feed forward neural network
class IRDistanceModel(nn.Module):
    def __init__(self):
        super(IRDistanceModel, self).__init__()
    
        self.fc1_full = nn.Linear(in_features=11, out_features=64) 
        self.fc1_ir_only = nn.Linear(in_features=7, out_features=64) 
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 13) 
    
    def forward(self, x):
    
        if x.shape[1] == 7:
            x = torch.relu(self.fc1_ir_only(x)) 
        else:
            x = torch.relu(self.fc1_full(x)) 
        x = torch.relu(self.fc2(x))
        x = self.fc3(x) 
        return F.softmax(x, dim=1) 

model = IRDistanceModel()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1)) 
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

print("Training completed.")

# Test the model on example input
new_ir_values = torch.tensor([[0.3, 0.5, 0.7, 0.2, 0.9, 0.6, 0.4]], dtype=torch.float32)

model.eval()

with torch.no_grad():
    predicted_probs = model(new_ir_values)  # Probabilities for each bin (0 to 12 inches)
    print(predicted_probs)
    
predicted_bin = torch.argmax(predicted_probs, dim=1)
print(f"Predicted distance bin: {predicted_bin.item()} inches")

# Save the entire model
torch.save(model, 'ir_distance_model_full.pth')
print("model saved")