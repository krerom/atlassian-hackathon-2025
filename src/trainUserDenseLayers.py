import math
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------
# CONFIG
# ---------------------------

import argparse

parser = argparse.ArgumentParser(description="Train user LSTM adapter from CSV")
parser.add_argument("--csv", required=True, help="Path to CSV file containing sprint data")
parser.add_argument("--output", required=True, help="Path to save user dense adapter")
args = parser.parse_args()

csv_path = args.csv
user_dense_path = args.output

device = 'cpu'
sequence_length = 6
batch_size = 2048
epochs = 1

base_model_path = "/root/sprintsense/lstm/lstm_model_10k_realistic.pth"

features = [
    'sprint_duration_days',
    'number_of_issues',
    'completed_issues_prev_sprint',
    'velocity_prev_sprint',
    'team_size',
    'avg_story_points_per_member'
]
target_column = [
    'velocity',
    'sprint_duration_days',
    'finished_on_time'
]

# ---------------------------
# MODEL DEFINITION
# ---------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.adapter_lstm = nn.LSTM(hidden_size, 32, batch_first=True)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out, _ = self.adapter_lstm(out)
        out = out[:, -1, :]  # Take last time step
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# ---------------------------
# LOAD USER DATA
# ---------------------------
df = pd.read_csv(csv_path)
data = df[features].dropna().values
target = df[target_column].dropna().values

scaler_data = RobustScaler()
scaler_target = RobustScaler()

scaled_data = scaler_data.fit_transform(data)
scaled_target = scaler_target.fit_transform(target)

def create_sequences(data, target, seq_len):
    X, Y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        Y.append(target[i])
    return np.array(X), np.array(Y)

x_seq, y_seq = create_sequences(scaled_data, scaled_target, sequence_length)
x_tensor = torch.tensor(x_seq, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(x_tensor, y_tensor),
                          batch_size=batch_size, shuffle=True)

# ---------------------------
# LOAD BASE MODEL
# ---------------------------
model = LSTMModel(input_size=len(features),
                  hidden_size=256, 
                  output_size=len(target_column)).to(device)

print(f"Loading base LSTM weights from: {base_model_path}")
base_weights = torch.load(base_model_path, map_location=device)
model_dict = model.state_dict()
model_dict.update(base_weights)
model.load_state_dict(model_dict)

# ---------------------------
# FREEZE NON-ADAPTER LAYERS
# ---------------------------
for name, param in model.named_parameters():
    if "adapter" not in name and "fc" not in name:
        param.requires_grad = False

# ---------------------------
# TRAIN DENSE LAYERS ONLY
# ---------------------------
trainable_params = [p for p in model.parameters() if p.requires_grad]
print("Trainable parameters (One LSTM and 4 Dense layers):")
for n, p in model.named_parameters():
    if p.requires_grad:
        print("  ", n)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(trainable_params, lr=0.001)

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    total_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch}/{epochs} - Loss: {total_loss:.6f}")

# ---------------------------
# SAVE USER TRAINED LAYERS ONLY
# ---------------------------
trained_params_only = {
    name: param.detach().cpu()
    for name, param in model.named_parameters()
    if param.requires_grad
}
torch.save(trained_params_only, user_dense_path)
print(f"Saved user-specific adapter layers to: {user_dense_path}")
exit(0)
