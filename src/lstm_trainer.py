import math
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ---------------------------
# History
# ---------------------------
"""
256 Hidden Layer Size | Batch Size 2048 |  50 Epochs | 100k Samples => Loss: 0.30   | (loss could be lower)
256 Hidden Layer Size | Batch Size 2048 |  90 Epochs | 100k Samples => Loss: 0.30   | (loss plateaued)
256 Hidden Layer Size | Batch Size 2048 | 100 Epochs | 100k Samples => Loss: 0.09   | (overfitting a bit, ideally slightly above 0.15)
256 Hidden Layer Size | Batch Size 2048 | 100 Epochs | 200k Samples => Loss: 0.16   | (amazing loss - 200k samples can give a good picture)
256 Hidden Layer Size | Batch Size 2048 | 150 Epochs | 300k Samples => Loss: 0.29   | (loss plateaued - maybe model complexity insufficient)
"""

# ---------------------------
# Check GPU
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.set_num_threads(os.cpu_count())
print("PyTorch CPU threads:", torch.get_num_threads())
torch.backends.cudnn.enabled = False
# ---------------------------
# Parameters
# ---------------------------
sequence_length = 6
batch_size = 2048
epochs = 19
output_name = "lstm_model_10k_realistic.pth"
csv_path = "/home/romankreiner/Documents/Hackathon/SprintSense/synthetic_sprints_realistic.csv"

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

checkpoint_dir = "./checkpoints"
finished_dir = "./finished_models"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(finished_dir, exist_ok=True)

# ---------------------------
# Data Preparation
# ---------------------------
df = pd.read_csv(csv_path)

data = df[features].dropna()
target = df[target_column].dropna()
dataset = data.values
target_values = target.values
training_data_len = math.ceil(len(dataset) * 0.8)

scaler_data = RobustScaler()
scaler_target = RobustScaler()
scaled_data = scaler_data.fit_transform(dataset)
scaled_target = scaler_target.fit_transform(target_values)

# Prepare sequences
def create_sequences(data, target, seq_len):
    x, y = [], []
    for i in range(seq_len, len(data)):
        x.append(data[i-seq_len:i])
        y.append(target[i])
    return np.array(x), np.array(y)

x_train, y_train = create_sequences(scaled_data[:training_data_len], scaled_target[:training_data_len], sequence_length)
x_test, y_test = create_sequences(scaled_data[training_data_len:], scaled_target[training_data_len:], sequence_length)

# Convert to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ---------------------------
# Model Definition
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

model = LSTMModel(input_size=len(features), hidden_size=256, output_size=len(target_column)).to(device)

# ---------------------------
# Loss and Optimizer
# ---------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# ---------------------------
# Training Loop
# ---------------------------
best_loss = float('inf')
plotData = []
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    
    epoch_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.6f}")
    plotData.append(epoch_loss)
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"lstm_sprint_epoch{epoch:03d}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    
    # Optional early stopping
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_path = os.path.join(finished_dir, output_name)
        torch.save(model.state_dict(), best_model_path)

print(f"Training complete. Best model saved at {best_model_path}")

# ---------------------------
# Save ONLY LSTM layers
# ---------------------------

lstm_only_weights = {}

for name, param in model.named_parameters():
    if "lstm" in name.lower():  # catches lstm1, lstm2, lstm3
        lstm_only_weights[name] = param.detach().cpu()

save_path = os.path.join(finished_dir, "base_lstm_layers_only.pth")
torch.save(lstm_only_weights, save_path)

print(f"Saved ONLY LSTM layer weights to: {save_path}")

import joblib
joblib.dump(scaler_data, os.path.join(finished_dir, "scaler_data.pkl"))
joblib.dump(scaler_target, os.path.join(finished_dir, "scaler_target.pkl"))


# ---------------------------
# Plotting
# ---------------------------

plt.plot(plotData)
plt.title("Loss Graph")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()