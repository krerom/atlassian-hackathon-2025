import math
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

# ---------------------------
# Check GPU & Optimizations
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.set_num_threads(os.cpu_count())
print("PyTorch CPU threads:", torch.get_num_threads())
# Disabling cuDNN for stability on older GPUs (remains from previous fix)
torch.backends.cudnn.enabled = False 

# ---------------------------
# Parameters
# ---------------------------
csv_path = "/home/romankreiner/Documents/Hackathon/SprintSense/synthetic_sprints.csv"


sequence_length = 6
batch_size = 4096 
epochs = 100

target_column = [
    'velocity',             # 1. Regression
    'sprint_duration_days', # 2. Regression
    'finished_on_time'      # 3. Classification (Binary)
]
features = [
    'sprint_duration_days',
    'number_of_issues',
    'completed_issues_prev_sprint',
    'velocity_prev_sprint',
    'team_size',
    'avg_story_points_per_member'
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
training_data_len = math.ceil(len(dataset) * 0.8)

scaler_data = RobustScaler()
scaler_target = RobustScaler()
scaled_data = scaler_data.fit_transform(dataset)

reg_targets = target[['velocity', 'sprint_duration_days']].values

cls_targets = target['finished_on_time'].values

scaler_target_reg = RobustScaler()
scaled_reg_targets = scaler_target_reg.fit_transform(reg_targets)

scaled_target_values = np.hstack((
    scaled_reg_targets, 
    cls_targets.astype(np.float32).reshape(-1, 1) 
))

def create_sequences(data, target, seq_len):
    x, y = [], []
    for i in range(seq_len, len(data)):
        x.append(data[i-seq_len:i])
        y.append(target[i])
    return np.array(x), np.array(y)

x_train, y_train = create_sequences(scaled_data[:training_data_len], scaled_target_values[:training_data_len], sequence_length)
x_test, y_test = create_sequences(scaled_data[training_data_len:], scaled_target_values[training_data_len:], sequence_length)

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

num_cpu_cores = os.cpu_count()
num_workers = max(1, num_cpu_cores - 2) # Use most cores for loading

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


# ---------------------------
# MODEL DEFINITION
# ---------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size): # output_size removed
        super(LSTMModel, self).__init__()
        
        self.lstm_stack = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True)
        
        self.adapter_lstm = nn.LSTM(hidden_size, 32, batch_first=True)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 64) # Latent feature size

        # Two regression outputs (Velocity, Sprint Duration)
        self.reg_head = nn.Linear(64, 2)
        # One binary classification output (Finished on Time)
        self.cls_head = nn.Linear(64, 1) 
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() # For classification head
    
    def forward(self, x):
        out, _ = self.lstm_stack(x)
        out, _ = self.adapter_lstm(out)
        out = out[:, -1, :]
        
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out)) # Latent features (64)
        
        # Split the output stream
        reg_output = self.reg_head(out)
        # Apply Sigmoid for probability prediction (Classification)
        cls_output = self.sigmoid(self.cls_head(out)) 
        
        # Concatenate for single return tensor: [2 Regression, 1 Classification]
        return torch.cat([reg_output, cls_output], dim=1)

# Initialize model
model = LSTMModel(input_size=len(features), hidden_size=256).to(device)


# ---------------------------
# MULTI-TASK LOSS AND OPTIMIZER
# ---------------------------
# Define individual loss functions
mse_loss_fn = nn.MSELoss()
bce_loss_fn = nn.BCEWithLogitsLoss() 

ALPHA = 0.5 

optimizer = torch.optim.Adam(model.parameters())

# ---------------------------
# TRAINING LOOP
# ---------------------------
best_loss = float('inf')
reg_loss = 0
cls_loss = 0

for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
    
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        
        optimizer.zero_grad()
        output = model(xb)
        
        reg_targets = yb[:, :2]  # First two columns (Velocity, Duration)
        cls_targets = yb[:, 2].unsqueeze(1) # Last column (Finished_on_Time), reshape for BCELoss
        
        # Predictions:
        reg_preds = output[:, :2]
        cls_preds = output[:, 2].unsqueeze(1)
        
        # Calculate individual losses
        reg_loss = mse_loss_fn(reg_preds, reg_targets)
        cls_loss = bce_loss_fn(cls_preds, cls_targets)

        total_loss = (1 - ALPHA) * reg_loss + ALPHA * cls_loss
        
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item() * xb.size(0)
    
    epoch_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch}/{epochs}, Total Weighted Loss: {epoch_loss:.6f}\nRegression Loss: {reg_loss} | Classification Loss: {cls_loss}")
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"lstm_sprint_epoch{epoch:03d}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    
    # Early stopping
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_path = os.path.join(finished_dir, "lstm_model_v2.pth")
        torch.save(model.state_dict(), best_model_path)

print(f"Training complete. Best model saved at {best_model_path}")

# ---------------------------
# Save Scalers
# ---------------------------
joblib.dump(scaler_data, os.path.join(finished_dir, "scaler_data.pkl"))
joblib.dump(scaler_target, os.path.join(finished_dir, "scaler_target.pkl"))
