import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn

# ---------------------------
# Parameters
# ---------------------------
sequence_length = 6
csv_path = "/home/romankreiner/Documents/Hackathon/SprintSense/synthetic_sprints.csv"
finished_model_path = "/home/romankreiner/Documents/Hackathon/SprintSense/finished_models/lstm_sprint_final.pt"

features = [
    'sprint_duration_days',
    'total_story_points',
    'number_of_issues',
    'completed_issues_prev_sprint',
    'velocity_prev_sprint',
    'team_size',
    'avg_story_points_per_member'
]
target_column = [
    'velocity',
    'sprint_duration_days',
    'finished_on_time',
    'predicted_duration_days'
]

device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv(csv_path)
data = df[features].dropna()
target = df[target_column].dropna()

# ---------------------------
# Scaling
# ---------------------------
scaler_data = RobustScaler()
scaler_target = RobustScaler()

scaled_data = scaler_data.fit_transform(data.values)
scaled_target = scaler_target.fit_transform(target.values)

# ---------------------------
# Prepare first sequence
# ---------------------------
x_input = scaled_data[:sequence_length]  # first 6 rows
x_input_tensor = torch.tensor(x_input[np.newaxis, :, :], dtype=torch.float32).to(device)  # shape: (1, 6, features)

y_actual = scaled_target[sequence_length]  # 7th row

# ---------------------------
# Define model (must match training)
# ---------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out = out[:, -1, :]  # Take last time step
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

model = LSTMModel(input_size=len(features), hidden_size=200, output_size=len(target_column)).to(device)
model.load_state_dict(torch.load(finished_model_path, map_location=device))
model.eval()
print("Model loaded for inference.")

# ---------------------------
# Inference
# ---------------------------
with torch.no_grad():
    y_pred_scaled = model(x_input_tensor).cpu().numpy()[0]

# Inverse transform to original scale
y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(1, -1))[0]
y_actual_orig = scaler_target.inverse_transform(y_actual.reshape(1, -1))[0]
y_pred = y_pred = np.round(y_pred)

# ---------------------------
# Compare prediction and actual
# ---------------------------
print(target_column)
print("Predicted 7th row:", y_pred)
print("Actual 7th row:   ", y_actual_orig)

# Variance (mean squared difference)
variance = np.mean((y_pred - y_actual_orig) ** 2)
print("Prediction variance (MSE):", variance)
