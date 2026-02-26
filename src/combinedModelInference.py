import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import sys
import json

# ---------------------------
# CONFIG
# ---------------------------

import argparse

parser = argparse.ArgumentParser(description="Predict users next sprint from past six sprints.")
parser.add_argument("--csv", required=True, help="Path to CSV file containing sprint data")
parser.add_argument("--adapter", required=True, help="Path to user Dense layer adapter.")
parser.add_argument("--output", required=True, help="Path to save user prediction CSV.")
args = parser.parse_args()

csv_path = args.csv
user_dense_path = args.adapter
user_output = args.output

# ---------------------------
# Parameters
# ---------------------------
sequence_length = 6


base_lstm_path = "/root/sprintsense/lstm/lstm_model_10k_realistic.pth"

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Model
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
# Load Dataset
# ---------------------------
df = pd.read_csv(csv_path)
data = df[features].fillna(0)
target = df[target_column].fillna(0)

scaler_data = joblib.load("/root/sprintsense/lstm/scaler_data.pkl")
scaler_target = joblib.load("/root/sprintsense/lstm/scaler_target.pkl")

scaled_data = scaler_data.transform(data.values)
scaled_target = scaler_target.transform(target.values)


# Prepare seq
x_input = scaled_data[-sequence_length:]
x_input_tensor = torch.tensor(x_input[np.newaxis], dtype=torch.float32).to(device)

y_actual_scaled = scaled_target[sequence_length]
y_actual_orig = scaler_target.inverse_transform(y_actual_scaled.reshape(1, -1))[0]

# ---------------------------
# Load base model + user dense
# ---------------------------

model = LSTMModel(input_size=len(features), hidden_size=256, output_size=len(target_column)).to(device)

base_state_dict = torch.load(base_lstm_path, map_location=device)

non_adapter_weights = {
    name: param
    for name, param in base_state_dict.items()
    if "adapter" not in name
}
model_dict = model.state_dict()
model_dict.update(non_adapter_weights)
model.load_state_dict(model_dict)

# Now load user Dense layers
user_dense_weights = torch.load(user_dense_path, map_location=device)
model_dict.update(user_dense_weights)
model.load_state_dict(model_dict)

model.eval()

# ---------------------------
# Inference
# ---------------------------
with torch.no_grad():
    y_pred_scaled = model(x_input_tensor).cpu().numpy()[0]

# inverse scale
y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(1, -1))[0]

# round everything
y_pred = np.round(y_pred)

# ---------------------------
# Results
# ---------------------------

y_pred_list = y_pred.tolist()

y_pred_dict = dict(zip(target_column, y_pred_list))

df_predictions = pd.DataFrame([y_pred_dict]) 
df_predictions.to_csv(user_output, index=False) # Use the argparse variable here

sys.stdout.write(json.dumps(y_pred_dict))
sys.stdout.flush()

exit(0)
