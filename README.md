
# 📘 How the LSTM Model Works

SprintSense uses a pretrained LSTM model to predict the key characteristics of the next Jira sprint based on the previous six sprints.
To build the base model, we generated a large synthetic dataset of ~100,000 sprint sequences that mimic realistic patterns commonly seen in Jira teams.

The dataset is then split into training and testing sets. For every group of seven consecutive sprints, the first six are used as inputs and the seventh sprint becomes the prediction target.
To avoid temporal leakage, each seventh sprint is placed in the test set, while the six preceding sprints remain in the training set.

During training, the LSTM model learns relationships between velocity, sprint duration, completion rates, and cycle patterns. With enough examples, it becomes capable of predicting the next sprint with useful accuracy.

---

# 🧩 Per-Team Personalization Through Lightweight Adapters

Every Jira team has its own style of working, pacing, and consistency. Because one global model cannot fully capture these differences, SprintSense uses a **transfer-learning adapter architecture**:

1. Load the global pretrained model (LSTM layers + Dense layers).
2. **Freeze the LSTM layers**, preserving the temporal reasoning learned from ~100k synthetic sprint sequences.
3. Fine-tune **only the Dense layers** using all historical sprints of a specific team.
4. Save **only the Dense layer weights** for that team.

This creates a lightweight “adapter” for each team:

* The **base model** (LSTM layers) is shared by all teams.
* Each team stores only a tiny set of **Dense-layer weights**, typically just a few kilobytes.
* At prediction time, SprintSense loads:

  * LSTM weights from the global base model
  * Dense weights from the team’s adapter

This keeps storage minimal and scales efficiently, even for large organizations with many teams.

Testing on synthetic data shows overall accuracies around ~89%, depending on the target feature.

---

# 🔍 Why Use an LSTM?

LSTMs are ideal for sequence forecasting problems. Here, the model processes the most recent **six** sprints and produces predictions for the **next** sprint:

* Velocity
* Sprint duration
* Finished-on-time probability
* Predicted duration (days)

These predictions can then be used to generate insights such as:

* Detecting performance trends
* Identifying expected bottlenecks
* Highlighting risk before a sprint begins
* Explaining predictions using an LLM (e.g., the ChatGPT API)
* Displaying upcoming expectations inside the SprintSense dashboard

By comparing predicted future behavior with past performance, SprintSense helps teams understand where adjustments may be beneficial — or confirms where their workflow is improving.

# Database schema: users

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    provider VARCHAR(255) NOT NULL,
    api_key VARCHAR(255) NOT NULL,
    account_id VARCHAR(255) NOT NULL,
    adapter_path VARCHAR(512),
    auth_token VARCHAR(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Prediction fields
    predicted_velocity FLOAT,
    predicted_sprint_duration_days FLOAT,
    predicted_finished_on_time TINYINT(1),
    predicted_duration_days FLOAT,

    -- Optional: additional prediction metadata
    prediction_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence_interval FLOAT,
    risk_flag TINYINT(1)
);

# Model Architecture

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