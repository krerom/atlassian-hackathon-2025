import pandas as pd
import numpy as np

# --- Parameters ---
total_sprints = 10000
sprints_per_sequence = 7
max_boards = total_sprints // sprints_per_sequence

outputPath = "synthetic_sprints_realistic.csv"

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

columns = ['board_id'] + features + target_column

data = []

# --- Data Generation Loop ---

# Initialize starting metrics for the first sprint (Sprint 0)
# These initial values are arbitrary but used to start the sequence logic
initial_velocity = np.random.randint(15, 45)
initial_completed_issues = np.random.randint(5, 15)

for board_id in range(max_boards):
    # Set stable parameters for the entire board/sequence (e.g., team size)
    # A team size is usually stable over a few sprints.
    team_size = np.random.randint(4, 8)
    # Most sprints are 10 or 14 days in practice
    sprint_duration_days = np.random.choice([10, 14])

    # Starting values for the first sprint in this board's sequence
    current_velocity_prev = initial_velocity
    current_completed_issues_prev = initial_completed_issues

    for sprint_num in range(sprints_per_sequence):
        # 1. Determine the expected capacity based on team stability
        # Base capacity (story points) is (team_size * duration) factor * velocity_prev, with noise.
        # This creates a trend correlation.
        
        # Base potential velocity (a proxy for effort capacity)
        base_potential_velocity = (team_size * 5) + (current_velocity_prev * 0.5)

        # 2. Introduce realistic variation and constraints
        
        # Issues for this sprint (proportional to velocity_prev to maintain flow)
        number_of_issues = int(np.round(current_velocity_prev / np.random.uniform(2.5, 4.5)))
        number_of_issues = np.clip(number_of_issues, 5, 25) # Cap issues

        # Current Velocity (Target): based on potential capacity and previous performance (with noise)
        # Velocity should generally track capacity with some random fluctuation (Gaussian noise).
        # Standard deviation is smaller for more stable teams.
        current_velocity = int(np.round(np.random.normal(base_potential_velocity, 5)))
        current_velocity = np.clip(current_velocity, 5, 60) # Cap velocity

        # 3. Derive correlated metrics
        
        # Avg Story Points Per Member (Velocity / Team Size, plus minor noise)
        # This MUST be derived from Velocity and Team Size for realism.
        ideal_sp_per_member = current_velocity / team_size
        avg_story_points_per_member = np.random.normal(ideal_sp_per_member, 0.5)
        avg_story_points_per_member = np.clip(avg_story_points_per_member, 1.0, 10.0)

        # 4. Finished On Time (Correlation logic)
        # Low velocity relative to team size/duration generally means not finishing on time (0).
        # We'll use a logistic function or a threshold based on velocity and expected range.
        # Expected velocity factor (how much they completed relative to a high baseline):
        expected_range_baseline = team_size * 7 # e.g., 7 SP per person is a good target
        success_ratio = current_velocity / expected_range_baseline
        
        # Probability of finishing on time increases sharply with success_ratio
        prob_finished = 1 / (1 + np.exp(-5 * (success_ratio - 0.7))) # Sigmoid function centered around 70% success_ratio
        finished_on_time = 1 if np.random.rand() < prob_finished else 0

        # Create the sprint record
        sprint = {
            "board_id": board_id,
            "sprint_duration_days": sprint_duration_days,
            "number_of_issues": number_of_issues,
            "completed_issues_prev_sprint": current_completed_issues_prev,
            "velocity_prev_sprint": current_velocity_prev,
            "team_size": team_size,
            "avg_story_points_per_member": avg_story_points_per_member,
            # targets
            "velocity": current_velocity,
            # We enforce 'sprint_duration_days' to be the same in the target column for consistency
            "sprint_duration_days_target": sprint_duration_days,
            "finished_on_time": finished_on_time
        }
        data.append(sprint)

        # 5. Update for the next sprint in the sequence (logic of creating consecutive sprints)
        current_velocity_prev = current_velocity
        # The number of completed issues in this sprint becomes the 'completed_issues_prev_sprint' for the next one.
        # For simplicity, we can estimate issues completed by number_of_issues * (velocity / (avg SP per issue))
        # A simpler realistic proxy: completed_issues = number_of_issues * (1 if finished_on_time else 0.8)
        issues_completed_this_sprint = int(np.round(number_of_issues * (0.8 + 0.2 * finished_on_time)))
        current_completed_issues_prev = issues_completed_this_sprint


# Create DataFrame
df = pd.DataFrame(data)

# Align column order with the original request, renaming the redundant column
df = df.rename(columns={"sprint_duration_days_target": "sprint_duration_days_target"})
final_columns = ['board_id'] + features + target_column
df = df[final_columns]

# Save to CSV
df.to_csv(outputPath, index=False)
print(f"Synthetic CSV saved as {outputPath}")
