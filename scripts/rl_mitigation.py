import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# Environment setup (This is a simple simulation; you may need a more complex environment based on your requirements)
class IoTEnv:
    def __init__(self):
        self.state = self.reset()

    def reset(self):
        # Reset the environment to an initial state
        self.state = np.random.rand(4)  # Example: 4 features representing the state
        return self.state

    def step(self, action):
        # Define how the environment responds to an action
        reward = -1 if action == 0 else 1  # Example reward structure
        next_state = np.random.rand(4)  # Example next state
        done = np.random.rand() > 0.95  # Randomly end the episode
        return next_state, reward, done

# Define the RL Model (Q-Network)
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Two possible actions (e.g., mitigate or not)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Training the RL Model
def train_rl_model():
    env = IoTEnv()
    model = QNetwork()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    save_directory = "C:/Users/Shaurya/Downloads/FuzzAIoT/models/"
    os.makedirs(save_directory, exist_ok=True)

    for episode in range(100):  # Training for 100 episodes
        state = torch.tensor(env.reset(), dtype=torch.float32)
        done = False
        while not done:
            # Choose action (simple epsilon-greedy)
            q_values = model(state)
            if np.random.rand() > 0.5:
                action = torch.argmax(q_values).item()
            else:
                action = np.random.choice([0, 1])

            next_state, reward, done = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            # Compute target
            target_q_value = reward + 0.99 * torch.max(model(next_state)).item()

            # Update Q-Value
            target_q_values = q_values.clone().detach()
            target_q_values[action] = target_q_value
            loss = criterion(q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        print(f'Episode {episode + 1}, Loss: {loss.item()}')

    torch.save(model.state_dict(), os.path.join(save_directory, 'rl_mitigation_model.pth'))
    print(f"Model saved to {os.path.join(save_directory, 'rl_mitigation_model.pth')}")

if __name__ == "__main__":
    train_rl_model()
