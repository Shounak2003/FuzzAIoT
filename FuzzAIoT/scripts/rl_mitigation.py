import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random

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

# Define the DDQN Model (Q-Network)
class DDQN(nn.Module):
    def __init__(self):
        super(DDQN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Two possible actions (e.g., mitigate or not)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Training the DDQN Model
def train_ddqn_model():
    env = IoTEnv()
    model = DDQN()
    target_model = DDQN()  # Target network for stability
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    replay_buffer = ReplayBuffer(10000)
    gamma = 0.99
    batch_size = 64
    update_target_frequency = 10

    save_directory = "C:/Users/Shaurya/Downloads/FuzzAIoT/models/"
    os.makedirs(save_directory, exist_ok=True)

    for episode in range(1000):  # Training for 1000 episodes
        state = torch.tensor(env.reset(), dtype=torch.float32)
        done = False
        episode_loss = 0  # Track loss for each episode
        steps = 0

        while not done:
            # Epsilon-greedy action selection
            epsilon = max(0.01, 1.0 - episode / 500)
            if np.random.rand() > epsilon:
                with torch.no_grad():
                    action = torch.argmax(model(state)).item()
            else:
                action = np.random.choice([0, 1])

            next_state, reward, done = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)

                batch_state = torch.stack(batch_state)
                batch_action = torch.tensor(batch_action, dtype=torch.int64)
                batch_reward = torch.tensor(batch_reward, dtype=torch.float32)
                batch_next_state = torch.stack(batch_next_state)
                batch_done = torch.tensor(batch_done, dtype=torch.float32)

                current_q_values = model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze()
                max_next_q_values = target_model(batch_next_state).max(1)[0]
                expected_q_values = batch_reward + (gamma * max_next_q_values * (1 - batch_done))

                loss = criterion(current_q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                episode_loss += loss.item()

            if steps % update_target_frequency == 0:
                target_model.load_state_dict(model.state_dict())

            steps += 1

        avg_loss = episode_loss / steps if steps > 0 else 0
        print(f'Episode {episode + 1}, Loss: {avg_loss:.6f}')

    torch.save(model.state_dict(), os.path.join(save_directory, 'ddqn_mitigation_model.pth'))
    print(f"Model saved to {os.path.join(save_directory, 'ddqn_mitigation_model.pth')}")

if __name__ == "__main__":
    train_ddqn_model()
