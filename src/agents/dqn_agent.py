import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class QNetwork(nn.Module):
    """Multi-Layer Perceptron approximating the Q-Value function."""

    def __init__(self, state_dim, n_actions):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, state):
        return self.fc(state)


class ReplayBuffer:
    """Storage unit for Experience Replay to break temporal correlations."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim,
        n_actions,
        lr=1e-3,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.9995,
        min_epsilon=0.01,
        batch_size=64,
        tau=0.005,  # Changed: Replaced target_update_freq with tau
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.tau = tau  # Changed: Store Polyak update factor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Main Network (used to pick actions and updated every step)
        self.policy_net = QNetwork(state_dim, n_actions).to(self.device)
        # Target Network (held stable to compute steady bellman targets)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=20000)

    def choose_action(self, vector_state, use_epsilon=True):
        """Selects action using epsilon-greedy strategy over continuous states."""
        if use_epsilon and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(vector_state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def decay_epsilon(self):
        """Smoothly dials back exploration over training iterations."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def _polyak_update(self):
        """
        Changed: Replaced hard copy with soft updates.
        Blends weights fractionally from policy network to target network.
        """
        with torch.no_grad():
            for target_param, policy_param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
                )

    def learn(self):
        """Samples a batch and performs one step of gradient descent."""
        if len(self.memory) < self.batch_size:
            return

        # Sample data from experience buffer
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # Convert to PyTorch Tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Get predicted Q-values for the actions taken
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # Calculate target values using the Target Network (Bellman Equation)
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q

        # Compute Mean Squared Error Loss
        loss = nn.MSELoss()(current_q, target_q)

        # Optimize the Policy Network
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents gradient explosions from massive reward jumps
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Changed: Instead of step_count check, run soft update every training step
        self._polyak_update()
