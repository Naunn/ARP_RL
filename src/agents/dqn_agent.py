import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class AttentionPoolingQNetwork(nn.Module):
    """Deep Sets Q-Network using max-pooling to achieve sequence-length invariance."""

    def __init__(self, fleet_dim, flight_feature_dim, hidden_dim=128, n_actions=3):
        super(AttentionPoolingQNetwork, self).__init__()

        # The 'Phi' network: Processes each individual flight's features independently
        self.flight_encoder = nn.Sequential(
            nn.Linear(flight_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # The 'Rho' network: Processes the aggregated global schedule combined with active aircraft statuses
        self.final_layers = nn.Sequential(
            nn.Linear(hidden_dim + fleet_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, fleet_state, flight_matrix):
        # Enforce batch tracking dimensions if single items are passed
        if fleet_state.dim() == 1:
            fleet_state = fleet_state.unsqueeze(0)
        if flight_matrix.dim() == 2:
            flight_matrix = flight_matrix.unsqueeze(0)

        # Encode all flights into hidden feature matrices
        encoded_flights = self.flight_encoder(flight_matrix)

        # Symmetric Mean-Pooling along the flight sequence dimension (dim=1)
        # This preserves schedule distribution information and avoids losing signal
        # from the current/active flight feature embedding.
        pooled_context = encoded_flights.mean(dim=1)

        # Combine systemic context bottleneck with static plane allocations
        combined_features = torch.cat([pooled_context, fleet_state], dim=1)

        # Map projections to Q-values per plane asset
        return self.final_layers(combined_features)


class ReplayBuffer:
    """Prioritized experience replay buffer."""

    def __init__(self, capacity=20000, alpha=0.6, beta=0.4, beta_increment=1e-4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(
        self, fleet_state, flight_matrix, action, reward, next_fleet, next_flights, done
    ):
        transition = (
            fleet_state,
            flight_matrix,
            action,
            reward,
            next_fleet,
            next_flights,
            done,
        )

        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        priorities = self.priorities[: len(self.buffer)]
        scaled_priorities = priorities**self.alpha
        sampling_probabilities = scaled_priorities / scaled_priorities.sum()

        indices = np.random.choice(
            len(self.buffer), batch_size, p=sampling_probabilities
        )
        samples = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * sampling_probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Unpack elements independently to build structural numpy arrays cleanly
        fleet_b = np.array([s[0] for s in samples], dtype=np.float32)
        flights_b = np.array([s[1] for s in samples], dtype=np.float32)
        action_b = np.array([s[2] for s in samples], dtype=np.int64)
        reward_b = np.array([s[3] for s in samples], dtype=np.float32)
        next_fleet_b = np.array([s[4] for s in samples], dtype=np.float32)
        next_flights_b = np.array([s[5] for s in samples], dtype=np.float32)
        done_b = np.array([s[6] for s in samples], dtype=np.float32)

        return (
            indices,
            weights.astype(np.float32),
            fleet_b,
            flights_b,
            action_b,
            reward_b,
            next_fleet_b,
            next_flights_b,
            done_b,
        )

    def __len__(self):
        return len(self.buffer)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = max(float(priority), 1e-6)


class DQNAgent:
    def __init__(
        self,
        fleet_dim,
        flight_feature_dim,
        n_actions,
        lr=1e-3,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.9995,
        min_epsilon=0.01,
        batch_size=64,
        tau=0.005,
    ):
        self.fleet_dim = fleet_dim
        self.flight_feature_dim = flight_feature_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.tau = tau

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Main Network (used to pick actions and updated every step)
        self.policy_net = AttentionPoolingQNetwork(
            fleet_dim, flight_feature_dim, n_actions=n_actions
        ).to(self.device)
        # Target Network (held stable to compute steady bellman targets)
        self.target_net = AttentionPoolingQNetwork(
            fleet_dim, flight_feature_dim, n_actions=n_actions
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.memory = ReplayBuffer(capacity=20000)

    def choose_action(self, state_tuple, use_epsilon=True):
        """Selects action using epsilon-greedy strategy over structural environment states."""
        if use_epsilon and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        # Unpack the tuple internal array layers inside the method scope
        fleet_state, flight_matrix = state_tuple

        with torch.no_grad():
            fleet_t = torch.FloatTensor(fleet_state).to(self.device)
            flights_t = torch.FloatTensor(flight_matrix).to(self.device)
            q_values = self.policy_net(fleet_t, flights_t)
            return q_values.argmax(dim=1).item()

    def decay_epsilon(self):
        """Smoothly dials back exploration over training iterations."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def store_transition(
        self, fleet_state, flight_matrix, action, reward, next_fleet, next_flights, done
    ):
        """Pushes a structural state transformation tracking sequence to the replay memory."""
        self.memory.push(
            fleet_state, flight_matrix, action, reward, next_fleet, next_flights, done
        )

    def _polyak_update(self):
        """Blends weights fractionally from policy network to target network."""
        with torch.no_grad():
            for target_param, policy_param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
                )

    def _sample_training_batch(self):
        return self.memory.sample(self.batch_size)

    def _compute_weighted_td_loss(self, current_q, target_q, weights_t):
        td_errors = current_q - target_q
        per_sample_loss = self.loss_fn(current_q, target_q)
        loss = (weights_t * per_sample_loss).mean()
        return loss, td_errors

    def learn(self):
        """Samples a multi-component batch and performs one step of gradient descent."""
        if len(self.memory) < self.batch_size:
            return

        # Sample data arrays from experience buffer
        (
            indices,
            weights,
            fleet_b,
            flights_b,
            actions,
            rewards,
            next_fleet_b,
            next_flights_b,
            dones,
        ) = self._sample_training_batch()

        # Convert layout blocks into PyTorch Tensors
        fleet_t = torch.FloatTensor(fleet_b).to(self.device)
        flights_t = torch.FloatTensor(flights_b).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_fleet_t = torch.FloatTensor(next_fleet_b).to(self.device)
        next_flights_t = torch.FloatTensor(next_flights_b).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights_t = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # Get predicted Q-values for the actions taken
        current_q = self.policy_net(fleet_t, flights_t).gather(1, actions_t)

        # Calculate target values using the Target Network (Bellman Equation)
        with torch.no_grad():
            max_next_q = self.target_net(next_fleet_t, next_flights_t).max(
                dim=1, keepdim=True
            )[0]
            target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q

        loss, td_errors = self._compute_weighted_td_loss(current_q, target_q, weights_t)

        # Optimize the Policy Network
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.memory.update_priorities(
            indices,
            np.atleast_1d(np.abs(td_errors.detach().cpu().numpy()).squeeze()) + 1e-6,
        )

        # Execute soft target updating
        self._polyak_update()


class DoubleDQNAgent(DQNAgent):
    """Double DQN agent using policy network for action selection and target network for value evaluation."""

    def learn(self):
        """Performs a DDQN update step using the policy network for argmax selection."""
        if len(self.memory) < self.batch_size:
            return

        (
            indices,
            weights,
            fleet_b,
            flights_b,
            actions,
            rewards,
            next_fleet_b,
            next_flights_b,
            dones,
        ) = self._sample_training_batch()

        fleet_t = torch.FloatTensor(fleet_b).to(self.device)
        flights_t = torch.FloatTensor(flights_b).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_fleet_t = torch.FloatTensor(next_fleet_b).to(self.device)
        next_flights_t = torch.FloatTensor(next_flights_b).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights_t = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        current_q = self.policy_net(fleet_t, flights_t).gather(1, actions_t)

        with torch.no_grad():
            next_policy_q = self.policy_net(next_fleet_t, next_flights_t)
            next_actions = next_policy_q.argmax(dim=1, keepdim=True)
            next_target_q = self.target_net(next_fleet_t, next_flights_t).gather(
                1, next_actions
            )
            target_q = rewards_t + (1 - dones_t) * self.gamma * next_target_q

        loss, td_errors = self._compute_weighted_td_loss(current_q, target_q, weights_t)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.memory.update_priorities(
            indices,
            np.atleast_1d(np.abs(td_errors.detach().cpu().numpy()).squeeze()) + 1e-6,
        )

        self._polyak_update()
