import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class AttentionPoolingQNetwork(nn.Module):
    """Deep Sets Q-Network with learned attention over the flight lookahead window."""

    def __init__(
        self,
        fleet_dim,
        flight_feature_dim,
        hidden_dim=128,
        n_actions=3,
        use_attention=True,
    ):
        super(AttentionPoolingQNetwork, self).__init__()
        self.use_attention = bool(use_attention)

        # The 'Phi' network: Processes each individual flight's features independently
        self.flight_encoder = nn.Sequential(
            nn.Linear(flight_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        attention_hidden_dim = max(16, hidden_dim // 2)
        self.flight_attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_hidden_dim),
            nn.ReLU(),
            nn.Linear(attention_hidden_dim, 1),
        )

        # The 'Rho' network: Processes the aggregated global schedule combined with active aircraft statuses
        self.final_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2 + fleet_dim, hidden_dim),
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

        valid_flights = flight_matrix.abs().sum(dim=-1) > 0
        encoded_flights = encoded_flights * valid_flights.unsqueeze(-1)

        mask_fill_value = torch.finfo(encoded_flights.dtype).min

        if self.use_attention:
            attention_logits = self.flight_attention(encoded_flights).squeeze(-1)
            attention_logits = attention_logits.masked_fill(~valid_flights, mask_fill_value)
            attention_weights = torch.softmax(attention_logits, dim=1)
            attended_context = torch.sum(
                encoded_flights * attention_weights.unsqueeze(-1),
                dim=1,
            )
        else:
            valid_counts = valid_flights.sum(dim=1, keepdim=True).clamp(min=1)
            attended_context = encoded_flights.sum(dim=1) / valid_counts.to(encoded_flights.dtype)

        masked_encoded = encoded_flights.masked_fill(
            ~valid_flights.unsqueeze(-1),
            mask_fill_value,
        )
        max_context = masked_encoded.max(dim=1).values
        max_context = torch.where(
            torch.isfinite(max_context),
            max_context,
            torch.zeros_like(max_context),
        )

        # Combine the highest-pressure flight signal with a learned aggregate context.
        combined_features = torch.cat([attended_context, max_context, fleet_state], dim=1)

        # Map projections to Q-values per plane asset
        return self.final_layers(combined_features)


class ReplayBuffer:
    """Prioritized experience replay buffer."""

    def __init__(self, capacity=20000, alpha=0.4, beta=0.4, beta_increment=5e-5):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(
        self,
        fleet_state,
        flight_matrix,
        action,
        reward,
        next_fleet,
        next_flights,
        done,
        action_mask,
        next_action_mask,
        expert_action,
    ):
        transition = (
            fleet_state,
            flight_matrix,
            action,
            reward,
            next_fleet,
            next_flights,
            done,
            action_mask,
            next_action_mask,
            expert_action,
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

        indices = np.random.choice(len(self.buffer), batch_size, p=sampling_probabilities)
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
        action_mask_b = np.array([s[7] for s in samples], dtype=np.bool_)
        next_action_mask_b = np.array([s[8] for s in samples], dtype=np.bool_)
        expert_action_b = np.array([s[9] for s in samples], dtype=np.int64)

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
            action_mask_b,
            next_action_mask_b,
            expert_action_b,
        )

    def __len__(self):
        return len(self.buffer)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            priority = float(np.nan_to_num(priority, nan=1.0, posinf=1.0, neginf=1.0))
            self.priorities[idx] = max(priority, 1e-6)


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
        hidden_dim=256,
        use_attention=True,
        use_expert_bias=True,
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
        self.use_expert_bias = bool(use_expert_bias)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"

        # Main Network (used to pick actions and updated every step)
        self.policy_net = AttentionPoolingQNetwork(
            fleet_dim,
            flight_feature_dim,
            hidden_dim=hidden_dim,
            n_actions=n_actions,
            use_attention=use_attention,
        ).to(self.device)
        # Target Network (held stable to compute steady bellman targets)
        self.target_net = AttentionPoolingQNetwork(
            fleet_dim,
            flight_feature_dim,
            hidden_dim=hidden_dim,
            n_actions=n_actions,
            use_attention=use_attention,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.imitation_loss_fn = nn.CrossEntropyLoss()
        self.memory = ReplayBuffer(capacity=20000, alpha=0.4, beta=0.4, beta_increment=5e-5)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)  # type: ignore

    def _to_device_tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _optimizer_step(self, loss):
        self.optimizer.zero_grad(set_to_none=True)

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()

    def _normalize_action_mask(self, action_mask):
        if action_mask is None:
            return np.ones(self.n_actions, dtype=np.bool_)
        mask = np.asarray(action_mask, dtype=np.bool_).reshape(-1)
        if mask.size != self.n_actions:
            raise ValueError(f"Action mask size {mask.size} does not match n_actions={self.n_actions}.")
        if not mask.any():
            mask[:] = True
        return mask

    def choose_action(self, state_tuple, action_mask=None, use_epsilon=True):
        """Selects action using epsilon-greedy strategy over structural environment states."""
        valid_mask = self._normalize_action_mask(action_mask)
        valid_actions = np.flatnonzero(valid_mask)

        if use_epsilon and random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))

        # Unpack the tuple internal array layers inside the method scope
        fleet_state, flight_matrix = state_tuple

        with torch.inference_mode():
            fleet_t = self._to_device_tensor(fleet_state)
            flights_t = self._to_device_tensor(flight_matrix)
            q_values = self.policy_net(fleet_t, flights_t)

            mask_t = self._to_device_tensor(valid_mask, dtype=torch.bool).unsqueeze(0)
            masked_q = q_values.masked_fill(~mask_t, float("-inf"))
            return masked_q.argmax(dim=1).item()

    def decay_epsilon(self):
        """Smoothly dials back exploration over training iterations."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def store_transition(
        self,
        fleet_state,
        flight_matrix,
        action,
        reward,
        next_fleet,
        next_flights,
        done,
        action_mask,
        next_action_mask,
        expert_action,
    ):
        """Pushes a structural state transformation tracking sequence to the replay memory."""
        self.memory.push(
            fleet_state,
            flight_matrix,
            action,
            reward,
            next_fleet,
            next_flights,
            done,
            action_mask,
            next_action_mask,
            expert_action,
        )

    def _polyak_update(self):
        """Blends weights fractionally from policy network to target network."""
        with torch.no_grad():
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def _sample_training_batch(self):
        return self.memory.sample(self.batch_size)

    def _compute_weighted_td_loss(self, current_q, target_q, weights_t):
        current_q = torch.nan_to_num(current_q, nan=0.0, posinf=1e4, neginf=-1e4)
        target_q = torch.nan_to_num(target_q, nan=0.0, posinf=1e4, neginf=-1e4)
        td_errors = current_q - target_q
        td_errors = torch.nan_to_num(td_errors, nan=0.0, posinf=1e4, neginf=-1e4)
        per_sample_loss = self.loss_fn(current_q, target_q)
        per_sample_loss = torch.nan_to_num(per_sample_loss, nan=0.0, posinf=1e4, neginf=1e4)
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
            action_masks,
            next_action_masks,
            expert_actions,
        ) = self._sample_training_batch()

        # Convert layout blocks into PyTorch Tensors
        fleet_t = self._to_device_tensor(fleet_b)
        flights_t = self._to_device_tensor(flights_b)
        actions_t = self._to_device_tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards_t = self._to_device_tensor(rewards).unsqueeze(1)
        next_fleet_t = self._to_device_tensor(next_fleet_b)
        next_flights_t = self._to_device_tensor(next_flights_b)
        dones_t = self._to_device_tensor(dones).unsqueeze(1)
        weights_t = self._to_device_tensor(weights).unsqueeze(1)
        action_masks_t = self._to_device_tensor(action_masks, dtype=torch.bool)
        next_action_masks_t = self._to_device_tensor(next_action_masks, dtype=torch.bool)
        expert_actions_t = self._to_device_tensor(expert_actions, dtype=torch.long)

        with torch.autocast(device_type="cuda", enabled=self.use_amp):
            q_logits = self.policy_net(fleet_t, flights_t)
            current_q = q_logits.gather(1, actions_t)

            # Calculate target values using the Target Network (Bellman Equation)
            with torch.no_grad():
                next_q = self.target_net(next_fleet_t, next_flights_t)
                masked_next_q = next_q.masked_fill(~next_action_masks_t, float("-inf"))
                max_next_q = masked_next_q.max(dim=1, keepdim=True)[0]
                max_next_q = torch.nan_to_num(max_next_q, nan=0.0, posinf=1e4, neginf=-1e4)
                target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q

            loss, td_errors = self._compute_weighted_td_loss(current_q, target_q, weights_t)
            if self.use_expert_bias:
                imitation_weight = 0.1 * float(self.epsilon)
                masked_logits = q_logits.masked_fill(~action_masks_t, float("-inf"))
                imitation_loss = self.imitation_loss_fn(masked_logits, expert_actions_t)
                total_loss = loss + (imitation_weight * imitation_loss)
            else:
                total_loss = loss

        # Optimize the Policy Network
        self._optimizer_step(total_loss)

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
            action_masks,
            next_action_masks,
            expert_actions,
        ) = self._sample_training_batch()

        fleet_t = self._to_device_tensor(fleet_b)
        flights_t = self._to_device_tensor(flights_b)
        actions_t = self._to_device_tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards_t = self._to_device_tensor(rewards).unsqueeze(1)
        next_fleet_t = self._to_device_tensor(next_fleet_b)
        next_flights_t = self._to_device_tensor(next_flights_b)
        dones_t = self._to_device_tensor(dones).unsqueeze(1)
        weights_t = self._to_device_tensor(weights).unsqueeze(1)
        action_masks_t = self._to_device_tensor(action_masks, dtype=torch.bool)
        next_action_masks_t = self._to_device_tensor(next_action_masks, dtype=torch.bool)
        expert_actions_t = self._to_device_tensor(expert_actions, dtype=torch.long)

        with torch.autocast(device_type="cuda", enabled=self.use_amp):
            q_logits = self.policy_net(fleet_t, flights_t)
            current_q = q_logits.gather(1, actions_t)

            with torch.no_grad():
                next_policy_q = self.policy_net(next_fleet_t, next_flights_t)
                masked_next_policy_q = next_policy_q.masked_fill(
                    ~next_action_masks_t,
                    float("-inf"),
                )
                next_actions = masked_next_policy_q.argmax(dim=1, keepdim=True)
                next_target_q = self.target_net(next_fleet_t, next_flights_t).gather(1, next_actions)
                next_target_q = torch.nan_to_num(next_target_q, nan=0.0, posinf=1e4, neginf=-1e4)
                target_q = rewards_t + (1 - dones_t) * self.gamma * next_target_q

            loss, td_errors = self._compute_weighted_td_loss(current_q, target_q, weights_t)
            if self.use_expert_bias:
                imitation_weight = 0.1 * float(self.epsilon)
                masked_logits = q_logits.masked_fill(~action_masks_t, float("-inf"))
                imitation_loss = self.imitation_loss_fn(masked_logits, expert_actions_t)
                total_loss = loss + (imitation_weight * imitation_loss)
            else:
                total_loss = loss

        self._optimizer_step(total_loss)

        self.memory.update_priorities(
            indices,
            np.atleast_1d(np.abs(td_errors.detach().cpu().numpy()).squeeze()) + 1e-6,
        )

        self._polyak_update()
