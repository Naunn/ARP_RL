import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler


class AttentionPoolingQNetwork(nn.Module):
    """
    Deep Sets Q-Network with learned attention over the flight lookahead window.
    Supports a pure mean-pooling aggregate baseline when use_attention=False.
    """

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

        # Conditionally allocate attention parameters to prevent parameter spillage
        if self.use_attention:
            attention_hidden_dim = max(16, hidden_dim // 2)
            self.flight_attention = nn.Sequential(
                nn.Linear(hidden_dim, attention_hidden_dim),
                nn.ReLU(),
                nn.Linear(attention_hidden_dim, 1),
            )
            # Input dimension to final layers: attended_context + max_context + fleet_dim
            final_input_dim = hidden_dim * 2 + fleet_dim
        else:
            self.flight_attention = None
            # Input dimension to final layers: mean_context + fleet_dim
            final_input_dim = hidden_dim + fleet_dim

        # The 'Rho' network: Processes aggregated features combined with aircraft statuses
        self.final_layers = nn.Sequential(
            nn.Linear(final_input_dim, hidden_dim),
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

        # Masking padding elements (where features are completely zeroed)
        valid_flights = flight_matrix.abs().sum(dim=-1) > 0
        encoded_flights = encoded_flights * valid_flights.unsqueeze(-1)

        mask_fill_value = torch.finfo(encoded_flights.dtype).min

        # Direct verification of the layer's existence satisfies Pylance type validation safely
        if self.use_attention and self.flight_attention is not None:
            # Compute attention weights over valid items
            attention_logits = self.flight_attention(encoded_flights).squeeze(-1)
            attention_logits = attention_logits.masked_fill(~valid_flights, mask_fill_value)
            attention_weights = torch.softmax(attention_logits, dim=1)

            # Extract weighted context
            context = torch.sum(encoded_flights * attention_weights.unsqueeze(-1), dim=1)

            # Extract max context (highest-pressure specific flight signature)
            masked_encoded = encoded_flights.masked_fill(~valid_flights.unsqueeze(-1), mask_fill_value)
            max_context = masked_encoded.max(dim=1).values
            max_context = torch.where(torch.isfinite(max_context), max_context, torch.zeros_like(max_context))

            # Concatenate all features
            combined_features = torch.cat([context, max_context, fleet_state], dim=1)
        else:
            # Clean aggregate mean pooling over active flights without maximum leaks
            valid_counts = valid_flights.sum(dim=1, keepdim=True).clamp(min=1)
            context = encoded_flights.sum(dim=1) / valid_counts.to(encoded_flights.dtype)

            # Concatenate features
            combined_features = torch.cat([context, fleet_state], dim=1)

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
        use_action_masking=True,
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
        self.use_action_masking = bool(use_action_masking)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"

        self.policy_net = AttentionPoolingQNetwork(
            fleet_dim,
            flight_feature_dim,
            hidden_dim=hidden_dim,
            n_actions=n_actions,
            use_attention=use_attention,
        ).to(self.device)

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

        if self.use_expert_bias:
            self.imitation_loss_fn = nn.CrossEntropyLoss()
        else:
            self.imitation_loss_fn = None

        self.memory = ReplayBuffer(capacity=20000, alpha=0.4, beta=0.4, beta_increment=5e-5)
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

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
        valid_mask = self._normalize_action_mask(action_mask if self.use_action_masking else None)
        valid_actions = np.flatnonzero(valid_mask)

        if use_epsilon and random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))

        fleet_state, flight_matrix = state_tuple
        with torch.inference_mode():
            fleet_t = self._to_device_tensor(fleet_state)
            flights_t = self._to_device_tensor(flight_matrix)
            q_values = self.policy_net(fleet_t, flights_t)

            mask_t = self._to_device_tensor(valid_mask, dtype=torch.bool).unsqueeze(0)
            masked_q = q_values.masked_fill(~mask_t, float("-inf"))
            return masked_q.argmax(dim=1).item()

    def decay_epsilon(self):
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
        safe_action_mask = self._normalize_action_mask(action_mask if self.use_action_masking else None)
        safe_next_action_mask = self._normalize_action_mask(next_action_mask if self.use_action_masking else None)
        self.memory.push(
            fleet_state,
            flight_matrix,
            action,
            reward,
            next_fleet,
            next_flights,
            done,
            safe_action_mask,
            safe_next_action_mask,
            expert_action,
        )

    def _polyak_update(self):
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
        if len(self.memory) < self.batch_size:
            return

        batch = self._sample_training_batch()
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
        ) = batch

        # Push everything to tensors
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

        with torch.autocast(device_type="cuda", enabled=self.use_amp):
            # Current Q-values
            q_logits = self.policy_net(fleet_t, flights_t)
            current_q = q_logits.gather(1, actions_t)

            # Target Q-values (Standard DQN: max over target network)
            with torch.no_grad():
                next_q = self.target_net(next_fleet_t, next_flights_t)
                if self.use_action_masking:
                    next_q = next_q.masked_fill(~next_action_masks_t, float("-inf"))

                max_next_q = next_q.max(dim=1, keepdim=True)[0]
                max_next_q = torch.nan_to_num(max_next_q, nan=0.0, posinf=1e4, neginf=-1e4)
                target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q

            # TD Loss
            loss, td_errors = self._compute_weighted_td_loss(current_q, target_q, weights_t)

            # Imitation Loss / Expert Bias (Masking Optional)
            if self.use_expert_bias and self.imitation_loss_fn is not None:
                expert_actions_t = self._to_device_tensor(expert_actions, dtype=torch.long)
                imitation_weight = 0.1 * float(self.epsilon)

                q_for_imitation = (
                    q_logits.masked_fill(~action_masks_t, float("-inf")) if self.use_action_masking else q_logits
                )
                imitation_loss = self.imitation_loss_fn(q_for_imitation, expert_actions_t)
                total_loss = loss + (imitation_weight * imitation_loss)
            else:
                total_loss = loss

        # Optimize & Update PER priorities
        self._optimizer_step(total_loss)
        self.memory.update_priorities(
            indices,
            np.atleast_1d(np.abs(td_errors.detach().cpu().numpy()).squeeze()) + 1e-6,
        )
        self._polyak_update()


class DoubleDQNAgent(DQNAgent):
    """Double DQN agent using policy network for action selection and target network for value evaluation."""

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self._sample_training_batch()
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
        ) = batch

        # Push everything to tensors
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

        with torch.autocast(device_type="cuda", enabled=self.use_amp):
            # Current Q-values
            q_logits = self.policy_net(fleet_t, flights_t)
            current_q = q_logits.gather(1, actions_t)

            # Target Q-values (Double DQN: Action selection from Policy, evaluation from Target)
            with torch.no_grad():
                next_policy_q = self.policy_net(next_fleet_t, next_flights_t)
                if self.use_action_masking:
                    next_policy_q = next_policy_q.masked_fill(~next_action_masks_t, float("-inf"))

                # Selection
                next_actions = next_policy_q.argmax(dim=1, keepdim=True)

                # Evaluation
                next_target_q = self.target_net(next_fleet_t, next_flights_t).gather(1, next_actions)
                next_target_q = torch.nan_to_num(next_target_q, nan=0.0, posinf=1e4, neginf=-1e4)
                target_q = rewards_t + (1 - dones_t) * self.gamma * next_target_q

            # TD Loss
            loss, td_errors = self._compute_weighted_td_loss(current_q, target_q, weights_t)

            # Imitation Loss / Expert Bias (Masking Optional)
            if self.use_expert_bias and self.imitation_loss_fn is not None:
                expert_actions_t = self._to_device_tensor(expert_actions, dtype=torch.long)
                imitation_weight = 0.1 * float(self.epsilon)

                q_for_imitation = (
                    q_logits.masked_fill(~action_masks_t, float("-inf")) if self.use_action_masking else q_logits
                )
                imitation_loss = self.imitation_loss_fn(q_for_imitation, expert_actions_t)
                total_loss = loss + (imitation_weight * imitation_loss)
            else:
                total_loss = loss

        # Optimize & Update PER priorities
        self._optimizer_step(total_loss)
        self.memory.update_priorities(
            indices,
            np.atleast_1d(np.abs(td_errors.detach().cpu().numpy()).squeeze()) + 1e-6,
        )
        self._polyak_update()
