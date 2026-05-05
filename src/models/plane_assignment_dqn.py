import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- LOGGING SETUP ---
from src.log_config import get_logger

logger = get_logger("DeepAirlines")

# --- DATA & CONSTANTS ---
CITY_MAP = {"lodz": 0, "praga": 1, "milan": 2, "wieden": 3}

dist_matrix = {
    ("lodz", "milan"): 1388,
    ("lodz", "wieden"): 581,
    ("lodz", "praga"): 503,
    ("praga", "milan"): 849,
    ("milan", "wieden"): 789,
    ("wieden", "praga"): 287,
    ("milan", "praga"): 849,
    ("wieden", "milan"): 789,
    ("praga", "wieden"): 287,
    ("lodz", "lodz"): 0,
    ("milan", "milan"): 0,
    ("wieden", "wieden"): 0,
    ("praga", "praga"): 0,
}

flights_data = [
    {"id": 101, "origin": "praga", "dest": "milan", "start": 600, "pass": 35},
    {"id": 102, "origin": "lodz", "dest": "wieden", "start": 655, "pass": 136},
    {"id": 103, "origin": "milan", "dest": "wieden", "start": 650, "pass": 136},
    {"id": 104, "origin": "milan", "dest": "lodz", "start": 660, "pass": 12},
    {"id": 105, "origin": "milan", "dest": "praga", "start": 665, "pass": 88},
    {"id": 106, "origin": "wieden", "dest": "praga", "start": 700, "pass": 51},
    {"id": 107, "origin": "praga", "dest": "wieden", "start": 700, "pass": 137},
    {"id": 108, "origin": "milan", "dest": "wieden", "start": 710, "pass": 87},
    {"id": 109, "origin": "milan", "dest": "praga", "start": 1100, "pass": 99},
]

# Actions: 0: Assign B1, 1: Assign B2, 2: Cancel
ACTIONS = [0, 1, 2]
CANCEL_PENALTY = -2000
FUEL_PRICE = 3
PLANE_CAPACITY = 150


# --- DEEP LEARNING MODEL ---
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.model = QNetwork(state_dim, action_dim)
        self.target = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.batch_size = 32

    def get_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.choice(ACTIONS)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        return torch.argmax(self.model(state_t)).item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        curr_q = self.model(states).gather(1, actions).squeeze()
        next_q = self.target(next_states).max(1)[0].detach()
        target_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = nn.MSELoss()(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)


# --- ENVIRONMENT LOGIC ---
def get_state_vector(b1_time, b2_time, b1_loc, b2_loc, f_idx):
    # Normalize time by 1440 mins and cities by mapping
    return np.array(
        [
            b1_time / 1440,
            b2_time / 1440,
            CITY_MAP[b1_loc] / 3,
            CITY_MAP[b2_loc] / 3,
            f_idx / len(flights_data),
        ],
        dtype=np.float32,
    )


def step(state_tuple, action, f_idx):
    b1_t, b2_t, b1_l, b2_l = state_tuple
    f = flights_data[f_idx]

    if action == 2:  # CANCEL
        return (b1_t, b2_t, b1_l, b2_l), CANCEL_PENALTY, "CANCELLED"

    # ASSIGNMENT LOGIC (Action 0 or 1)
    p_time = b1_t if action == 0 else b2_t
    p_loc = b1_l if action == 0 else b2_l

    reloc_dist = dist_matrix.get((p_loc, f["origin"]), 500)
    flight_dist = dist_matrix.get((f["origin"], f["dest"]), 500)

    reloc_time = reloc_dist / (900 / 60)
    flight_time = flight_dist / (900 / 60)

    actual_start = max(f["start"], p_time + reloc_time)
    arrival_time = actual_start + flight_time

    delay = max(0, actual_start - f["start"])
    revenue = min(f["pass"], PLANE_CAPACITY) * flight_dist * 1.0
    cost = (reloc_dist + flight_dist) * 9.0 * FUEL_PRICE / 100  # Approx fuel burn
    reward = revenue - cost - (delay * 100)  # $100 penalty per min delay

    if action == 0:
        new_state = (arrival_time, b2_t, f["dest"], b2_l)
    else:
        new_state = (b1_t, arrival_time, b1_l, f["dest"])

    return new_state, reward, f"ARRIVED @ {arrival_time:.0f}"


# --- MAIN EXECUTION ---
agent = DQNAgent(state_dim=5, action_dim=3)

# Training Loop
logger.info("Starting Training...")
for ep in range(1000):
    raw_state = (0, 0, "lodz", "praga")
    for i in range(len(flights_data)):
        state_vec = get_state_vector(*raw_state, i)
        action = agent.get_action(state_vec)
        next_raw, reward, _ = step(raw_state, action, i)

        done = 1 if i == len(flights_data) - 1 else 0
        next_vec = get_state_vector(*next_raw, i + 1) if not done else state_vec

        agent.memory.append((state_vec, action, reward / 1000, next_vec, done))
        agent.train_step()
        raw_state = next_raw

    if ep % 100 == 0:
        agent.target.load_state_dict(self_model := agent.model.state_dict())

# Final Evaluation & Logging
logger.info("\nFinal Optimized Schedule (Detailed):")
logger.info("=" * 110)
logger.info(
    f"{'ID':<5} | {'PLANE':<10} | {'FROM (Start)':<12} | {'TO (Origin)':<12} | {'DEST':<10} | {'STATUS':<15} | {'PROFIT'}"
)
logger.info("-" * 110)

eval_state = (0, 0, "lodz", "praga")  # Initial: B1 @ Lodz, B2 @ Praga
total_p = 0

for i in range(len(flights_data)):
    # 1. Capture current positions before the action
    b1_t, b2_t, b1_l, b2_l = eval_state

    # 2. Get Agent's Decision
    s_vec = get_state_vector(*eval_state, i)
    act = agent.get_action(s_vec, eval_mode=True)

    # 3. Identify which plane is starting from where
    if act == 0:
        p_name = "BOEING 1"
        start_loc = b1_l
    elif act == 1:
        p_name = "BOEING 2"
        start_loc = b2_l
    else:
        p_name = "N/A"
        start_loc = "CANCELLED"

    # 4. Execute step
    next_s, rew, status = step(eval_state, act, i)
    f = flights_data[i]

    # 5. Log details
    # We show: Plane, where it was parked, where it had to go to pick up pax, and final destination
    origin_display = f"{f['origin']}" if start_loc == f["origin"] else f"{f['origin']}*"

    logger.info(
        f"{f['id']:<5} | {p_name:<10} | {start_loc:<12} | {origin_display:<12} | "
        f"{f['dest']:<10} | {status:<15} | ${rew:>8,.0f}"
    )

    eval_state = next_s
    total_p += rew

logger.info("-" * 110)
logger.info(f"TOTAL SYSTEM PROFIT: ${total_p:,.2f}")
logger.info("(*) Asterisk indicates a relocation flight was required.")
logger.info("=" * 110)
