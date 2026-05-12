import random

import numpy as np
import pandas as pd

from src.log_config import get_logger
from src.utils.dist import create_dist_dict

logger = get_logger("plane_assignment")

# --- DATA SETUP ---
flights = [
    {"id": 101, "origin": "praga", "dest": "milan", "start": 600, "pass": 35},
    {"id": 103, "origin": "milan", "dest": "wieden", "start": 650, "pass": 136},
    {"id": 104, "origin": "milan", "dest": "lodz", "start": 660, "pass": 12},
    {"id": 105, "origin": "milan", "dest": "praga", "start": 665, "pass": 88},
    {"id": 106, "origin": "wieden", "dest": "praga", "start": 700, "pass": 51},
    {"id": 107, "origin": "praga", "dest": "wieden", "start": 700, "pass": 137},
    {"id": 108, "origin": "milan", "dest": "wieden", "start": 710, "pass": 87},
    {"id": 109, "origin": "milan", "dest": "praga", "start": 1100, "pass": 99},
]

dist_dict = create_dist_dict(
    list({f["origin"] for f in flights} | {f["dest"] for f in flights})
)

FUEL_PRICE = 3

# --- PLANE CONFIGURATION ---
plane_configs = {
    "boeing1": {
        "fuel_use": 900.0,
        "seats": 150,
        "speed": 900,
        "base_fare": 50,
        "rate_per_km": 0.15,
    },
    "boeing2": {
        "fuel_use": 900.0,
        "seats": 150,
        "speed": 900,
        "base_fare": 50,
        "rate_per_km": 0.15,
    },
    "airbus_premium": {
        "fuel_use": 850.0,
        "seats": 120,
        "speed": 850,
        "base_fare": 120,
        "rate_per_km": 0.28,
    },
}

planes = list(plane_configs.keys())
actions = list(range(len(planes)))

initial_state = (tuple([0] * len(planes)), tuple(["lodz"] * len(planes)), 0)
q_table = {}


def get_q(state):
    times, locs, f_idx = state
    rounded_times = tuple(round(t, -1) for t in times)
    compact_state = (rounded_times, locs, f_idx)
    if compact_state not in q_table:
        q_table[compact_state] = np.zeros(len(actions))
    return q_table[compact_state]


def take_step(state, action_idx):
    times, locs, f_idx = state
    current_flight = flights[f_idx]
    p_name = planes[action_idx]
    p_cfg = plane_configs[p_name]

    p_free_time = times[action_idx]
    p_loc = locs[action_idx]

    reloc_dist = dist_dict.get((p_loc, current_flight["origin"]), 500)
    flight_dist = dist_dict.get((current_flight["origin"], current_flight["dest"]), 500)

    reloc_time = reloc_dist / (p_cfg["speed"] / 60)
    flight_time = flight_dist / (p_cfg["speed"] / 60)

    actual_start = max(current_flight["start"], p_free_time + reloc_time)
    arrival_at_dest = actual_start + flight_time

    # Revenue & Cost
    ticket_price = p_cfg["base_fare"] + (flight_dist * p_cfg["rate_per_km"])
    revenue = min(current_flight["pass"], p_cfg["seats"]) * ticket_price
    fuel_cost = (reloc_dist + flight_dist) * (p_cfg["fuel_use"] / 100) * FUEL_PRICE
    delay_penalty = max(0, actual_start - current_flight["start"]) * 500

    reward = revenue - fuel_cost - delay_penalty

    new_times = list(times)
    new_locs = list(locs)
    new_times[action_idx] = arrival_at_dest
    new_locs[action_idx] = current_flight["dest"]

    return (
        (tuple(new_times), tuple(new_locs), f_idx + 1),
        reward,
        actual_start,
        arrival_at_dest,
    )


# --- TRAINING ---
for _ in range(50000):  # Increased iterations for larger state space
    state = initial_state
    for f_idx in range(len(flights)):
        action = (
            random.choice(actions) if random.random() < 0.2 else np.argmax(get_q(state))
        )
        next_state, reward, _, _ = take_step(state, action)
        old_q = get_q(state)[action]
        next_max = 0 if (f_idx == len(flights) - 1) else np.max(get_q(next_state))
        get_q(state)[action] = old_q + 0.1 * (reward + 0.9 * next_max - old_q)
        state = next_state

# --- FINAL SCHEDULE EXECUTION ---
logger.info(f"Schedule:\n{pd.DataFrame(flights)}")
state = initial_state
total_profit = 0

logger.info("\n" + "=" * 105)
logger.info(
    f"{'FLIGHT':<8} | {'PLANE':<14} | {'ORIGIN':<12} | {'DEST':<10} | {'SCHED':<6} | {'ACTUAL':<6} | {'ARRIVE':<6} | {'PROFIT'}"
)
logger.info("-" * 105)

for i in range(len(flights)):
    action_idx = np.argmax(get_q(state))
    p_name = planes[action_idx]
    p_cfg = plane_configs[p_name]

    # Correctly unpack fleet state
    times, locs, _ = state
    p_free_time = times[action_idx]
    p_curr_loc = locs[action_idx]

    f = flights[i]

    # Step through the logic to get final metrics
    next_state, reward, act_start, arr_time = take_step(state, action_idx)
    total_profit += reward

    # Formatting
    start_display = f"{act_start:>4.0f}{'!' if act_start > f['start'] else ' '}"
    origin_display = f"{f['origin']}{'*' if p_curr_loc != f['origin'] else ''}"

    logger.info(
        f"{f['id']:<8} | {p_name.upper():<14} | {origin_display:<12} | {f['dest']:<10} | "
        f"{f['start']:>6.0f} | {start_display:<6} | {arr_time:>6.0f} | ${reward:>8,.0f}"
    )

    state = next_state

logger.info("-" * 105)
logger.info(f"TOTAL SYSTEM PROFIT: ${total_profit:,.2f}")
logger.info(
    "(*) Relocation required | (!) Flight delayed | Plane Count: " + str(len(planes))
)
logger.info("=" * 105)
