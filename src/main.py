import random

import numpy as np
import pandas as pd

from src.log_config import logger

# --- DATA SETUP ---
# Real-world schedule (flights must be sorted by start time)
flights = [
    {"id": 101, "origin": "praga", "dest": "milan", "start": 600, "pass": 35},
    {"id": 102, "origin": "milan", "dest": "wieden", "start": 650, "pass": 126},
    {"id": 103, "origin": "wieden", "dest": "praga", "start": 700, "pass": 51},
    {"id": 104, "origin": "praga", "dest": "wieden", "start": 700, "pass": 137},
    {"id": 105, "origin": "milan", "dest": "wieden", "start": 710, "pass": 87},
    {"id": 106, "origin": "milan", "dest": "praga", "start": 1100, "pass": 99},
]

# Distances from LODZ (as you provided) + Inter-city distances
# We use a dictionary where (A, B) returns distance
dist_matrix = {
    # Distances from Lodz (Home)
    ("lodz", "milan"): 1388,
    ("lodz", "wieden"): 581,
    ("lodz", "praga"): 503,
    # Distances between cities (approximate or calculated)
    ("praga", "milan"): 849,
    ("milan", "wieden"): 789,
    ("wieden", "praga"): 287,
    ("milan", "praga"): 849,
    ("wieden", "milan"): 789,
    ("praga", "wieden"): 287,
    # Staying put
    ("lodz", "lodz"): 0,
    ("milan", "milan"): 0,
    ("wieden", "wieden"): 0,
    ("praga", "praga"): 0,
}

planes = ["boeing1", "boeing2"]
fuel_use = {"boeing1": 900.0, "boeing2": 900.0}
number_of_passengers = {"boeing1": 150, "boeing2": 150}
ticket_price = {"boeing1": 0.1, "boeing2": 0.1}  # per_km
speeds = {"boeing1": 900, "boeing2": 900}
FUEL_PRICE = 3
TIME_PENALTY_RATE = 2.0

# --- Q-LEARNING SETUP ---
# State: (B1_FreeTime, B2_FreeTime, B1_Loc, B2_Loc, FlightIdx)
initial_state = (0, 0, "lodz", "lodz", 0)
actions = [0, 1]
q_table = {}


def get_q(state):
    if state not in q_table:
        q_table[state] = np.zeros(len(actions))
    return q_table[state]


# --- DYNAMIC TIMELINE STEP FUNCTION ---
def take_step(state, action_idx):
    b1_time, b2_time, b1_loc, b2_loc, f_idx = state
    current_flight = flights[f_idx]

    p_name = planes[action_idx]
    p_free_time = b1_time if action_idx == 0 else b2_time
    p_loc = b1_loc if action_idx == 0 else b2_loc

    reloc_dist = dist_matrix.get((p_loc, current_flight["origin"]), 500)
    flight_dist = dist_matrix.get(
        (current_flight["origin"], current_flight["dest"]), 500
    )

    reloc_time = reloc_dist / (speeds[p_name] / 60)
    flight_time = flight_dist / (speeds[p_name] / 60)

    earliest_at_origin = p_free_time + reloc_time

    # Actual start time (If plane is late, flight starts when plane arrives)
    actual_start = max(current_flight["start"], earliest_at_origin)
    arrival_at_dest = actual_start + flight_time

    # PENALTIES & REWARDS
    delay = max(0, actual_start - current_flight["start"])
    delay_penalty = delay * 500  # $500 per minute late

    # Idle Penalty: Reward the agent for not letting planes sit around too long
    idle_time = max(0, current_flight["start"] - earliest_at_origin)
    idle_penalty = idle_time * 5  # $5 per minute waiting

    fuel_cost = (reloc_dist + flight_dist) * (fuel_use[p_name] / 100) * FUEL_PRICE
    revenue = (
        min(current_flight["pass"], number_of_passengers[p_name])
        * flight_dist
        * ticket_price[p_name]
    )

    reward = revenue - fuel_cost - delay_penalty - idle_penalty

    # UPDATE STATE
    new_times = [b1_time, b2_time]
    new_locs = [b1_loc, b2_loc]

    new_times[action_idx] = (
        arrival_at_dest  # Plane is now free at destination at this time
    )
    new_locs[action_idx] = current_flight["dest"]

    new_state = (new_times[0], new_times[1], new_locs[0], new_locs[1], f_idx + 1)
    return new_state, reward, actual_start, arrival_at_dest


# --- TRAINING ---
# Increase episodes as the state space is now much larger (due to time variables)
for _ in range(20000):
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

# --- FINAL SCHEDULE EXECUTION (ALIGNED WITH TIMELINE) ---
state = initial_state
total_profit = 0

logger.info("\n" + "=" * 95)
logger.info(
    f"{'FLIGHT':<8} | {'PLANE':<8} | {'ORIGIN':<10} | {'DEST':<10} | {'SCHED':<6} | {'ACTUAL':<6} | {'ARRIVE':<6} | {'PROFIT'}"
)
logger.info("-" * 95)

for i in range(len(flights)):
    # Determine best assignment from the Q-Table
    action_idx = np.argmax(get_q(state))
    p_name = planes[action_idx]

    # Extract current state details
    # State order: (B1_FreeTime, B2_FreeTime, B1_Loc, B2_Loc, FlightIdx)
    p1_time, p2_time, p1_loc, p2_loc, _ = state
    p_free_time = p1_time if action_idx == 0 else p2_time
    p_curr_loc = p1_loc if action_idx == 0 else p2_loc

    # Get flight details
    f = flights[i]

    # Calculate the Moving Timeline metrics
    # Relocation
    reloc_dist = dist_matrix.get((p_curr_loc, f["origin"]), 500)
    reloc_time = reloc_dist / (speeds[p_name] / 60)

    # Timeline points
    earliest_at_origin = p_free_time + reloc_time
    actual_start = max(f["start"], earliest_at_origin)

    flight_dist = dist_matrix.get((f["origin"], f["dest"]), 500)
    flight_time = flight_dist / (speeds[p_name] / 60)
    arrival_time = actual_start + flight_time

    # Step forward in the environment
    # We use take_step to get the reward and move the state
    next_state, reward, act_start, arr_time = take_step(state, action_idx)
    total_profit += reward

    # Formatting the Output
    # Add an asterisk if the plane was late to the scheduled start
    start_display = f"{actual_start:>4.0f}"
    if actual_start > f["start"]:
        start_display += "!"  # '!' indicates a delay

    # Add '*' if relocation was required
    origin_display = (
        f"{f['origin']}" if p_curr_loc == f["origin"] else f"{f['origin']}*"
    )

    print(
        f"{f['id']:<8} | {p_name.upper():<8} | {origin_display:<10} | {f['dest']:<10} | "
        f"{f['start']:>6.0f} | {start_display:<6} | {arrival_time:>6.0f} | ${reward:>8,.0f}"
    )

    state = next_state

logger.info("-" * 95)
logger.info(f"TOTAL SYSTEM PROFIT: ${total_profit:,.2f}")
logger.info("(*) Relocation flight required | (!) Flight delayed by plane availability")
logger.info("=" * 95)
