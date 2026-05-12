import time

import numpy as np

from src.agents.q_learning_agent import QAgent
from src.log_config import get_logger
from src.utils.dist import create_dist_dict
from src.utils.envs import AirlineEnv
from src.utils.fleet import generate_fleet
from src.utils.schedule import (
    calculate_max_concurrency,
    check_global_feasibility,
    generate_random_flights,
)

logger = get_logger("plane_assignment")

# --- DATA SETUP ---
cities_list = ["praga", "milan", "wieden", "lodz", "berlin", "paris"]
n_flights = 25

FLIGHTS = generate_random_flights(
    n=n_flights,
    cities=cities_list,
    start_time_range=(5 * 60, 23 * 60),
    pass_range=(10, 180),
)

dist_dict = create_dist_dict(cities_list)

# --- FLEET GENERATION ---
fleet_config = {"BOEING": 5, "AIRBUS": 3}
PLANES = generate_fleet(fleet_config)

# Add custom one-off planes
PLANES["private_jet_custom"] = {
    "fuel_use": 400.0,
    "seats": 10,
    "speed": 950,
    "base_fare": 500,
    "rate_per_km": 0.80,
}

# Log the generated fleet for verification
logger.info(f"Fleet generated with {len(PLANES)} aircraft:")
for name in PLANES.keys():
    logger.info(f" - {name.upper()}")

# --- FEASIBILITY CHECKS ---
utilization = check_global_feasibility(FLIGHTS, list(PLANES.keys()), PLANES, dist_dict)
max_req_planes = calculate_max_concurrency(FLIGHTS, PLANES, dist_dict)
num_planes = len(PLANES)

logger.info("=" * 60)
logger.info("SCHEDULE FEASIBILITY REPORT")
logger.info(f"Total Flights: {n_flights} | Fleet Size: {num_planes}")
logger.info(f"Global Utilization: {utilization:.1f}%")
logger.info(f"Peak Concurrency: {max_req_planes} planes required simultaneously")

if utilization > 100 or max_req_planes > num_planes:
    logger.warning(
        "!!! WARNING: Schedule may be physically UNSOLVABLE without major delays !!!"
    )
elif utilization > 80:
    logger.info(
        "Status: Schedule is very tight. Expect low rewards due to relocation needs."
    )
else:
    logger.info("Status: Schedule looks healthy.")
logger.info("=" * 60)

# --- INITIALIZE ENV & AGENT ---
env = AirlineEnv(FLIGHTS, PLANES, dist_dict)
agent = QAgent(n_actions=len(env.planes))

# --- TRAINING SETTINGS ---
n_episodes = 1_000_000
log_interval = 10_000
scores = []
start_wall_time = time.time()

logger.info(f"Starting training for {n_episodes:,} episodes...")

for i in range(1, n_episodes + 1):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

    scores.append(episode_reward)

    agent.decay_epsilon()

    if i % log_interval == 0:
        avg_score = np.mean(scores[-log_interval:])
        percent_complete = (i / n_episodes) * 100
        episodes_left = n_episodes - i

        elapsed_time = time.time() - start_wall_time
        avg_time_per_ep = elapsed_time / i
        remaining_sec = int(episodes_left * avg_time_per_ep)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining_sec))

        logger.info(
            f"Progress: {percent_complete:>5.1f}% | "
            f"Left: {episodes_left:>7,} | "
            f"Avg Profit: ${avg_score:>10.0f} | "
            f"ETA: {eta_str}"
        )

logger.info("Training complete. Moving to final schedule execution...")

# --- FINAL SCHEDULE EXECUTION ---
state = env.reset()
total_profit = 0
done = False

header = f"{'FLIGHT':<8} | {'PLANE':<14} | {'ORIGIN':<12} | {'DEST':<10} | {'SCHED':<7} | {'ACTUAL':<7} | {'ARRIVE':<7} | {'PROFIT'}"
line_width = len(header) + 2

logger.info("\n" + "=" * line_width)
logger.info(header)
logger.info("-" * line_width)

while not done:
    f = FLIGHTS[env.current_f_idx]
    action = agent.choose_action(state, use_epsilon=False)
    p_name = env.planes[action]

    next_state, reward, done, info = env.step(action)
    total_profit += reward

    delay_flag = "!" if info["actual_start"] > f["start"] else ""
    start_display = f"{info['actual_start']:>4.0f}{delay_flag}"
    reloc_flag = "*" if info["p_curr_loc"] != f["origin"] else ""
    origin_display = f"{f['origin']}{reloc_flag}"

    logger.info(
        f"{f['id']:<8} | {p_name.upper():<14} | {origin_display:<12} | {f['dest']:<10} | "
        f"{f['start']:>7.0f} | {start_display:<7} | {info['arrival_at_dest']:>7.0f} | ${reward:>10,.0f}"
    )
    state = next_state

logger.info("-" * line_width)
logger.info(f"TOTAL SYSTEM PROFIT: ${total_profit:>12,.2f}")
logger.info(
    f"(*) Relocation required | (!) Flight delayed | Plane Count: {len(env.planes)}"
)
logger.info("=" * line_width)
