"""Gurobi vs VNS vs Double DQN on the same schedule, optionally with Gurobi and VNS limited to the DQN's time.

    python -m src.workspace.compare_methods

Order: the Double DQN is trained and builds its plan first (so its time is known), then Gurobi and VNS solve
the same schedule. Every plan is scored by replaying it through the same AirlineEnv.
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.dqn_agent import DoubleDQNAgent
from src.baselines import BaselineSolverError
from src.baselines.gurobi_solver import solve_assignment
from src.config import MODEL_HYPERPARAMS, REWARD_CONFIG
from src.instances import build_flight_pool, build_planes
from src.utils.dist import create_dist_dict_from_airports
from src.utils.envs import AirlineEnv, ClosestPlaneGreedySolver
from src.utils.seeding import set_seed
from src.utils.training_engine import initialize_agent, reset_agent_exploration, train_dqn_iteration

SCRIPT_DIR = str(Path.cwd())  # str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)


def resolve_project_root() -> Path:
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    return start


PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# SETTINGS
# ============================================================================
SEED = 42  # None = new schedule every run
N = 120  # flights sampled from data/training
P_N = 10  # planes sampled
DQN_EPISODES = 500

# True: Gurobi and VNS may only run as long as the DQN took (DQN_TIME decides which DQN time).
# False: Gurobi uses BASELINE_SOLVER_CONFIG (proven optimal) and VNS runs for VNS_TIME_LIMIT_S.
LIMIT_TO_DQN_TIME = True
DQN_TIME = "training"  # "training" (time to train) or "inference" (time to build one plan once trained)
VNS_TIME_LIMIT_S = 60

SWAP_WINDOW_MIN = 240  # VNS local search only swaps flights departing within this many minutes
# ============================================================================


# --- pretty printing (same style as cplex.py / gurobi.py) -------------------------------------------
def _format_table_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        if value.is_integer():
            return f"{value:,.0f}"
        return f"{value:,.2f}"
    return str(value)


def print_pretty_table(title: str, df: pd.DataFrame, columns: list[str] | None = None) -> None:
    frame = df.copy()
    if columns is not None:
        frame = frame.loc[:, [col for col in columns if col in frame.columns]]

    if frame.empty:
        print(f"\n=== {title} ===")
        print("(no rows)")
        return

    frame = frame.reset_index(drop=True)
    frame.insert(0, "#", frame.index + 1)
    formatted = frame.map(_format_table_value)

    widths = {col: max(len(str(col)), formatted[col].map(len).max()) for col in formatted.columns}

    def render_row(values: pd.Series) -> str:
        return "| " + " | ".join(f"{str(values[col]):<{widths[col]}}" for col in formatted.columns) + " |"

    border = "+-" + "-+-".join("-" * widths[col] for col in formatted.columns) + "-+"
    header = "| " + " | ".join(f"{str(col):<{widths[col]}}" for col in formatted.columns) + " |"

    print(f"\n{title}")
    print(border)
    print(header)
    print(border)
    for _, row in formatted.iterrows():
        print(render_row(row))
    print(border)


# --- evaluation: the real environment is the objective ---------------------------------------------
def plan_reward(env: AirlineEnv, plan: list[int]) -> float:
    """Total env reward of flying plane plan[i] on flight i (flights in env.flights order)."""
    env.reset()
    total = 0.0
    for action in plan:
        _, reward, _, _ = env.step(action)
        total += reward
    return total


def plan_breakdown(env: AirlineEnv, plan: list[int]) -> pd.DataFrame:
    env.reset()
    rows = []
    for i, action in enumerate(plan):
        f = env.flights[i]
        _, reward, _, info = env.step(action)
        rows.append(
            {
                "flight_id": f["id"],
                "origin": f["origin"],
                "dest": f["dest"],
                "start": f["start"],
                "plane": env.planes[action],
                "delay_min": info["delay_minutes"],
                "reward": reward,
            }
        )
    return pd.DataFrame(rows)


# --- Double DQN -----------------------------------------------------------------------------------
def train_double_dqn(env: AirlineEnv, n_episodes: int):
    agent = initialize_agent(env, DoubleDQNAgent, MODEL_HYPERPARAMS["DOUBLE_DQN"])
    reset_agent_exploration(agent, n_episodes, MODEL_HYPERPARAMS["DOUBLE_DQN"])
    scores = train_dqn_iteration(
        agent,
        env,
        n_episodes=n_episodes,
        iteration=1,
        model_name="DOUBLE_DQN",
        training_name="compare_methods",
        verbose=True,
    )
    return agent, scores


def dqn_plan(env: AirlineEnv, agent) -> list[int]:
    """Plan built by the trained agent: one greedy (no exploration) action per flight."""
    if hasattr(agent, "policy_net"):
        agent.policy_net.eval()
    env.reset()
    plan, done = [], False
    while not done:
        action_mask = env.get_action_mask() if getattr(agent, "use_action_masking", False) else None
        action = agent.choose_action(env.get_vector_state(), action_mask=action_mask, use_epsilon=False)
        _, _, done, _ = env.step(action)
        plan.append(action)
    return plan


# --- VNS (same as vns.py, but every loop stops at the deadline) ---------------------------------
def greedy_plan(env: AirlineEnv) -> list[int]:
    solver = ClosestPlaneGreedySolver()
    state = env.get_vector_state(env.reset())
    plan, done = [], False
    while not done:
        action = solver.choose_action(state, env)
        raw_state, _, done, _ = env.step(action)
        state = env.get_vector_state(raw_state)
        plan.append(action)
    return plan


def shake_relocate(plan: list[int], n_planes: int, rng: random.Random) -> list[int]:
    """N1: move one random flight to a different random plane."""
    new = plan.copy()
    i = rng.randrange(len(new))
    new[i] = rng.choice([p for p in range(n_planes) if p != new[i]])
    return new


def shake_swap(plan: list[int], n_planes: int, rng: random.Random) -> list[int]:
    """N2: exchange the planes of two random flights flown by different planes."""
    new = plan.copy()
    pairs = [(i, j) for i in range(len(new)) for j in range(i + 1, len(new)) if new[i] != new[j]]
    if pairs:
        i, j = rng.choice(pairs)
        new[i], new[j] = new[j], new[i]
    return new


def shake_tail_swap(plan: list[int], n_planes: int, rng: random.Random) -> list[int]:
    """N3: two random planes exchange everything they fly from a random flight onwards."""
    new = plan.copy()
    a, b = rng.sample(range(n_planes), 2)
    t = rng.randrange(len(new))
    for i in range(t, len(new)):
        if new[i] == a:
            new[i] = b
        elif new[i] == b:
            new[i] = a
    return new


def shake_block_move(plan: list[int], n_planes: int, rng: random.Random) -> list[int]:
    """N4: up to 4 consecutive flights of one plane move to another plane."""
    new = plan.copy()
    a = rng.choice(sorted(set(new)))
    b = rng.choice([p for p in range(n_planes) if p != a])
    flights_of_a = [i for i, p in enumerate(new) if p == a]
    start = rng.randrange(len(flights_of_a))
    for i in flights_of_a[start : start + rng.randint(1, 4)]:
        new[i] = b
    return new


SHAKES = [
    ("N1 relocate", shake_relocate),
    ("N2 swap", shake_swap),
    ("N3 tail swap", shake_tail_swap),
    ("N4 block move", shake_block_move),
]


def local_search(
    env: AirlineEnv, plan: list[int], reward: float, n_planes: int, deadline: float
) -> tuple[list[int], float]:
    """VND over relocate (N1) and time-windowed swap (N2), first improvement; returns early at the deadline."""
    starts = [f["start"] for f in env.flights]
    improved = True
    while improved:
        improved = False
        # N1: relocate one flight
        for i in range(len(plan)):
            for p in range(n_planes):
                if time.perf_counter() > deadline:
                    return plan, reward
                if p == plan[i]:
                    continue
                candidate = plan.copy()
                candidate[i] = p
                r = plan_reward(env, candidate)
                if r > reward + 1e-6:
                    plan, reward, improved = candidate, r, True
                    break
            if improved:
                break
        if improved:
            continue
        # N2: swap two flights departing close in time
        for i in range(len(plan)):
            for j in range(i + 1, len(plan)):
                if time.perf_counter() > deadline:
                    return plan, reward
                if starts[j] - starts[i] > SWAP_WINDOW_MIN:
                    break  # flights are sorted by start time
                if plan[i] == plan[j]:
                    continue
                candidate = plan.copy()
                candidate[i], candidate[j] = candidate[j], candidate[i]
                r = plan_reward(env, candidate)
                if r > reward + 1e-6:
                    plan, reward, improved = candidate, r, True
                    break
            if improved:
                break
    return plan, reward


def vns(env: AirlineEnv, time_limit_s: float, seed: int | None) -> tuple[list[int], float]:
    """Basic VNS from the greedy plan. The greedy plan is always built, even if it alone exceeds the limit."""
    rng = random.Random(seed)
    n_planes = len(env.planes)
    t0 = time.perf_counter()
    deadline = t0 + time_limit_s

    initial = greedy_plan(env)
    best, best_reward = local_search(env, initial, plan_reward(env, initial), n_planes, deadline)
    print(f"  {time.perf_counter() - t0:8.3f}s  {best_reward:>14,.2f}  greedy plan after local search")

    iteration = 0
    while time.perf_counter() < deadline:
        k = 0
        while k < len(SHAKES) and time.perf_counter() < deadline:
            iteration += 1
            name, shake = SHAKES[k]
            candidate = shake(best, n_planes, rng)
            candidate, candidate_reward = local_search(env, candidate, plan_reward(env, candidate), n_planes, deadline)
            if candidate_reward > best_reward + 1e-6:
                best, best_reward = candidate, candidate_reward
                print(
                    f"  {time.perf_counter() - t0:8.3f}s  {best_reward:>14,.2f}  "
                    f"improved via {name} (iteration {iteration})"
                )
                k = 0
            else:
                k += 1
    print(f"  VNS finished: {iteration} shake + local-search iterations")
    return best, best_reward


if __name__ == "__main__":
    set_seed(SEED)
    TRAINING_DATA_DIR = PROJECT_ROOT / "data" / "training"
    flights_df = pd.read_csv(TRAINING_DATA_DIR / "flights.csv")
    itineraries_df = pd.read_csv(TRAINING_DATA_DIR / "itineraries.csv")
    aircraft_df = pd.read_csv(TRAINING_DATA_DIR / "aircraft.csv").drop_duplicates(
        subset="fixed_cost  hourly_cost initial_airport  seats  speed".split(),
    )

    flights = build_flight_pool(
        flights_df.sample(N).sort_values("start_min", ascending=True),
        itineraries_df,
        empty_flights=True,
    )
    fleet = build_planes(aircraft_df.sample(P_N))
    airports = sorted(
        {f["origin"] for f in flights} | {f["dest"] for f in flights} | {p["initial_airport"] for p in fleet.values()}
    )
    dist_matrix = create_dist_dict_from_airports(airports)
    env = AirlineEnv(
        flights,
        fleet,
        dist_matrix,
        airports,
        penalty_per_min=REWARD_CONFIG["penalty_per_min"],
        use_clipping=REWARD_CONFIG["train_use_clipping"],
    )
    print(f"\nSchedule: {len(flights)} flights, {len(fleet)} planes, {len(airports)} airports")

    # --- Double DQN: train, then build its plan ---
    train_start = time.perf_counter()
    agent, training_scores = train_double_dqn(env, DQN_EPISODES)
    train_seconds = time.perf_counter() - train_start
    print(f"\nDouble DQN training time: {train_seconds:.2f}s")

    inference_start = time.perf_counter()
    ddqn = dqn_plan(env, agent)
    inference_seconds = time.perf_counter() - inference_start
    print(f"Double DQN inference time: {inference_seconds:.4f}s")

    if LIMIT_TO_DQN_TIME:
        budget_s = train_seconds if DQN_TIME == "training" else inference_seconds
        print(f"\nGurobi and VNS limited to the DQN {DQN_TIME} time: {budget_s:.4f}s")
    else:
        budget_s = None

    plans = {"Double DQN": ddqn}
    seconds = {"Double DQN": inference_seconds}
    notes = {"Double DQN": f"inference; trained {train_seconds:.2f}s on {DQN_EPISODES} episodes"}

    # --- Gurobi ---
    solve_start = time.perf_counter()
    try:
        gurobi = solve_assignment(
            flights,
            fleet,
            dist_matrix,
            penalty_per_min=REWARD_CONFIG["penalty_per_min"],
            use_clipping=REWARD_CONFIG["train_use_clipping"],
            time_limit_s=budget_s,
            require_optimal=False if LIMIT_TO_DQN_TIME else None,
        )
        plans["Gurobi"] = [env.planes.index(p) for p in gurobi.planes]
        notes["Gurobi"] = "proven optimal" if gurobi.proven_optimal else f"best found, gap to bound {gurobi.gap:.1%}"
    except BaselineSolverError as e:
        print(f"\nGurobi gave no plan: {e}")
        notes["Gurobi"] = "no plan within the limit"
    seconds["Gurobi"] = time.perf_counter() - solve_start
    print(f"\nGurobi time: {seconds['Gurobi']:.4f}s ({notes['Gurobi']})")

    # --- VNS ---
    vns_limit_s = budget_s if LIMIT_TO_DQN_TIME else VNS_TIME_LIMIT_S
    print(f"\nVNS ({vns_limit_s:.4f}s budget), improvements:")
    vns_start = time.perf_counter()
    plans["VNS"], _ = vns(env, vns_limit_s, SEED)
    seconds["VNS"] = time.perf_counter() - vns_start
    notes["VNS"] = "greedy start + search"
    print(f"VNS time: {seconds['VNS']:.4f}s")

    # --- results ---
    rewards = {name: plan_reward(env, plan) for name, plan in plans.items()}
    reference = rewards.get("Gurobi")

    per_flight = None
    for name, plan in plans.items():
        frame = plan_breakdown(env, plan)
        frame = frame.rename(
            columns={"plane": f"plane_{name}", "delay_min": f"delay_{name}", "reward": f"reward_{name}"}
        )
        per_flight = (
            frame if per_flight is None else per_flight.merge(frame, on=["flight_id", "origin", "dest", "start"])
        )
    print_pretty_table("PER-FLIGHT PLANS", per_flight)

    print()
    for name in ["Gurobi", "VNS", "Double DQN"]:
        if name not in rewards:
            print(f"{name} env-style total reward: (no plan)")
        elif name == "Gurobi" or not reference:
            print(f"{name} env-style total reward: {rewards[name]:,.2f}")
        else:
            print(
                f"{name} env-style total reward: {rewards[name]:,.2f} ({(rewards[name] - reference) / abs(reference):+.1%})"
            )

    print_pretty_table(
        "SUMMARY",
        pd.DataFrame(
            [
                {
                    "method": name,
                    "reward": rewards.get(name),
                    "vs_gurobi": (
                        f"{(rewards[name] - reference) / abs(reference):+.1%}"
                        if name in rewards and reference and name != "Gurobi"
                        else ""
                    ),
                    "seconds": round(seconds[name], 4),
                    "note": notes[name],
                }
                for name in ["Gurobi", "VNS", "Double DQN"]
            ]
        ),
    )
