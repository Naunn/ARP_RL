"""Variable Neighbourhood Search (VNS) for the aircraft assignment problem, evaluated in AirlineEnv.

    python -m src.workspace.vns

A plan is a list `plan[i] = index of the plane flying flights[i]`. Every plan is scored by replaying
it through the real AirlineEnv (total reward), so VNS optimizes exactly what the RL agents and the
Gurobi baseline are measured on.

Basic VNS (Mladenovic & Hansen, 1997):

    x = initial plan (greedy)
    repeat until the time budget runs out:
        k = 1
        while k <= K:
            x'  = shake(x, k)          # random plan in the k-th neighbourhood N_k(x): escape
            x'' = local_search(x')     # VND: descend with cheap moves until no move improves
            if f(x'') > f(x):  x = x''; k = 1     # improvement: go back to the smallest change
            else:              k = k + 1          # no luck: try a more disruptive neighbourhood

Neighbourhoods used for shaking, from small to large changes:
    N1 relocate   : move one flight to another plane
    N2 swap       : exchange the planes of two flights
    N3 tail swap  : two planes exchange their whole remaining day from some flight onwards
                    (the classic "aircraft swap" of airline recovery)
    N4 block move : a run of consecutive flights of one plane moves to another plane

Local search (VND) uses only N1 and N2, with first improvement, restarting from N1 after every
improving move; swaps are only tried between flights departing within SWAP_WINDOW_MIN of each
other (planes rarely profit from swapping flights hours apart, and it keeps the scan fast).
"""

import random
import sys
import time
from pathlib import Path

import pandas as pd

from src.config import REWARD_CONFIG
from src.instances import build_flight_pool, build_planes
from src.utils.dist import create_dist_dict_from_airports
from src.utils.envs import AirlineEnv, ClosestPlaneGreedySolver
from src.utils.seeding import set_seed

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
SEED = None  # None = new schedule every run
N = 60  # flights sampled from data/training
P_N = 10  # planes sampled
VNS_TIME_LIMIT_S = 60  # VNS time budget
SWAP_WINDOW_MIN = 240  # local search only swaps flights departing within this many minutes
RUN_GUROBI = True  # also solve with Gurobi (best plan within GUROBI_TIME_LIMIT_S) for reference
GUROBI_TIME_LIMIT_S = VNS_TIME_LIMIT_S
# ============================================================================


# --- pretty printing (same style as cplex.py / gurobi.py) -------------------------------------------
def _format_table_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
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
                "actual_start": info["actual_start"],
                "delay_min": info["delay_minutes"],
                "reward": reward,
            }
        )
    return pd.DataFrame(rows)


def policy_plan(env: AirlineEnv, solver) -> list[int]:
    """Plan produced by a one-flight-at-a-time solver (here: the greedy start for VNS) in the env."""
    state = env.get_vector_state(env.reset())
    plan, done = [], False
    while not done:
        action = solver.choose_action(state, env)
        raw_state, _, done, _ = env.step(action)
        state = env.get_vector_state(raw_state)
        plan.append(action)
    return plan


# --- neighbourhoods -------------------------------------------------------------------------------
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


def local_search(env: AirlineEnv, plan: list[int], reward: float, n_planes: int) -> tuple[list[int], float]:
    """VND over relocate (N1) and time-windowed swap (N2), first improvement, restart at N1 on success."""
    starts = [f["start"] for f in env.flights]
    improved = True
    while improved:
        improved = False
        # N1: relocate one flight
        for i in range(len(plan)):
            for p in range(n_planes):
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


def vns(env: AirlineEnv, initial: list[int], time_limit_s: float, seed: int | None) -> tuple[list[int], float, list]:
    """Basic VNS. Returns (best plan, its reward, history of (seconds, reward, neighbourhood) improvements)."""
    rng = random.Random(seed)
    n_planes = len(env.planes)
    t0 = time.time()

    best, best_reward = local_search(env, initial, plan_reward(env, initial), n_planes)
    history = [(time.time() - t0, best_reward, "initial + local search")]
    print(f"  {history[0][0]:6.1f}s  {best_reward:>14,.2f}  greedy plan after local search")

    iteration = 0
    while time.time() - t0 < time_limit_s:
        k = 0
        while k < len(SHAKES) and time.time() - t0 < time_limit_s:
            iteration += 1
            name, shake = SHAKES[k]
            candidate = shake(best, n_planes, rng)
            candidate, candidate_reward = local_search(env, candidate, plan_reward(env, candidate), n_planes)
            if candidate_reward > best_reward + 1e-6:
                best, best_reward = candidate, candidate_reward
                history.append((time.time() - t0, best_reward, name))
                print(f"  {history[-1][0]:6.1f}s  {best_reward:>14,.2f}  improved via {name} (iteration {iteration})")
                k = 0
            else:
                k += 1
    print(f"  VNS finished: {iteration} shake + local-search iterations in {time.time() - t0:.1f}s")
    return best, best_reward, history


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

    vns_start = time.perf_counter()
    greedy_plan = policy_plan(env, ClosestPlaneGreedySolver())  # VNS starting point only

    print(f"\nVNS ({VNS_TIME_LIMIT_S}s budget), improvements:")
    vns_plan, vns_reward, history = vns(env, greedy_plan, VNS_TIME_LIMIT_S, SEED)
    vns_seconds = time.perf_counter() - vns_start
    print(f"  VNS time taken (greedy start + search): {vns_seconds:.2f}s")
    results = {"VNS": (vns_reward, vns_seconds)}
    gurobi_plan = None

    if RUN_GUROBI:
        from src.baselines import BaselineSolverError
        from src.baselines.gurobi_solver import solve_assignment

        try:
            gurobi_start = time.perf_counter()
            gurobi = solve_assignment(
                flights,
                fleet,
                dist_matrix,
                REWARD_CONFIG["penalty_per_min"],
                REWARD_CONFIG["train_use_clipping"],
                time_limit_s=GUROBI_TIME_LIMIT_S,
                require_optimal=False,
            )
            status = "proven optimal" if gurobi.proven_optimal else f"best found, gap {gurobi.gap:.0%}"
            gurobi_seconds = time.perf_counter() - gurobi_start
            print(f"\nGurobi time taken: {gurobi_seconds:.2f}s")
            gurobi_plan = [env.planes.index(p) for p in gurobi.planes]
            results[f"Gurobi ({status})"] = (plan_reward(env, gurobi_plan), gurobi_seconds)
        except BaselineSolverError as e:
            print(f"\nGurobi skipped: {e}")

    vns_breakdown = plan_breakdown(env, vns_plan)
    if gurobi_plan is None:
        print_pretty_table("PER-FLIGHT VNS PLAN", vns_breakdown)
    else:
        comparison = vns_breakdown.merge(
            plan_breakdown(env, gurobi_plan)[["flight_id", "plane", "delay_min", "reward"]],
            on="flight_id",
            suffixes=("_vns", "_gurobi"),
        )
        print_pretty_table(
            "PER-FLIGHT VNS VS GUROBI",
            comparison,
            columns=[
                "flight_id",
                "origin",
                "dest",
                "start",
                "plane_vns",
                "plane_gurobi",
                "delay_min_vns",
                "delay_min_gurobi",
                "reward_vns",
                "reward_gurobi",
            ],
        )
    print_pretty_table(
        "TOTAL ENV REWARD",
        pd.DataFrame(
            [{"method": name, "reward": float(r), "seconds": round(sec, 2)} for name, (r, sec) in results.items()]
        ),
    )
    print(
        f"\nVNS: {len(history) - 1} improving shakes; by neighbourhood: "
        f"{pd.Series([h[2] for h in history[1:]]).value_counts().to_dict()}"
    )
