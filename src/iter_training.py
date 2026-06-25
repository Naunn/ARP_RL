"""Core operational pipeline script executing structural agent evaluation and generational loops."""

import copy
import sys
from pathlib import Path

import pandas as pd
import torch

from src.agents.dqn_agent import DoubleDQNAgent, DQNAgent
from src.config import (
    MODEL_HYPERPARAMS,
    MODEL_TRAINING_PARAMS,
    N_ITERATIONS,
    REWARD_CONFIG,
)

# Core imports handled after structural runtime layout positioning checks
from src.utils import (
    AirlineEnv,
    ClosestPlaneGreedySolver,
    DQNSolver,
    RandomSolver,
    build_flight_pool,
    build_planes,
    create_dist_dict_from_airports,
    get_model_filename,
    initialize_dqn_agent,
    log_iteration_start,
    logger,
    reset_agent_exploration,
    run_unified_execution,
    setup_checkpoint_dir,
    train_dqn_iteration,
)


def resolve_project_root() -> Path:
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    return start


PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --- STREAMLINED INLINE EVALUATION UTILITY ---
def evaluate_agent_performance(env, solver, name: str) -> tuple[float, float]:
    """Runs a single evaluation pass of a solver using the system's execution utility."""
    if hasattr(solver, "agent") and hasattr(solver.agent, "policy_net"):
        solver.agent.policy_net.eval()

    profit, delay = run_unified_execution(
        env=copy.deepcopy(env),
        solver=solver,
        flights=env.flights,
        solver_name=name,
    )
    return profit, delay


def print_results_table(results: dict):
    """Prints a beautiful, aligned evaluation matrix."""
    logger.info("\n" + "=" * 70)
    logger.info(f"{'STRATEGY':<20} | {'TEST PROFIT':>16} | {'TEST DELAY':>14}")
    logger.info("-" * 70)
    for name, (profit, delay) in results.items():
        logger.info(f"{name:<20} | ${profit:>14,.0f} | {delay:>12,.0f}m")
    logger.info("=" * 70 + "\n")


# --- DATA ENTRY ---
TRAINING_DATA_DIR = PROJECT_ROOT / "data" / "training"
flights_df = pd.read_csv(TRAINING_DATA_DIR / "flights.csv")
itineraries_df = pd.read_csv(TRAINING_DATA_DIR / "itineraries.csv")
aircraft_df = pd.read_csv(TRAINING_DATA_DIR / "aircraft.csv").drop_duplicates(
    subset="fixed_cost  hourly_cost initial_airport  seats  speed".split()
)

PLANES = build_planes(aircraft_df)
FLIGHTS = build_flight_pool(flights_df, itineraries_df)
AIRPORTS = sorted(
    {f["origin"] for f in FLIGHTS} | {f["dest"] for f in FLIGHTS} | {p["initial_airport"] for p in PLANES.values()}
)
dist_dict = create_dist_dict_from_airports(airports_list=AIRPORTS)
penalty: int = REWARD_CONFIG["penalty_per_min"]

# --- INITIALIZATION ---
dummy_env = AirlineEnv(
    flights=FLIGHTS,
    plane_configs=PLANES,
    dist_dict=dist_dict,
    cities=AIRPORTS,
    penalty_per_min=penalty,
    use_clipping=REWARD_CONFIG["train_use_clipping"],
)

setup_checkpoint_dir()
dqn_agent = initialize_dqn_agent(dummy_env, DQNAgent, MODEL_HYPERPARAMS["DQN"])
double_dqn_agent = initialize_dqn_agent(dummy_env, DoubleDQNAgent, MODEL_HYPERPARAMS["DOUBLE_DQN"])

meta_dims = (len(FLIGHTS), len(AIRPORTS), len(PLANES))

# --- THE TRAINING LOOP ---
for iteration in range(1, N_ITERATIONS + 1):
    log_iteration_start(iteration, N_ITERATIONS)

    dqn_eps: int = MODEL_TRAINING_PARAMS["DQN"]["n_episodes"]
    ddqn_eps: int = MODEL_TRAINING_PARAMS["DOUBLE_DQN"]["n_episodes"]

    reset_agent_exploration(dqn_agent, dqn_eps, MODEL_HYPERPARAMS["DQN"])
    reset_agent_exploration(double_dqn_agent, ddqn_eps, MODEL_HYPERPARAMS["DOUBLE_DQN"])

    dqn_env = AirlineEnv(
        FLIGHTS,
        PLANES,
        dist_dict,
        AIRPORTS,
        penalty,
        use_clipping=REWARD_CONFIG["train_use_clipping"],
    )
    ddqn_env = AirlineEnv(
        FLIGHTS,
        PLANES,
        dist_dict,
        AIRPORTS,
        penalty,
        use_clipping=REWARD_CONFIG["train_use_clipping"],
    )
    eval_env = AirlineEnv(
        FLIGHTS,
        PLANES,
        dist_dict,
        AIRPORTS,
        penalty,
        use_clipping=REWARD_CONFIG["eval_use_clipping"],
    )

    train_dqn_iteration(dqn_agent, dqn_env, dqn_eps, iteration, model_name="DQN")
    train_dqn_iteration(double_dqn_agent, ddqn_env, ddqn_eps, iteration, model_name="DOUBLE_DQN")

    p1 = get_model_filename(iteration, *meta_dims, dqn_eps, "DQN")
    p2 = get_model_filename(iteration, *meta_dims, ddqn_eps, "DOUBLE_DQN")
    torch.save(dqn_agent.policy_net.state_dict(), p1)
    torch.save(double_dqn_agent.policy_net.state_dict(), p2)

    logger.info(f"--- Iteration {iteration} Mid-Train Summary ---")
    mid_solvers = {
        "Random Baseline": RandomSolver(),
        "Greedy Baseline": ClosestPlaneGreedySolver(),
        "DQN Agent": DQNSolver(dqn_agent),
        "Double DQN Agent": DQNSolver(double_dqn_agent),
    }
    mid_results = {name: evaluate_agent_performance(eval_env, solver, name) for name, solver in mid_solvers.items()}
    print_results_table(mid_results)

logger.info("Global training cycle finished across all iterations.")

# --- FINAL CONSOLIDATED CROSS-VALIDATION TEST ---
logger.info("\nExecuting Final Test Evaluation Validation Stage...")
final_env = AirlineEnv(
    FLIGHTS,
    PLANES,
    dist_dict,
    AIRPORTS,
    penalty,
    use_clipping=REWARD_CONFIG["final_eval_use_clipping"],
)

final_solvers = {
    "Random Baseline": RandomSolver(),
    "Greedy Baseline": ClosestPlaneGreedySolver(),
    "Trained DQN": DQNSolver(dqn_agent),
    "Trained Double DQN": DQNSolver(double_dqn_agent),
}

final_results = {name: evaluate_agent_performance(final_env, solver, name) for name, solver in final_solvers.items()}
print_results_table(final_results)
