"""Core operational pipeline script executing structural agent evaluation and generational loops."""

import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# --- DATA ENTRY ---
import numpy as np
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
    DisruptionGenerator,
    DQNSolver,
    RandomSolver,
    build_flight_pool,
    build_planes,
    create_dist_dict_from_airports,
    generate_random_flights,
    generate_trap_schedule,
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


def print_results_table(results: dict, name: str):
    """Prints a beautiful, aligned evaluation matrix."""
    # Define exact column widths so they are easy to change
    w1, w2, w3 = 26, 21, 21

    # Create the headers first so we can calculate the exact separator line length
    header_strategy = "STRATEGY"
    header_profit = f"{name} PROFIT"
    header_delay = f"{name} DELAY"

    # Format the header row
    header_row = f"{header_strategy:<{w1}} | {header_profit:>{w2}} | {header_delay:>{w3}}"
    line_length = len(header_row)

    # Print the table
    logger.info("\n" + "=" * line_length)
    logger.info(header_row)
    logger.info("-" * line_length)

    for strat_name, (profit, delay) in results.items():
        # Format the values as strings first so the symbols ($ and m) don't break the alignment
        formatted_profit = f"${profit:,.0f}"
        formatted_delay = f"{delay:,.0f}m"

        logger.info(f"{strat_name:<{w1}} | {formatted_profit:>{w2}} | {formatted_delay:>{w3}}")

    logger.info("=" * line_length + "\n")


TRAINING_DATA_DIR = PROJECT_ROOT / "data" / "training"
flights_df = pd.read_csv(TRAINING_DATA_DIR / "flights.csv")
itineraries_df = pd.read_csv(TRAINING_DATA_DIR / "itineraries.csv")
aircraft_df = pd.read_csv(TRAINING_DATA_DIR / "aircraft.csv").drop_duplicates(
    subset="fixed_cost  hourly_cost initial_airport  seats  speed".split()
)

FLIGHTS = build_flight_pool(flights_df, itineraries_df)
N = 15
C_N = 3
P_N = 2
trap = True
if trap:
    SAMP_FLIGHTS = pd.DataFrame(
        generate_trap_schedule(
            n=N,
            cities=[
                "AMS",
                "LGW",
                "BOD",
            ],  # list(np.random.choice(flights_df.origin.unique(), C_N)),
            start_time_range=(flights_df.start_min.min(), flights_df.arrival_min.max()),
            pass_range=(
                itineraries_df.total_passenger_count.min(),
                itineraries_df.total_passenger_count.max(),
            ),
        )
    )
else:
    SAMP_FLIGHTS = pd.DataFrame(
        generate_random_flights(
            n=N,
            cities=[
                "AMS",
                "LGW",
                "BOD",
            ],  # list(np.random.choice(flights_df.origin.unique(), C_N)),
            start_time_range=(flights_df.start_min.min(), flights_df.arrival_min.max()),
            pass_range=(
                itineraries_df.total_passenger_count.min(),
                itineraries_df.total_passenger_count.max(),
            ),
        )
    )


SAMP_FLIGHTS["total_ticket_price"] = SAMP_FLIGHTS["pass"] * np.average(
    pd.DataFrame(FLIGHTS)["total_ticket_price"] / pd.DataFrame(FLIGHTS)["pass"],
    weights=pd.DataFrame(FLIGHTS)["pass"],
)
FLIGHTS = SAMP_FLIGHTS.to_dict("records")

PLANES = build_planes(aircraft_df.sample(P_N))
PLANES = build_planes(aircraft_df[aircraft_df.aircraft_id.isin(["A319#6", "ERJ145#4"])])

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

dqn_agent_idle_config = copy.deepcopy(MODEL_HYPERPARAMS["DQN"])
dqn_agent_idle_config["use_attention"] = False
dqn_agent_idle_config["use_expert_bias"] = False
dqn_agent_idle_config["use_action_masking"] = False
# dqn_agent_idle_config["init_epsilon"] = 0.8

double_dqn_agent_idle_config = copy.deepcopy(MODEL_HYPERPARAMS["DOUBLE_DQN"])
double_dqn_agent_idle_config["use_attention"] = False
double_dqn_agent_idle_config["use_expert_bias"] = False
double_dqn_agent_idle_config["use_action_masking"] = False
# double_dqn_agent_idle_config["init_epsilon"] = 0.8

dqn_agent_idle = initialize_dqn_agent(dummy_env, DQNAgent, dqn_agent_idle_config)
double_dqn_agent_idle = initialize_dqn_agent(dummy_env, DoubleDQNAgent, double_dqn_agent_idle_config)

dqn_agent_no_bias_config = copy.deepcopy(MODEL_HYPERPARAMS["DQN"])
dqn_agent_no_bias_config["use_expert_bias"] = False
dqn_agent_no_bias_config["use_action_masking"] = False

double_dqn_agent_no_bias_config = copy.deepcopy(MODEL_HYPERPARAMS["DOUBLE_DQN"])
double_dqn_agent_no_bias_config["use_expert_bias"] = False
double_dqn_agent_no_bias_config["use_action_masking"] = False

dqn_agent_no_bias = initialize_dqn_agent(dummy_env, DQNAgent, dqn_agent_no_bias_config)
double_dqn_agent_no_bias = initialize_dqn_agent(dummy_env, DoubleDQNAgent, double_dqn_agent_no_bias_config)

meta_dims = (len(FLIGHTS), len(AIRPORTS), len(PLANES))

# --- THE TRAINING LOOP ---
dqn_agent_scores = []
double_dqn_agent_scores = []
dqn_agent_idle_scores = []
double_dqn_agent_idle_scores = []
dqn_agent_no_bias_scores = []
double_dqn_agent_no_bias_scores = []

for iteration in range(1, N_ITERATIONS + 1):
    log_iteration_start(iteration, N_ITERATIONS)

    dqn_eps: int = MODEL_TRAINING_PARAMS["DQN"]["n_episodes"]
    ddqn_eps: int = MODEL_TRAINING_PARAMS["DOUBLE_DQN"]["n_episodes"]

    reset_agent_exploration(dqn_agent, dqn_eps, MODEL_HYPERPARAMS["DQN"])
    reset_agent_exploration(double_dqn_agent, ddqn_eps, MODEL_HYPERPARAMS["DOUBLE_DQN"])
    reset_agent_exploration(dqn_agent_idle, dqn_eps, dqn_agent_idle_config)
    reset_agent_exploration(double_dqn_agent_idle, ddqn_eps, double_dqn_agent_idle_config)
    reset_agent_exploration(dqn_agent_no_bias, dqn_eps, dqn_agent_no_bias_config)
    reset_agent_exploration(double_dqn_agent_no_bias, ddqn_eps, double_dqn_agent_no_bias_config)

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

    dqn_agent_idle_scores = train_dqn_iteration(
        dqn_agent_idle,
        dqn_env,
        dqn_eps,
        iteration,
        model_name="DQN",
        training_name="idle",
    )
    logger.info(f"--- dqn_agent_idle_scores: {dqn_agent_idle_scores} ---")
    double_dqn_agent_idle_scores = train_dqn_iteration(
        double_dqn_agent_idle,
        ddqn_env,
        ddqn_eps,
        iteration,
        model_name="DOUBLE_DQN",
        training_name="idle",
    )
    logger.info(f"--- double_dqn_agent_no_bias_scores: {double_dqn_agent_no_bias_scores} ---")
    dqn_agent_no_bias_scores = train_dqn_iteration(
        dqn_agent_no_bias,
        dqn_env,
        dqn_eps,
        iteration,
        model_name="DQN",
        training_name="no_bias",
    )
    logger.info(f"--- dqn_agent_no_bias_scores: {dqn_agent_no_bias_scores} ---")
    double_dqn_agent_no_bias_scores = train_dqn_iteration(
        double_dqn_agent_no_bias,
        ddqn_env,
        ddqn_eps,
        iteration,
        model_name="DOUBLE_DQN",
        training_name="no_bias",
    )
    logger.info(f"--- double_dqn_agent_no_bias_scores: {double_dqn_agent_no_bias_scores} ---")
    dqn_agent_scores = train_dqn_iteration(
        dqn_agent,
        dqn_env,
        dqn_eps,
        iteration,
        model_name="DQN",
        training_name="full",
    )
    logger.info(f"--- dqn_agent_scores: {dqn_agent_scores} ---")
    double_dqn_agent_scores = train_dqn_iteration(
        double_dqn_agent,
        ddqn_env,
        ddqn_eps,
        iteration,
        model_name="DOUBLE_DQN",
        training_name="full",
    )
    logger.info(f"--- double_dqn_agent_scores: {double_dqn_agent_scores} ---")

    p1 = get_model_filename(iteration, *meta_dims, dqn_eps, "DQN")
    p2 = get_model_filename(iteration, *meta_dims, ddqn_eps, "DOUBLE_DQN")
    p1_idle = get_model_filename(iteration, *meta_dims, dqn_eps, "DQN_idle")
    p2_idle = get_model_filename(iteration, *meta_dims, ddqn_eps, "DOUBLE_DQN_idle")
    p1_no_bias = get_model_filename(iteration, *meta_dims, dqn_eps, "DQN_no_bias")
    p2_no_bias = get_model_filename(iteration, *meta_dims, ddqn_eps, "DOUBLE_DQN_no_bias")
    torch.save(dqn_agent.policy_net.state_dict(), p1)
    torch.save(double_dqn_agent.policy_net.state_dict(), p2)
    torch.save(dqn_agent_idle.policy_net.state_dict(), p1_idle)
    torch.save(double_dqn_agent_idle.policy_net.state_dict(), p2_idle)
    torch.save(dqn_agent_no_bias.policy_net.state_dict(), p1_no_bias)
    torch.save(double_dqn_agent_no_bias.policy_net.state_dict(), p2_no_bias)

    logger.info(f"--- Iteration {iteration} Mid-Train Summary ---")
    mid_solvers = {
        "Random Baseline": RandomSolver(),
        "Greedy Baseline": ClosestPlaneGreedySolver(),
        "DQN Agent (idle)": DQNSolver(dqn_agent_idle),
        "Double DQN Agent (idle)": DQNSolver(double_dqn_agent_idle),
        "DQN Agent (no_bias)": DQNSolver(dqn_agent_no_bias),
        "Double DQN Agent (no_bias)": DQNSolver(double_dqn_agent_no_bias),
        "DQN Agent": DQNSolver(dqn_agent),
        "Double DQN Agent": DQNSolver(double_dqn_agent),
    }
    mid_results = {name: evaluate_agent_performance(eval_env, solver, name) for name, solver in mid_solvers.items()}
    print_results_table(mid_results, f"EVAL {iteration}")

logger.info("Global training cycle finished across all iterations.")

# --- FINAL TRAIN EVALUATION ---
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
    "DQN Agent (idle)": DQNSolver(dqn_agent_idle),
    "Double DQN Agent (idle)": DQNSolver(double_dqn_agent_idle),
    "DQN Agent (no_bias)": DQNSolver(dqn_agent_no_bias),
    "Double DQN Agent (no_bias)": DQNSolver(double_dqn_agent_no_bias),
    "DQN (full)": DQNSolver(dqn_agent),
    "Double DQN (full)": DQNSolver(double_dqn_agent),
}

final_results = {name: evaluate_agent_performance(final_env, solver, name) for name, solver in final_solvers.items()}
print_results_table(final_results, "TRAINED FINAL")

fig, ax = plt.subplots(figsize=(10, 5))

models = {
    "DQN (full)": (dqn_agent_scores, "#1f77b4"),
    "Double DQN (full)": (double_dqn_agent_scores, "#ff7f0e"),
    "DQN (Idle)": (dqn_agent_idle_scores, "#a6cee3"),
    "Double DQN (Idle)": (double_dqn_agent_idle_scores, "#fdbf6f"),
    "DQN (no_bias)": (dqn_agent_no_bias_scores, "#a6cee3"),
    "Double DQN (no_bias)": (double_dqn_agent_no_bias_scores, "#fdbf6f"),
}

for name, (scores, color) in models.items():
    if scores:
        ax.plot(scores, color=color, alpha=0.15, linewidth=1)
        window = max(1, len(scores) // 20)
        smoothed = pd.Series(scores).rolling(window=window, min_periods=1).mean()
        ax.plot(smoothed, color=color, linewidth=2, label=name)

ax.set_title("Agent Training Performance Comparison", fontweight="bold")
ax.set_xlabel("Episodes")
ax.set_ylabel("Scores")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper left")

plt.tight_layout()

# Save the plot directly under the project root next to your logging targets
save_path = PROJECT_ROOT / "training_performance.png"
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Plot saved successfully at: {save_path}")

# --- DISRUPTED SCHEDULE EVALUATION ---
logger.info("\nGenerating disrupted schedule variant for robustness validation...")
dg = DisruptionGenerator(AIRPORTS, dist_dict)
FLIGHTS_TEST = dg.generate(
    FLIGHTS,  # type: ignore
    actions=[
        {
            "action": "add_delay",
            "target": "random",
            "count": max(1, N // 4),  # Changed from N_FLIGHTS to N
            "min_delay": 60,
            "max_delay": 180,
        },
        {
            "action": "replace_airport",
            "target": list(np.random.choice(AIRPORTS, size=1)),  # Enforce iterable target if expected
            "field": "origin",
            "method": "closest",
        },
    ],
)

disrupted_env = AirlineEnv(
    FLIGHTS_TEST,  # Injecting the disrupted pool
    PLANES,
    dist_dict,
    AIRPORTS,
    penalty,
    use_clipping=REWARD_CONFIG["final_eval_use_clipping"],
)

disrupted_solvers = {
    "Random Baseline": RandomSolver(),
    "Greedy Baseline": ClosestPlaneGreedySolver(),
    "DQN (Idle)": DQNSolver(dqn_agent_idle),
    "Double DQN (Idle)": DQNSolver(double_dqn_agent_idle),
    "DQN (no_bias)": DQNSolver(dqn_agent_no_bias),
    "Double DQN (no_bias)": DQNSolver(double_dqn_agent_no_bias),
    "DQN (full)": DQNSolver(dqn_agent),
    "Double DQN (full)": DQNSolver(double_dqn_agent),
}

logger.info("Executing Robustness Evaluation Matrix...")
disrupted_results = {
    name: evaluate_agent_performance(disrupted_env, solver, name) for name, solver in disrupted_solvers.items()
}
print_results_table(disrupted_results, "DISRUPTED TEST")
