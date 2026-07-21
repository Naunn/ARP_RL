"""Core operational pipeline script executing structural agent evaluation and generational loops."""

import copy
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, cast

import matplotlib.pyplot as plt

# --- DATA ENTRY ---
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter1d

from src.agents.dqn_agent import DoubleDQNAgent, DQNAgent
from src.config import (
    MODEL_HYPERPARAMS,
    MODEL_TRAINING_PARAMS,
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

# Fixed examples


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
def evaluate_agent_performance(env, solver, name: str, verbose: bool = True) -> tuple[float, float]:
    """Runs a single evaluation pass of a solver using the system's execution utility."""
    if hasattr(solver, "agent") and hasattr(solver.agent, "policy_net"):
        solver.agent.policy_net.eval()

    profit, delay = run_unified_execution(
        env=copy.deepcopy(env),
        solver=solver,
        flights=env.flights,
        solver_name=name,
        verbose=verbose,
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


def build_disruption_actions() -> List[Dict[str, Any]]:
    return [
        {
            "action": "add_delay",
            "target": "random",
            "count": max(1, N // 3),
            "min_delay": 60,
            "max_delay": 180,
        },
        {
            "action": "replace_airport",
            "target": "random",
            "field": "origin",
            "method": "closest",
        },
    ]


def train_agents_on_schedule(
    schedule_flights: List[Dict[str, Any]],
    n_iterations: int,
    phase_name: str,
    per_iteration_schedule_fn: Callable[[int], List[Dict[str, Any]]] | None = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    dqn_agent_scores: list[float] = []
    double_dqn_agent_scores: list[float] = []
    dqn_agent_idle_scores: list[float] = []
    double_dqn_agent_idle_scores: list[float] = []
    dqn_agent_no_bias_scores: list[float] = []
    double_dqn_agent_no_bias_scores: list[float] = []

    for iteration in range(1, n_iterations + 1):
        log_iteration_start(iteration, n_iterations)
        logger.info(f"[{phase_name}] Schedule training iteration {iteration}/{n_iterations}")

        current_schedule = per_iteration_schedule_fn(iteration) if per_iteration_schedule_fn else schedule_flights

        dqn_eps: int = MODEL_TRAINING_PARAMS["DQN"]["n_episodes"]
        ddqn_eps: int = MODEL_TRAINING_PARAMS["DOUBLE_DQN"]["n_episodes"]

        reset_agent_exploration(dqn_agent, dqn_eps, MODEL_HYPERPARAMS["DQN"])
        reset_agent_exploration(double_dqn_agent, ddqn_eps, MODEL_HYPERPARAMS["DOUBLE_DQN"])
        reset_agent_exploration(dqn_agent_idle, dqn_eps, dqn_agent_idle_config)
        reset_agent_exploration(double_dqn_agent_idle, ddqn_eps, double_dqn_agent_idle_config)
        reset_agent_exploration(dqn_agent_no_bias, dqn_eps, dqn_agent_no_bias_config)
        reset_agent_exploration(double_dqn_agent_no_bias, ddqn_eps, double_dqn_agent_no_bias_config)

        dqn_env = AirlineEnv(
            current_schedule,
            PLANES,
            dist_dict,
            AIRPORTS,
            penalty,
            use_clipping=REWARD_CONFIG["train_use_clipping"],
        )
        ddqn_env = AirlineEnv(
            current_schedule,
            PLANES,
            dist_dict,
            AIRPORTS,
            penalty,
            use_clipping=REWARD_CONFIG["train_use_clipping"],
        )
        # eval_env = AirlineEnv(
        #     current_schedule,
        #     PLANES,
        #     dist_dict,
        #     AIRPORTS,
        #     penalty,
        #     use_clipping=REWARD_CONFIG["eval_use_clipping"],
        # )

        dqn_agent_idle_scores = train_dqn_iteration(
            dqn_agent_idle,
            dqn_env,
            dqn_eps,
            iteration,
            model_name="DQN",
            training_name=f"{phase_name}_idle",
            verbose=verbose,
        )
        # logger.info(
        #     f"--- dqn_agent_idle_scores ({phase_name}): {dqn_agent_idle_scores} ---"
        # )
        double_dqn_agent_idle_scores = train_dqn_iteration(
            double_dqn_agent_idle,
            ddqn_env,
            ddqn_eps,
            iteration,
            model_name="DOUBLE_DQN",
            training_name=f"{phase_name}_idle",
            verbose=verbose,
        )
        # logger.info(
        #     f"--- double_dqn_agent_idle_scores ({phase_name}): {double_dqn_agent_idle_scores} ---"
        # )
        dqn_agent_no_bias_scores = train_dqn_iteration(
            dqn_agent_no_bias,
            dqn_env,
            dqn_eps,
            iteration,
            model_name="DQN",
            training_name=f"{phase_name}_no_bias",
            verbose=verbose,
        )
        # logger.info(
        #     f"--- dqn_agent_no_bias_scores ({phase_name}): {dqn_agent_no_bias_scores} ---"
        # )
        double_dqn_agent_no_bias_scores = train_dqn_iteration(
            double_dqn_agent_no_bias,
            ddqn_env,
            ddqn_eps,
            iteration,
            model_name="DOUBLE_DQN",
            training_name=f"{phase_name}_no_bias",
            verbose=verbose,
        )
        # logger.info(
        #     f"--- double_dqn_agent_no_bias_scores ({phase_name}): {double_dqn_agent_no_bias_scores} ---"
        # )
        dqn_agent_scores = train_dqn_iteration(
            dqn_agent,
            dqn_env,
            dqn_eps,
            iteration,
            model_name="DQN",
            training_name=f"{phase_name}_full",
            verbose=verbose,
        )
        # logger.info(f"--- dqn_agent_scores ({phase_name}): {dqn_agent_scores} ---")
        double_dqn_agent_scores = train_dqn_iteration(
            double_dqn_agent,
            ddqn_env,
            ddqn_eps,
            iteration,
            model_name="DOUBLE_DQN",
            training_name=f"{phase_name}_full",
            verbose=verbose,
        )
        # logger.info(
        #     f"--- double_dqn_agent_scores ({phase_name}): {double_dqn_agent_scores} ---"
        # )

        p1 = get_model_filename(iteration, *meta_dims, dqn_eps, f"DQN_{phase_name}")
        p2 = get_model_filename(iteration, *meta_dims, ddqn_eps, f"DOUBLE_DQN_{phase_name}")
        p1_idle = get_model_filename(iteration, *meta_dims, dqn_eps, f"DQN_{phase_name}_idle")
        p2_idle = get_model_filename(iteration, *meta_dims, ddqn_eps, f"DOUBLE_DQN_{phase_name}_idle")
        p1_no_bias = get_model_filename(iteration, *meta_dims, dqn_eps, f"DQN_{phase_name}_no_bias")
        p2_no_bias = get_model_filename(
            iteration,
            *meta_dims,
            ddqn_eps,
            f"DOUBLE_DQN_{phase_name}_no_bias",
        )
        torch.save(dqn_agent.policy_net.state_dict(), p1)
        torch.save(double_dqn_agent.policy_net.state_dict(), p2)
        torch.save(dqn_agent_idle.policy_net.state_dict(), p1_idle)
        torch.save(double_dqn_agent_idle.policy_net.state_dict(), p2_idle)
        torch.save(dqn_agent_no_bias.policy_net.state_dict(), p1_no_bias)
        torch.save(double_dqn_agent_no_bias.policy_net.state_dict(), p2_no_bias)

        # mid_solvers = {
        #     "Random Baseline": RandomSolver(),
        #     "Greedy Baseline": ClosestPlaneGreedySolver(),
        #     "DQN Agent (idle)": DQNSolver(dqn_agent_idle),
        #     "Double DQN Agent (idle)": DQNSolver(double_dqn_agent_idle),
        #     "DQN Agent (no_bias)": DQNSolver(dqn_agent_no_bias),
        #     "Double DQN Agent (no_bias)": DQNSolver(double_dqn_agent_no_bias),
        #     "DQN Agent": DQNSolver(dqn_agent),
        #     "Double DQN Agent": DQNSolver(double_dqn_agent),
        # }
        # mid_results = {
        #     name: evaluate_agent_performance(eval_env, solver, name)
        #     for name, solver in mid_solvers.items()
        # }
        # print_results_table(mid_results, f"{phase_name.upper()} EVAL {iteration}")

    logger.info(f"[{phase_name}] Training cycle finished across all iterations.")
    return {
        "dqn_full": dqn_agent_scores,
        "double_dqn_full": double_dqn_agent_scores,
        "dqn_idle": dqn_agent_idle_scores,
        "double_dqn_idle": double_dqn_agent_idle_scores,
        "dqn_no_bias": dqn_agent_no_bias_scores,
        "double_dqn_no_bias": double_dqn_agent_no_bias_scores,
    }


def build_current_solvers() -> Dict[str, Any]:
    return {
        "Random Baseline": RandomSolver(),
        "Greedy Baseline": ClosestPlaneGreedySolver(),
        "DQN Agent (idle)": DQNSolver(dqn_agent_idle),
        "Double DQN Agent (idle)": DQNSolver(double_dqn_agent_idle),
        "DQN Agent (no_bias)": DQNSolver(dqn_agent_no_bias),
        "Double DQN Agent (no_bias)": DQNSolver(double_dqn_agent_no_bias),
        "DQN (full)": DQNSolver(dqn_agent),
        "Double DQN (full)": DQNSolver(double_dqn_agent),
    }


def evaluate_models_on_schedule(schedule_flights: List[Dict[str, Any]], eval_label: str, show_schedule: bool = True):
    eval_env = AirlineEnv(
        schedule_flights,
        PLANES,
        dist_dict,
        AIRPORTS,
        penalty,
        use_clipping=REWARD_CONFIG["final_eval_use_clipping"],
    )
    results = {
        name: evaluate_agent_performance(eval_env, solver, name, show_schedule)
        for name, solver in build_current_solvers().items()
    }
    print_results_table(results, eval_label)
    return results


TRAINING_DATA_DIR = PROJECT_ROOT / "data" / "training"
flights_df = pd.read_csv(TRAINING_DATA_DIR / "flights.csv")
itineraries_df = pd.read_csv(TRAINING_DATA_DIR / "itineraries.csv")
aircraft_df = pd.read_csv(TRAINING_DATA_DIR / "aircraft.csv").drop_duplicates(
    subset="fixed_cost  hourly_cost initial_airport  seats  speed".split()
)

N = 15  # 15 for testing random + trap (100 iter); round(flights_df.shape[0]/3) for sampled - too long for 100 iter :(
C_N = 3  # 3 for testing random + trap (100 iter); for sampled does not matter
P_N = 2  # 2 for testing random + trap (100 iter); round(aircraft_df.shape[0]/3) for sampled - too long for 100 iter :(
trap = False

iter_viz = {}
for i in range(100):
    logger.info(
        f"\n====================================================== [ITERATION {i + 1}] ======================================================"
    )
    FLIGHTS = build_flight_pool(flights_df.sample(N).sort_values("start_min", ascending=True), itineraries_df)

    if trap:
        SAMP_FLIGHTS = pd.DataFrame(
            generate_trap_schedule(
                n=N,
                cities=list(np.random.choice(flights_df.origin.unique(), C_N)),
                start_time_range=(
                    flights_df.start_min.min(),
                    flights_df.arrival_min.max(),
                ),
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
                cities=list(np.random.choice(flights_df.origin.unique(), C_N)),
                start_time_range=(
                    flights_df.start_min.min(),
                    flights_df.arrival_min.max(),
                ),
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

    # # FIXED RANDOM SCHEDULE EXAMPLE
    # FLIGHTS = EXAMPLES["random"][0].to_dict("records")
    # PLANES = build_planes(EXAMPLES["random"][1])

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
    FLIGHTS_INITIAL = cast(List[Dict[str, Any]], copy.deepcopy(FLIGHTS))
    dg = DisruptionGenerator(AIRPORTS, dist_dict)

    logger.info("\nPhase 1/3: Training on initial schedule...")
    training_scores = train_agents_on_schedule(
        FLIGHTS_INITIAL,
        1,
        "initial",
    )
    eval_results = evaluate_models_on_schedule(FLIGHTS_INITIAL, "POST INITIAL TRAIN", False)

    iter_viz[f"{i}"] = (training_scores, eval_results)


with open("/home/bartosz/repos/ARP_RL/data/experiments/random_100_itr.pkl", "wb") as f:
    pickle.dump(iter_viz, f)

with open("/home/bartosz/repos/ARP_RL/data/experiments/random_100_itr.pkl", "rb") as f:
    loaded_dict = pickle.load(f)

iter_viz = loaded_dict


# Extract data by model across iterations
models = iter_viz["0"][1].keys()
iteration_indices = list(range(len(iter_viz)))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Define colors for consistency
colors = plt.cm.tab10(range(len(models)))

# Plot profit
for idx, model in enumerate(models):
    profits = [iter_viz[f"{i}"][1][model][0] for i in iteration_indices]

    # Scatter plot with transparency
    ax1.scatter(
        iteration_indices,
        profits,
        alpha=0.4,
        s=100,
        color=colors[idx],
        label=model,
        edgecolors="black",
        linewidth=0.5,
    )

    # Smoothed trend line
    if len(profits) > 1:
        smoothed = uniform_filter1d(profits, size=max(1, len(profits) // 3))
        ax1.plot(iteration_indices, smoothed, color=colors[idx], linewidth=2.5, alpha=0.8)

ax1.set_xlabel("Iteration", fontsize=11, fontweight="bold")
ax1.set_ylabel("Profit ($)", fontsize=11, fontweight="bold")
ax1.set_title("Model Profit Across Iterations", fontsize=12, fontweight="bold")
ax1.legend(loc="best", framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle="--")

# Plot delay
for idx, model in enumerate(models):
    delays = [iter_viz[f"{i}"][1][model][1] for i in iteration_indices]

    # Scatter plot with transparency
    ax2.scatter(
        iteration_indices,
        delays,
        alpha=0.4,
        s=100,
        color=colors[idx],
        label=model,
        edgecolors="black",
        linewidth=0.5,
    )

    # Smoothed trend line
    if len(delays) > 1:
        smoothed = uniform_filter1d(delays, size=max(1, len(delays) // 3))
        ax2.plot(iteration_indices, smoothed, color=colors[idx], linewidth=2.5, alpha=0.8)

ax2.set_xlabel("Iteration", fontsize=11, fontweight="bold")
ax2.set_ylabel("Delay (minutes)", fontsize=11, fontweight="bold")
ax2.set_title("Model Delay Across Iterations", fontsize=12, fontweight="bold")
ax2.legend(loc="best", framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle="--")

plt.tight_layout()
plt.show()

# gathered viz
with open("/home/bartosz/repos/ARP_RL/data/experiments/random_100_itr.pkl", "rb") as f:
    random_100_itr = pickle.load(f)
with open("/home/bartosz/repos/ARP_RL/data/experiments/trap_100_itr.pkl", "rb") as f:
    trap_100_itr = pickle.load(f)
with open("/home/bartosz/repos/ARP_RL/data/experiments/sample_100_itr.pkl", "rb") as f:
    sample_100_itr = pickle.load(f)


def sorted_iteration_keys(exp_dict: dict[str, Any]) -> list[str]:
    return sorted(exp_dict.keys(), key=lambda k: int(k))


# Use shared model ordering and colors so all subplots/legend stay consistent.
models = list(sample_100_itr[sorted_iteration_keys(sample_100_itr)[0]][1].keys())
colors = plt.cm.tab10(range(len(models)))
model_colors = {model_name: colors[idx] for idx, model_name in enumerate(models)}
model_display_names = {
    "Random Baseline": "Random Baseline",
    "Greedy Baseline": "Greedy Baseline",
    "DQN Agent (idle)": "DQN Agent (idle)",
    "Double DQN Agent (idle)": "Double DQN Agent (idle)",
    "DQN Agent (no_bias)": "DQN Agent (attention only)",
    "Double DQN Agent (no_bias)": "Double DQN Agent (attention only)",
    "DQN (full)": "DQN (full)",
    "Double DQN (full)": "Double DQN (full)",
    "dqn_full": "DQN (full)",
    "double_dqn_full": "Double DQN (full)",
    "dqn_idle": "DQN Agent (idle)",
    "double_dqn_idle": "Double DQN Agent (idle)",
    "dqn_no_bias": "DQN Agent (attention only)",
    "double_dqn_no_bias": "Double DQN Agent (attention only)",
}
line_styles = ["-", "--", "-.", ":"]
markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

gathered_sets: list[tuple[str, dict[str, Any]]] = [
    ("Random Schedule", random_100_itr),
    ("Trap Schedule", trap_100_itr),
    ("Sample Schedule", sample_100_itr),
]


def compute_average_rewards_by_schedule(
    datasets: list[tuple[str, dict[str, Any]]], model_names: list[str]
) -> pd.DataFrame:
    summary_rows: list[dict[str, float | str]] = []
    for schedule_name, dataset in datasets:
        iter_keys = sorted_iteration_keys(dataset)
        row: dict[str, float | str] = {"schedule": schedule_name}
        for model_name in model_names:
            rewards = [dataset[k][1][model_name][0] for k in iter_keys]
            row[model_name] = float(np.mean(rewards)) if rewards else float("nan")
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def format_k_value(value: float) -> str:
    return f"{value / 1000:.1f}k"


avg_reward_df = compute_average_rewards_by_schedule(gathered_sets, models)
logger.info("\nAverage reward over iterations (per model, per schedule):")
avg_reward_df_display = avg_reward_df.copy()
for model_name in models:
    avg_reward_df_display[model_name] = avg_reward_df_display[model_name].map(format_k_value)
logger.info("\n" + avg_reward_df_display.to_string(index=False))

max_iterations = max(len(dataset) for _, dataset in gathered_sets)

PLOT_TITLE_FONTSIZE = 17
AXIS_LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 15

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

for ax, (title, dataset) in zip(axes, gathered_sets):
    iter_keys = sorted_iteration_keys(dataset)
    iteration_indices = list(range(len(iter_keys)))

    for idx, model in enumerate(models):
        rewards = [dataset[k][1][model][0] for k in iter_keys]

        ax.scatter(
            iteration_indices,
            rewards,
            alpha=0.35,
            s=85,
            color=colors[idx],
            edgecolors="black",
            linewidth=0.4,
        )

        if len(rewards) > 1:
            smoothed = uniform_filter1d(rewards, size=max(1, len(rewards) // 3))
            ax.plot(
                iteration_indices,
                smoothed,
                color=colors[idx],
                linewidth=2.8,
                alpha=0.9,
                linestyle=line_styles[idx % len(line_styles)],
                marker=markers[idx % len(markers)],
                markersize=4.5,
                markerfacecolor="white",
                markeredgewidth=0.9,
                markevery=max(1, len(rewards) // 12),
                label=model_display_names.get(model, model),
            )

    ax.set_title(
        f"{title}: Reward Across Iterations",
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.set_ylabel("Reward ($)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, max_iterations - 1)

axes[-1].set_xlabel("Iteration", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")

# One shared legend across all three subplots.
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.525, 0.0015),
    ncol=len(labels) // 2,
    framealpha=0.95,
    fontsize=LEGEND_FONTSIZE,
    handlelength=1.4,
    handletextpad=0.35,
    columnspacing=0.8,
    borderpad=0.25,
    labelspacing=0.25,
)

# fig.suptitle("Gathered Reward Signals by Schedule Type", fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.97))
plt.show()

# convergence curve: average normalized trajectory shape over all iterations
train_models = list(sample_100_itr[sorted_iteration_keys(sample_100_itr)[0]][0].keys())
train_model_colors = {
    model_key: model_colors[color_name]
    for model_key, color_name in {
        "dqn_full": "DQN (full)",
        "double_dqn_full": "Double DQN (full)",
        "dqn_idle": "DQN Agent (idle)",
        "double_dqn_idle": "Double DQN Agent (idle)",
        "dqn_no_bias": "DQN Agent (no_bias)",
        "double_dqn_no_bias": "Double DQN Agent (no_bias)",
    }.items()
    if color_name in model_colors
}


def normalize_curve_shape(scores: list[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return np.array([])
    v_min = np.min(values)
    v_max = np.max(values)
    if np.isclose(v_max, v_min):
        return np.zeros_like(values)
    return (values - v_min) / (v_max - v_min)


def compute_average_normalized_shapes(
    datasets: list[tuple[str, dict[str, Any]]], model_names: list[str]
) -> dict[str, dict[str, np.ndarray]]:
    schedule_to_model_shapes: dict[str, dict[str, np.ndarray]] = {}

    for schedule_name, dataset in datasets:
        iter_keys = sorted_iteration_keys(dataset)
        model_shape_lists: dict[str, list[np.ndarray]] = {m: [] for m in model_names}

        for key in iter_keys:
            training_scores = dataset[key][0]
            for model_name in model_names:
                scores = training_scores.get(model_name, [])
                if len(scores) > 0:
                    model_shape_lists[model_name].append(normalize_curve_shape(scores))

        avg_shapes_for_models: dict[str, np.ndarray] = {}
        for model_name, shape_lists in model_shape_lists.items():
            if not shape_lists:
                avg_shapes_for_models[model_name] = np.array([])
                continue

            min_len = min(len(shape) for shape in shape_lists)
            aligned = np.array([shape[:min_len] for shape in shape_lists], dtype=float)
            avg_shapes_for_models[model_name] = np.nanmean(aligned, axis=0)

        schedule_to_model_shapes[schedule_name] = avg_shapes_for_models

    return schedule_to_model_shapes


avg_normalized_shapes = compute_average_normalized_shapes(gathered_sets, train_models)
MAX_SHAPE_EPOCHS = 300

fig2, axes2 = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

for ax, (schedule_name, _) in zip(axes2, gathered_sets):
    for idx, model_name in enumerate(train_models):
        y = avg_normalized_shapes[schedule_name][model_name]
        if y.size == 0:
            continue

        y_plot = y[:MAX_SHAPE_EPOCHS]
        x = np.arange(1, len(y_plot) + 1)
        smooth_window = max(3, len(y_plot) // 20)
        y_smooth = uniform_filter1d(y_plot, size=smooth_window)

        # Draw raw and smoothed curves for a stable shape comparison.
        ax.plot(
            x,
            y_plot,
            color=train_model_colors.get(model_name, colors[idx % len(colors)]),
            linewidth=1.1,
            alpha=0.22,
            linestyle=line_styles[idx % len(line_styles)],
            marker=None,
        )
        ax.plot(
            x,
            y_smooth,
            color=train_model_colors.get(model_name, colors[idx % len(colors)]),
            linewidth=3.2,
            alpha=0.95,
            linestyle=line_styles[idx % len(line_styles)],
            marker=markers[idx % len(markers)],
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=0.9,
            markevery=max(1, len(x) // 14),
            label=model_display_names.get(model_name, model_name),
        )

    ax.set_title(
        f"{schedule_name}: Average Normalized Training Shape",
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.set_ylabel("Normalized reward", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(0.2, 1)
    ax.set_xlim(1, MAX_SHAPE_EPOCHS)

axes2[-1].set_xlabel("Epoch", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")

handles2, labels2 = axes2[0].get_legend_handles_labels()
fig2.legend(
    handles2,
    labels2,
    loc="lower center",
    bbox_to_anchor=(0.525, 0.0015),
    ncol=max(1, len(labels2) // 2),
    framealpha=0.95,
    fontsize=LEGEND_FONTSIZE,
    handlelength=1.4,
    handletextpad=0.35,
    columnspacing=0.8,
    borderpad=0.25,
    labelspacing=0.25,
)

fig2.tight_layout(rect=(0.0, 0.08, 1.0, 0.97))
plt.show()
