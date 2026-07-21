"""Core operational pipeline script executing structural agent evaluation and generational loops."""

import copy
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, cast

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

# Core imports handled after structural runtime layout positioning checks


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
trap = True

iter_viz = {}

for i in range(10):
    logger.info(f"\n[initial_train_cycle] Iteration {i + 1}: generating disruption from original schedule")
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
    init_schedule_eval = evaluate_models_on_schedule(FLIGHTS_INITIAL, "POST INITIAL TRAIN", False)

    # --- DISRUPTED SCHEDULE RETRAINING ---
    logger.info("\nPhase 2/3: Iterative disruption cycle (generate -> evaluate -> retrain)...")

    disrupted_scores: dict[str, list[float]] = {
        "dqn_full": [],
        "double_dqn_full": [],
        "dqn_idle": [],
        "double_dqn_idle": [],
        "dqn_no_bias": [],
        "double_dqn_no_bias": [],
    }
    schedules: List[tuple[str, List[Dict[str, Any]]]] = [("INITIAL SCHEDULE", FLIGHTS_INITIAL)]

    # iter_viz = {}  # {"0": (init_schedule, init_schedule_eval, {"0": (disrupted_schedule, disrupted_pretraining_eval, disrupted_posttraining_eval, initial_schedule_post_distr_training_eval)}, final_disrupted_reeval)}
    iter_viz[f"{i}"] = (FLIGHTS_INITIAL, init_schedule_eval, {})

    for disruption_iteration in range(1, N_ITERATIONS + 1):
        logger.info(
            f"[disruption_cycle] Iteration {disruption_iteration}/{N_ITERATIONS}: generating disruption from original schedule"
        )
        flights_disrupted_iteration = cast(
            List[Dict[str, Any]],
            dg.generate(FLIGHTS_INITIAL, actions=build_disruption_actions()),
        )
        schedules.append((f"DISRUPTED SCHEDULE {disruption_iteration}", flights_disrupted_iteration))

        logger.info(
            f"[PRE-RETRAINING EVAL] Iteration {disruption_iteration}/{N_ITERATIONS}: generating disruption from original schedule"
        )
        disrupted_pretraining_eval = evaluate_models_on_schedule(
            flights_disrupted_iteration,
            f"DISRUPTED PRE-RETRAIN {disruption_iteration}",
            False,
        )

        output = train_agents_on_schedule(
            flights_disrupted_iteration,
            1,
            f"disrupted_retrain_iter{disruption_iteration}",
        )

        logger.info(
            f"[POST-RETRAINING EVAL] Iteration {disruption_iteration}/{N_ITERATIONS}: generating disruption from original schedule"
        )
        disrupted_posttraining_eval = evaluate_models_on_schedule(
            flights_disrupted_iteration,
            f"DISRUPTED POST-RETRAIN {disruption_iteration}",
            False,
        )

        initial_schedule_post_distr_training_eval = evaluate_models_on_schedule(
            FLIGHTS_INITIAL,
            f"INITIAL SCHEDULE POST-RETRAIN {disruption_iteration}",
            False,
        )

        for key, scores in output.items():
            if key in disrupted_scores.keys():
                disrupted_scores[key].extend(scores)

        iter_viz[f"{i}"][2][f"{disruption_iteration}"] = (
            flights_disrupted_iteration,
            disrupted_pretraining_eval,
            disrupted_posttraining_eval,
            initial_schedule_post_distr_training_eval,
        )

    logger.info("\nPhase 3/3: Re-evaluating final models on every disrupted schedule...")
    final_disrupted_reeval: dict[str, dict[str, tuple[float, float]]] = {}
    for disruption_key in sorted(iter_viz[f"{i}"][2].keys(), key=int):
        disrupted_schedule = iter_viz[f"{i}"][2][disruption_key][0]
        final_disrupted_reeval[disruption_key] = evaluate_models_on_schedule(
            disrupted_schedule,
            f"DISRUPTED REEVAL AFTER ALL RETRAINS {disruption_key}",
            False,
        )
    iter_viz[f"{i}"] = (*iter_viz[f"{i}"], final_disrupted_reeval)

# # --- FINAL EVALUATION ON BOTH SCHEDULES ---
# logger.info("\nPhase 3/3: Final evaluation on initial and disrupted schedules...")
# FLIGHTS_FINAL_DISRUPTED = cast(
#     List[Dict[str, Any]],
#     dg.generate(FLIGHTS_INITIAL, actions=build_disruption_actions()),
# )
# schedules.append(("NOT RETRAINED DISRUPTED SCHEDULE", FLIGHTS_FINAL_DISRUPTED))
# for _ in schedules:
#     evaluate_models_on_schedule(_[1], _[0], False)


# # assume all lists have same length
# x = range(len(next(iter(initial_scores.values()))))

# # plot each key as a separate line
# for key, values in initial_scores.items():
#     plt.plot(x, values, marker="o", label=key)

# # labels and legend
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.title("Scores by index")
# plt.legend()
# plt.grid(True)

# plt.show()

# # Create a 1x2 grid of plots sharing the same Y-axis scale
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# # Define models mapping directly to their targeted subplot
# plots_config = [
#     (ax1, "Initial Training Baseline", {
#         "DQN (full)": (initial_scores["dqn_full"], "#2a9d8f"),
#         "Double DQN (full)": (initial_scores["double_dqn_full"], "#e76f51"),
#     }),
#     (ax2, "Post-Disruption Retraining", {
#         "DQN (full)": (disrupted_scores["dqn_full"], "#1f77b4"),
#         "Double DQN (full)": (disrupted_scores["double_dqn_full"], "#ff7f0e"),
#         "DQN (Idle)": (disrupted_scores["dqn_idle"], "#a6cee3"),
#         "Double DQN (Idle)": (disrupted_scores["double_dqn_idle"], "#fdbf6f"),
#         "DQN (no_bias)": (disrupted_scores["dqn_no_bias"], "#8dd3c7"),
#         "Double DQN (no_bias)": (disrupted_scores["double_dqn_no_bias"], "#fb8072"),
#     })
# ]

# for ax, title, models in plots_config:
#     for name, (scores, color) in models.items():
#         if scores:
#             ax.plot(scores, color=color, alpha=0.15, linewidth=1)
#             window = max(1, len(scores) // 20)
#             smoothed = pd.Series(scores).rolling(window=window, min_periods=1).mean()
#             ax.plot(smoothed, color=color, linewidth=2, label=name)

#     ax.set_title(title, fontweight="bold")
#     ax.set_xlabel("Episodes")
#     ax.grid(True, linestyle="--", alpha=0.5)
#     ax.legend(loc="upper left")

# ax1.set_ylabel("Scores")
# fig.suptitle("Agent Training Performance Comparison", fontsize=14, fontweight="bold", y=1.02)

# plt.tight_layout()
# save_path = PROJECT_ROOT / "training_performance_subplots.png"
# plt.savefig(save_path, dpi=300, bbox_inches="tight")
# plt.close()

# iter_viz = {}  # {"0": (init_schedule, init_schedule_eval ,{"0":(distrupted_schedule, disrupted_pretraining_eval, disrupted_posttraining_eval, initial_schedule_post_distr_training_eval)})}


# with open(
#     "/home/bartosz/repos/ARP_RL/data/experiments/trap_disruption_10_itr.pkl", "wb"
# ) as f:
#     pickle.dump(iter_viz, f)

# Model reactions to initial schedule and disrupted schedules across retraining.
models = [
    "Greedy Baseline",
    "DQN Agent (idle)",
    "Double DQN Agent (idle)",
    "DQN Agent (no_bias)",
    "Double DQN Agent (no_bias)",
    "DQN (full)",
    "Double DQN (full)",
]

# fig, axes = plt.subplots(2, 4, figsize=(30, 10.5), sharex="col")

# for model in models:
#     initial_profit = [
#         float(np.mean([iter_viz[run_key][1][model][0] for run_key in run_keys]))
#     ] + [
#         float(np.mean([iter_viz[run_key][2][key][3][model][0] for run_key in run_keys]))
#         for key in disruption_keys
#     ]
#     initial_delay = [
#         float(np.mean([iter_viz[run_key][1][model][1] for run_key in run_keys]))
#     ] + [
#         float(np.mean([iter_viz[run_key][2][key][3][model][1] for run_key in run_keys]))
#         for key in disruption_keys
#     ]

#     pretraining_profit = [
#         float(np.mean([iter_viz[run_key][2][key][1][model][0] for run_key in run_keys]))
#         for key in disruption_keys
#     ]
#     pretraining_delay = [
#         float(np.mean([iter_viz[run_key][2][key][1][model][1] for run_key in run_keys]))
#         for key in disruption_keys
#     ]

#     posttraining_profit = [
#         float(np.mean([iter_viz[run_key][2][key][2][model][0] for run_key in run_keys]))
#         for key in disruption_keys
#     ]
#     posttraining_delay = [
#         float(np.mean([iter_viz[run_key][2][key][2][model][1] for run_key in run_keys]))
#         for key in disruption_keys
#     ]

#     final_reeval_profit = [
#         float(np.mean([iter_viz[run_key][3][key][model][0] for run_key in run_keys]))
#         for key in disruption_keys
#     ]
#     final_reeval_delay = [
#         float(np.mean([iter_viz[run_key][3][key][model][1] for run_key in run_keys]))
#         for key in disruption_keys
#     ]

#     axes[0, 0].plot(
#         x_initial, initial_profit, marker="o", markersize=4, linewidth=1.6, label=model
#     )
#     axes[1, 0].plot(
#         x_initial, initial_delay, marker="o", markersize=4, linewidth=1.6, label=model
#     )

#     axes[0, 1].plot(
#         x_disrupt,
#         pretraining_profit,
#         marker="o",
#         markersize=4,
#         linewidth=1.6,
#         label=model,
#     )
#     axes[1, 1].plot(
#         x_disrupt,
#         pretraining_delay,
#         marker="o",
#         markersize=4,
#         linewidth=1.6,
#         label=model,
#     )

#     axes[0, 2].plot(
#         x_disrupt,
#         posttraining_profit,
#         marker="o",
#         markersize=4,
#         linewidth=1.6,
#         label=model,
#     )
#     axes[1, 2].plot(
#         x_disrupt,
#         posttraining_delay,
#         marker="o",
#         markersize=4,
#         linewidth=1.6,
#         label=model,
#     )

#     axes[0, 3].plot(
#         x_disrupt,
#         final_reeval_profit,
#         marker="o",
#         markersize=4,
#         linewidth=1.6,
#         label=model,
#     )
#     axes[1, 3].plot(
#         x_disrupt,
#         final_reeval_delay,
#         marker="o",
#         markersize=4,
#         linewidth=1.6,
#         label=model,
#     )

# axes[0, 0].set_title(
#     "Initial schedule\nacross retraining iterations",
#     fontweight="bold",
#     fontsize=11,
#     pad=12,
# )
# axes[0, 1].set_title(
#     "Disrupted schedule\nbefore per-iteration retraining",
#     fontweight="bold",
#     fontsize=11,
#     pad=12,
# )
# axes[0, 2].set_title(
#     "Disrupted schedule\nafter per-iteration retraining",
#     fontweight="bold",
#     fontsize=11,
#     pad=12,
# )
# axes[0, 3].set_title(
#     "Disrupted schedule after full retraining\n(final model re-evaluation)",
#     fontweight="bold",
#     fontsize=11,
#     pad=12,
# )

# axes[0, 0].set_ylabel("Profit ($)", fontsize=11)
# axes[1, 0].set_ylabel("Delay (minutes)", fontsize=11)

# for row in range(2):
#     for col in range(4):
#         axes[row, col].grid(True, linestyle="--", alpha=0.4)
#         axes[row, col].tick_params(labelsize=10)

# axes[1, 0].set_xlabel("Retraining iteration index", fontsize=11)
# axes[1, 1].set_xlabel("Disruption scenario index", fontsize=11)
# axes[1, 2].set_xlabel("Disruption scenario index", fontsize=11)
# axes[1, 3].set_xlabel("Disruption scenario index", fontsize=11)

# # Keep a single legend to avoid visual clutter.
# handles, labels = axes[0, 3].get_legend_handles_labels()
# fig.legend(
#     handles,
#     labels,
#     loc="upper center",
#     ncol=4,
#     frameon=True,
#     fontsize=10,
#     bbox_to_anchor=(0.5, 1.02),
# )

# fig.suptitle(
#     "Model performance across disruption retraining stages (averaged across runs)",
#     fontsize=15,
#     fontweight="bold",
#     y=1.04,
# )
# plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

# experiments_dir = PROJECT_ROOT / "data" / "experiments"
# experiments_dir.mkdir(parents=True, exist_ok=True)
# save_path = experiments_dir / "iter_viz_initial_pre_post_final_reeval_lines.png"
# plt.savefig(save_path, dpi=300, bbox_inches="tight")
# plt.savefig(
#     experiments_dir / "iter_viz_initial_pre_post_final_reeval_lines.pdf",
#     dpi=300,
#     bbox_inches="tight",
# )
# plt.show()

# ============================================================================
# COMPUTE AND LOG RECOVERY PROFIT TABLE
# ============================================================================
logger.info("\n" + "=" * 80)
logger.info("RECOVERY PROFIT ANALYSIS TABLE")
logger.info("=" * 80)
logger.info("\nColumn Definitions:")
logger.info("  Initial Schedule    : Reward on initial schedule (start -> after all retraining with % change)")
logger.info("  Pre-Disrupt Avg     : Average profit on disrupted schedule BEFORE retraining")
logger.info("  Post-Retrain Avg    : Average profit on disrupted schedule AFTER retraining")
logger.info("  Final Eval Avg      : Average profit in final re-evaluation on all disrupted schedules")
logger.info("  Avg Recovery %      : Average % of lost profit recovered through retraining")

# Load all disruption histories and combine them into one table.
experiment_files = [
    ("Random", PROJECT_ROOT / "data" / "experiments" / "random_disruption_10_itr.pkl"),
    ("Trap", PROJECT_ROOT / "data" / "experiments" / "trap_disruption_10_itr.pkl"),
    ("Sample", PROJECT_ROOT / "data" / "experiments" / "sample_disruption_10_itr.pkl"),
]

recovery_tables = []
for schedule_type, disruption_data_file in experiment_files:
    logger.info(f"Loading disruption recovery data from {disruption_data_file}")
    with open(disruption_data_file, "rb") as f:
        iter_viz = pickle.load(f)

    run_keys = sorted(iter_viz.keys(), key=int)
    if not run_keys:
        raise ValueError(f"iter_viz is empty for {disruption_data_file}. No runs available for plotting.")

    disruption_keys = sorted(
        set.intersection(*(set(iter_viz[k][2].keys()) for k in run_keys)),
        key=int,
    )
    if not disruption_keys:
        raise ValueError(f"No common disruption keys in {disruption_data_file}; cannot build recovery table.")

    recovery_rows = []
    for model in models:
        initial_profit_start = float(np.mean([iter_viz[run_key][1][model][0] for run_key in run_keys]))
        initial_profit_after_retrain = float(
            np.mean([iter_viz[run_key][2][key][3][model][0] for run_key in run_keys for key in disruption_keys])
        )
        initial_change_pct = (
            (initial_profit_after_retrain - initial_profit_start) / initial_profit_start * 100
            if initial_profit_start != 0
            else 0
        )

        pre_retrain_profits = [
            iter_viz[run_key][2][key][1][model][0] for run_key in run_keys for key in disruption_keys
        ]
        post_retrain_profits = [
            iter_viz[run_key][2][key][2][model][0] for run_key in run_keys for key in disruption_keys
        ]
        final_profits = [iter_viz[run_key][3][key][model][0] for run_key in run_keys for key in disruption_keys]

        recovery_pcts = [
            (
                (post_retrain - pre_retrain) / abs(initial_profit_start - pre_retrain) * 100
                if initial_profit_start != pre_retrain
                else 0
            )
            for pre_retrain, post_retrain in zip(pre_retrain_profits, post_retrain_profits)
        ]

        recovery_rows.append(
            {
                "Schedule Type": schedule_type,
                "Model": model,
                "Initial Schedule": f"${initial_profit_start:.0f} -> ${initial_profit_after_retrain:.0f} ({initial_change_pct:+.1f}%)",
                "Pre-Disrupt Avg": f"${np.mean(pre_retrain_profits):.0f}",
                "Post-Retrain Avg": f"${np.mean(post_retrain_profits):.0f}",
                "Final Eval Avg": f"${np.mean(final_profits):.0f}",
                "Avg Recovery %": f"{np.mean(recovery_pcts):.1f}%",
            }
        )

    recovery_tables.append(pd.DataFrame(recovery_rows))

recovery_df = pd.concat(recovery_tables, ignore_index=True)

logger.info("\n" + recovery_df.to_string(index=False))
logger.info("=" * 80 + "\n")
