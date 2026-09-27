"""Shared orchestration helpers for the experiment scripts.

The experiments train and compare the same set of DQN / Double DQN ablation variants ("full"
model, attention-only / no expert bias, and the fully-vanilla "idle" baseline) using the same
train/evaluate/report logic; this module is the single place that builds those variants and runs
them, so each experiment script only differs in the loop it builds around them.
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from src.agents.dqn_agent import DoubleDQNAgent, DQNAgent
from src.config import (
    AGENT_VARIANT_OVERRIDES,
    DISRUPTION_ACTIONS_CONFIG,
    MODEL_HYPERPARAMS,
    MODEL_TRAINING_PARAMS,
    REWARD_CONFIG,
)
from src.utils.envs import (
    AirlineEnv,
    ClosestPlaneGreedySolver,
    DQNSolver,
    RandomSolver,
    run_unified_execution,
)
from src.utils.logging import log_subsection, logger
from src.utils.training_engine import (
    get_model_filename,
    initialize_agent,
    reset_agent_exploration,
    train_dqn_iteration,
)

AGENT_CLASSES: Dict[str, type] = {"DQN": DQNAgent, "DOUBLE_DQN": DoubleDQNAgent}
ALGO_LABELS: Dict[str, str] = {"DQN": "DQN", "DOUBLE_DQN": "Double DQN"}

AgentTable = Dict[str, Dict[str, Any]]  # agents[variant][algo] -> agent instance


def resolve_project_root() -> Path:
    """Walks up from this file to find the repo root (the directory holding pyproject.toml + src/)."""
    start = Path(__file__).resolve().parent
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    return start


def _variant_tag(variant: str) -> str:
    """Checkpoint/training-name suffix for a variant: "full" stays unsuffixed for backward-compat."""
    return "" if variant == "full" else f"_{variant}"


def variant_label(variant: str, algo: str) -> str:
    """Human-readable display name, e.g. 'Double DQN (idle)' / 'DQN (full)'."""
    return f"{ALGO_LABELS[algo]} ({variant})"


def variant_score_key(variant: str, algo: str) -> str:
    """Machine key used in returned score dicts, e.g. 'double_dqn_idle' / 'dqn_full'."""
    return f"{algo.lower()}_{variant}"


def build_disruption_actions(n_flights: int) -> List[Dict[str, Any]]:
    """Builds the disruption-action list (delay a fraction of flights + reroute one origin)."""
    cfg = DISRUPTION_ACTIONS_CONFIG
    return [
        {
            "action": "add_delay",
            "target": "random",
            "count": max(1, int(n_flights * cfg["delay_count_fraction"])),
            "min_delay": cfg["delay_min_minutes"],
            "max_delay": cfg["delay_max_minutes"],
        },
        {
            "action": "replace_airport",
            "target": "random",
            "field": cfg["replace_airport_field"],
            "method": cfg["replace_airport_method"],
        },
    ]


def build_variant_agents(
    dummy_env: AirlineEnv,
    algos: List[str],
    variants: List[str] | None = None,
) -> AgentTable:
    """Instantiates one agent per (variant, algo) pair declared in AGENT_VARIANT_OVERRIDES.

    Returns agents[variant][algo] -> agent, e.g. agents["full"]["DOUBLE_DQN"].
    """
    variants = variants if variants is not None else list(AGENT_VARIANT_OVERRIDES.keys())
    agents: AgentTable = {}
    for variant in variants:
        agents[variant] = {}
        for algo in algos:
            hyperparams = copy.deepcopy(MODEL_HYPERPARAMS[algo])
            hyperparams.update(AGENT_VARIANT_OVERRIDES[variant])
            agents[variant][algo] = initialize_agent(dummy_env, AGENT_CLASSES[algo], hyperparams)
    return agents


def evaluate_agent_performance(env: AirlineEnv, solver: Any, name: str, verbose: bool = False) -> Tuple[float, float]:
    """Runs one evaluation pass of a solver; verbose=True also logs the per-flight schedule."""
    if hasattr(solver, "agent") and hasattr(solver.agent, "policy_net"):
        solver.agent.policy_net.eval()

    return run_unified_execution(
        env=copy.deepcopy(env),
        solver=solver,
        flights=env.flights,
        solver_name=name,
        verbose=verbose,
    )


def print_results_table(results: Dict[str, Tuple[float, float]], name: str) -> None:
    """Logs an aligned profit/delay table per strategy under an "Evaluation: <name>" header."""
    log_subsection(f"Evaluation: {name}")
    rows = [f"  {'Strategy':<22} {'Profit':>14} {'Delay':>12}"]
    rows += [
        f"  {strategy:<22} {f'${profit:,.0f}':>14} {f'{delay:,.0f} min':>12}"
        for strategy, (profit, delay) in results.items()
    ]
    logger.info("\n".join(rows))


def build_solvers(agents: AgentTable) -> Dict[str, Any]:
    """Baselines plus one DQNSolver per (variant, algo) agent currently built."""
    solvers: Dict[str, Any] = {
        "Random Baseline": RandomSolver(),
        "Greedy Baseline": ClosestPlaneGreedySolver(),
    }
    for variant, by_algo in agents.items():
        for algo, agent in by_algo.items():
            solvers[variant_label(variant, algo)] = DQNSolver(agent)
    return solvers


def evaluate_models_on_schedule(
    agents: AgentTable,
    schedule_flights: List[Dict[str, Any]],
    planes: Dict[str, Any],
    dist_dict: Dict[Any, float],
    airports: List[str],
    penalty_per_min: int,
    eval_label: str,
    show_schedule: bool = False,
) -> Dict[str, Tuple[float, float]]:
    """Evaluates every current solver (baselines + all built agent variants) on one schedule."""
    eval_env = AirlineEnv(
        schedule_flights,
        planes,
        dist_dict,
        airports,
        penalty_per_min,
        use_clipping=REWARD_CONFIG["final_eval_use_clipping"],
    )
    results = {
        name: evaluate_agent_performance(eval_env, solver, name, show_schedule)
        for name, solver in build_solvers(agents).items()
    }
    print_results_table(results, eval_label)
    return results


def train_agents_on_schedule(
    agents: AgentTable,
    schedule_flights: List[Dict[str, Any]],
    planes: Dict[str, Any],
    dist_dict: Dict[Any, float],
    airports: List[str],
    penalty_per_min: int,
    meta_dims: Tuple[int, int, int],
    checkpoint_dir: Path,
    n_iterations: int,
    phase_name: str,
    verbose: bool = True,
    n_episodes: int | None = None,
) -> Dict[str, List[float]]:
    """Trains every (variant, algo) agent for n_iterations training-iteration passes on a schedule,
    saving each agent's weights to checkpoint_dir after every iteration. n_episodes overrides the
    per-algo MODEL_TRAINING_PARAMS episode count.

    Matches the historical behavior of this loop: the returned score list per (variant, algo) is
    whichever training iteration ran *last* (each iteration reassigns, not accumulates).
    """
    scores: Dict[str, List[float]] = {
        variant_score_key(variant, algo): [] for variant, by_algo in agents.items() for algo in by_algo
    }

    for iteration in range(1, n_iterations + 1):
        rounds = f", round {iteration}/{n_iterations}" if n_iterations > 1 else ""
        log_subsection(f"Training: {phase_name}{rounds}")

        for variant, by_algo in agents.items():
            for algo, agent in by_algo.items():
                episodes = n_episodes if n_episodes is not None else MODEL_TRAINING_PARAMS[algo]["n_episodes"]
                hyperparams = copy.deepcopy(MODEL_HYPERPARAMS[algo])
                hyperparams.update(AGENT_VARIANT_OVERRIDES[variant])
                reset_agent_exploration(agent, episodes, hyperparams)

                env = AirlineEnv(
                    schedule_flights,
                    planes,
                    dist_dict,
                    airports,
                    penalty_per_min,
                    use_clipping=REWARD_CONFIG["train_use_clipping"],
                )
                scores[variant_score_key(variant, algo)] = train_dqn_iteration(
                    agent,
                    env,
                    episodes,
                    iteration,
                    model_name=algo,
                    training_name=variant_label(variant, algo),
                    verbose=verbose,
                )

                model_tag = f"{algo}_{phase_name}{_variant_tag(variant)}"
                checkpoint_path = get_model_filename(checkpoint_dir, iteration, *meta_dims, episodes, model_tag)
                torch.save(agent.policy_net.state_dict(), checkpoint_path)

    return scores
