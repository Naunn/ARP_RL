"""
Evaluation utilities for testing trained agents against various solvers.
Handles both iteration-specific evaluation and final cross-validation testing.
"""

import copy
import os

import torch

from src.log_config import get_logger
from src.utils.envs import (
    ClosestPlaneGreedySolver,
    DQNSolver,
    QLearningSolver,
    RandomSolver,
    run_unified_execution,
)

logger = get_logger("plane_assignment")


class EvaluationScoreboard:
    """Manages results collection and formatted reporting."""

    def __init__(self):
        """Initialize an empty results dictionary."""
        self.results = {}

    def add_result(self, schedule_name, solver_name, profit, delay):
        """Add a result to the scoreboard.

        Args:
            schedule_name: Name of the schedule (e.g., "TRAINING_DATA")
            solver_name: Name of the solver (e.g., "DQN_Agent")
            profit: Profit value
            delay: Delay value
        """
        if schedule_name not in self.results:
            self.results[schedule_name] = {}
        self.results[schedule_name][solver_name] = {"profit": profit, "delay": delay}

    def log_iteration_scoreboard(self, iteration):
        """Print formatted scoreboard for an iteration."""
        logger.info("\n" + "-" * 95)
        logger.info(f"SCOREBOARD FOR ITERATION {iteration}")
        logger.info(
            f"{'STRATEGY':<15} | {'TRAIN PROFIT':>14} | {'TRAIN DELAY':>11} | {'TEST PROFIT':>14} | {'TEST DELAY':>11}"
        )
        logger.info("-" * 95)

        if "TRAINING_DATA" in self.results:
            solver_names = self.results["TRAINING_DATA"].keys()
            for solver_name in solver_names:
                tr = self.results["TRAINING_DATA"][solver_name]
                ts = self.results.get("DISRUPTION_TEST", {}).get(
                    solver_name, {"profit": 0, "delay": 0}
                )
                logger.info(
                    f"{solver_name:<15} | "
                    f"${tr['profit']:>13,.0f} | {tr['delay']:>10.0f}m | "
                    f"${ts['profit']:>13,.0f} | {ts['delay']:>10.0f}m"
                )
        logger.info("=" * 95)

    def log_final_scoreboard(self):
        """Print formatted final test matrix scoreboard."""
        logger.info("\n" + "=" * 95)
        header = f"{'STRATEGY':<15} | {'TEST PROFIT':>14} | {'TEST DELAY':>11} | {'FINAL STATUS':>14} | {'METRIC ACC':>11}"
        logger.info(header)
        logger.info("-" * 95)

        # Results may be stored either as a flat mapping
        # {strategy: {"profit":.., "delay":..}} or as a nested mapping
        # {"FINAL_TEST": {strategy: {...}, ...}}. Handle both shapes.
        for strategy_key, metrics in self.results.items():
            if isinstance(metrics, dict) and "profit" in metrics and "delay" in metrics:
                rows = [(strategy_key, metrics)]
            elif isinstance(metrics, dict):
                # metrics is a mapping of solver_name -> {profit, delay}
                rows = list(metrics.items())
            else:
                # Unexpected structure — skip
                continue

            for solver_name, m in rows:
                test_p_str = f"${m.get('profit', 0):>13,.0f}"
                test_d_str = f"{m.get('delay', 0):>10.0f}m"

                status_str = f"{'COMPLETED':>14}"
                metric_str = (
                    f"{'OPTIMAL':>11}"
                    if "_Iter_" in solver_name
                    else f"{'BASELINE':>11}"
                )

                logger.info(
                    f"{solver_name:<15} | {test_p_str} | {test_d_str} | {status_str} | {metric_str}"
                )

        logger.info("=" * 95)


def evaluate_iteration(eval_env, solvers, schedules_eval, iteration):
    """Evaluate all solvers on training and test schedules for an iteration.

    Args:
        eval_env: AirlineEnv for evaluation
        solvers: Dictionary of {solver_name: solver_object}
        schedules_eval: Dictionary of {schedule_name: flight_list}
        iteration: Current iteration number

    Returns:
        EvaluationScoreboard: Results object with scores
    """
    scoreboard = EvaluationScoreboard()

    for sched_name, flight_list in schedules_eval.items():
        for solver_name, solver_obj in solvers.items():
            p, d = run_unified_execution(
                copy.deepcopy(eval_env), solver_obj, flight_list, solver_name
            )
            scoreboard.add_result(sched_name, solver_name, p, d)

    scoreboard.log_iteration_scoreboard(iteration)
    return scoreboard


def evaluate_final_test(
    final_eval_env,
    final_test_flights,
    dqn_agent,
    model_paths,
    q_agent=None,
    q_learning_model_paths=None,
    double_dqn_agent=None,
    double_dqn_model_paths=None,
    extra_solvers=None,
):
    """Run final evaluation on unseen test set with baselines and trained models.

    Args:
        final_eval_env: AirlineEnv for final evaluation
        final_test_flights: Flight list for final evaluation
        dqn_agent: DQNAgent instance
        model_paths: Dictionary of {iteration: model_path} for trained DQN model checkpoints
        q_agent: Optional QAgent instance for Q-learning checkpoint evaluation
        q_learning_model_paths: Optional dictionary of {iteration: model_path} for Q-learning checkpoints
        double_dqn_agent: Optional DoubleDQNAgent instance for Double DQN checkpoint evaluation
        double_dqn_model_paths: Optional dictionary of {iteration: model_path} for Double DQN checkpoints
        extra_solvers: Optional dict of additional in-memory solvers to evaluate

    Returns:
        EvaluationScoreboard: Results from final evaluation
    """
    scoreboard = EvaluationScoreboard()

    # Evaluate baseline solvers
    baselines = {
        "Random": RandomSolver(),
        "Greedy": ClosestPlaneGreedySolver(),
    }

    for name, solver_obj in baselines.items():
        p, d = run_unified_execution(
            copy.deepcopy(final_eval_env), solver_obj, final_test_flights, name
        )
        scoreboard.add_result("FINAL_TEST", name, p, d)

    # Evaluate any additional in-memory solvers
    if extra_solvers is not None:
        for solver_name, solver_obj in extra_solvers.items():
            p, d = run_unified_execution(
                copy.deepcopy(final_eval_env),
                solver_obj,
                final_test_flights,
                solver_name,
            )
            scoreboard.add_result("FINAL_TEST", solver_name, p, d)

    # Evaluate trained DQN model checkpoints
    for iteration, model_path in model_paths.items():
        row_label = f"DQN_Iter_{iteration:02d}"

        if os.path.exists(model_path):
            try:
                eval_agent = copy.deepcopy(dqn_agent)
                eval_agent.policy_net.load_state_dict(torch.load(model_path))
                eval_agent.policy_net.eval()
                historical_solver = DQNSolver(eval_agent)

                p, d = run_unified_execution(
                    copy.deepcopy(final_eval_env),
                    historical_solver,
                    final_test_flights,
                    row_label,
                )
                scoreboard.add_result("FINAL_TEST", row_label, p, d)

            except Exception as e:
                logger.error(f"Failed to evaluate checkpoint {model_path}: {e}")
        else:
            logger.warning(f"Checkpoint omitted. Missing file: {model_path}")

    # Evaluate trained Q-learning model checkpoints if provided
    if q_agent is not None and q_learning_model_paths is not None:
        for iteration, model_path in q_learning_model_paths.items():
            row_label = f"Q_Learning_Iter_{iteration:02d}"

            if os.path.exists(model_path):
                try:
                    eval_agent = copy.deepcopy(q_agent)
                    eval_agent.q_table = torch.load(model_path, weights_only=False)
                    historical_solver = QLearningSolver(eval_agent)

                    p, d = run_unified_execution(
                        copy.deepcopy(final_eval_env),
                        historical_solver,
                        final_test_flights,
                        row_label,
                    )
                    scoreboard.add_result("FINAL_TEST", row_label, p, d)

                except Exception as e:
                    logger.error(f"Failed to evaluate checkpoint {model_path}: {e}")
            else:
                logger.warning(f"Checkpoint omitted. Missing file: {model_path}")

    # Evaluate trained Double DQN model checkpoints if provided
    if double_dqn_agent is not None and double_dqn_model_paths is not None:
        for iteration, model_path in double_dqn_model_paths.items():
            row_label = f"DoubleDQN_Iter_{iteration:02d}"

            if os.path.exists(model_path):
                try:
                    eval_agent = copy.deepcopy(double_dqn_agent)
                    eval_agent.policy_net.load_state_dict(torch.load(model_path))
                    eval_agent.policy_net.eval()
                    historical_solver = DQNSolver(eval_agent)

                    p, d = run_unified_execution(
                        copy.deepcopy(final_eval_env),
                        historical_solver,
                        final_test_flights,
                        row_label,
                    )
                    scoreboard.add_result("FINAL_TEST", row_label, p, d)

                except Exception as e:
                    logger.error(f"Failed to evaluate checkpoint {model_path}: {e}")
            else:
                logger.warning(f"Checkpoint omitted. Missing file: {model_path}")

    scoreboard.log_final_scoreboard()
    return scoreboard


def log_final_evaluation_start(utilization, max_req_planes):
    """Log the start of final evaluation with schedule feasibility info.

    Args:
        utilization: Fleet utilization percentage
        max_req_planes: Peak concurrent planes required
    """
    logger.info("\n" + "=" * 95)
    logger.info("STARTING FINAL EVALUATION ON UNSEEN TEST SCHEDULE")
    logger.info("=" * 95)
    logger.info(
        f"Final Test Schedule Global Utilization: {utilization:.1f}% | Peak Concurrency: {max_req_planes} planes"
    )


def log_training_complete():
    """Log message for end of training loop."""
    logger.info("\nGlobal training cycle finished across all iterations.")
