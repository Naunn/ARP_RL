"""Per-flight comparison of the optimal CPLEX baseline (src/baselines/cplex_solver.py) with a Double DQN
trained on the same schedule."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.dqn_agent import DoubleDQNAgent
from src.baselines.cplex_solver import solve_assignment
from src.config import MODEL_HYPERPARAMS, MODEL_TRAINING_PARAMS, REWARD_CONFIG
from src.instances import build_flight_pool, build_planes, generate_random_flights
from src.utils.dist import create_dist_dict_from_airports
from src.utils.envs import AirlineEnv
from src.utils.training_engine import (
    initialize_agent,
    reset_agent_exploration,
    train_dqn_iteration,
)

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


def print_fleet_table(fleet: dict) -> None:
    fleet_rows = [
        {
            "plane": plane_id,
            "initial_airport": data["initial_airport"],
            "seats": data["seats"],
            "speed": data["speed"],
            "fixed_cost": data["fixed_cost"],
            "hourly_cost": data["hourly_cost"],
        }
        for plane_id, data in fleet.items()
    ]
    fleet_df = pd.DataFrame(fleet_rows).sort_values(["initial_airport", "plane"]).reset_index(drop=True)
    print_pretty_table(
        "FLEET / PLANES",
        fleet_df,
        columns=[
            "plane",
            "initial_airport",
            "seats",
            "speed",
            "fixed_cost",
            "hourly_cost",
        ],
    )


def print_schedule_table(flights: list[dict], title: str = "PRESCHEDULE") -> None:
    schedule_df = pd.DataFrame(flights)
    if schedule_df.empty:
        print_pretty_table(title, schedule_df)
        return

    columns = ["id", "origin", "dest", "start", "pass", "total_ticket_price"]
    available_columns = [col for col in columns if col in schedule_df.columns]
    schedule_df = schedule_df.loc[:, available_columns].sort_values(["start", "id"]).reset_index(drop=True)
    print_pretty_table(title, schedule_df, columns=available_columns)


def replay_schedule_with_policy(
    schedule: list[dict],
    fleet: dict,
    dist_dict: dict,
    action_selector,
    penalty_per_min: float,
    use_clipping: bool,
):
    airports = sorted(
        {f["origin"] for f in schedule} | {f["dest"] for f in schedule} | {p["initial_airport"] for p in fleet.values()}
    )
    env = AirlineEnv(
        flights=schedule,
        plane_configs=fleet,
        dist_dict=dist_dict,
        cities=airports,
        penalty_per_min=penalty_per_min,
        use_clipping=use_clipping,
    )

    total_reward = 0.0
    rows = []
    env.reset()

    for flight in schedule:
        action_idx = action_selector(env, flight)
        plane_name = env.planes[action_idx]
        _, reward, done, info = env.step(action_idx)
        total_reward += reward

        rows.append(
            {
                "flight_id": flight["id"],
                "origin": flight["origin"],
                "dest": flight["dest"],
                "start": flight["start"],
                "passengers": flight["pass"],
                "plane": plane_name,
                "reward": float(reward),
                "cumulative_reward": float(total_reward),
                "delay_minutes": float(info["delay_minutes"]),
                "actual_start": float(info["actual_start"]),
                "arrival_at_dest": float(info["arrival_at_dest"]),
                "assignment_cost": float(info["assignment_cost"]),
                "revenue": float(info["revenue"]),
                "relocation_penalty": float(info["relocation_penalty"]),
                "delay_penalty": float(info["delay_penalty"]),
                "capacity_slack_penalty": float(info["capacity_slack_penalty"]),
            }
        )

        if done:
            break

    return float(total_reward), pd.DataFrame(rows)


def build_schedule_pool(
    flights_df: pd.DataFrame, itineraries_df: pd.DataFrame, n: int, cities: list[str]
) -> list[dict]:
    sampled_flights = flights_df.sample(n).sort_values("start_min", ascending=True)
    flight_pool = build_flight_pool(sampled_flights, itineraries_df)

    if not flight_pool:
        return []

    synthetic_flights = generate_random_flights(
        n=n,
        cities=cities,
        start_time_range=(flights_df.start_min.min(), flights_df.arrival_min.max()),
        pass_range=(
            itineraries_df.total_passenger_count.min(),
            itineraries_df.total_passenger_count.max(),
        ),
    )
    synthetic_flights = [
        {
            "id": int(f["id"]),
            "origin": str(f["origin"]).strip().upper(),
            "dest": str(f["dest"]).strip().upper(),
            "start": int(f["start"]),
            "pass": int(f["pass"]),
            "total_ticket_price": float(
                np.average(
                    pd.DataFrame(flight_pool)["total_ticket_price"] / pd.DataFrame(flight_pool)["pass"],
                    weights=pd.DataFrame(flight_pool)["pass"],
                )
                * int(f["pass"])
            ),
        }
        for f in synthetic_flights
    ]
    return synthetic_flights


def evaluate_schedule_with_env(
    schedule: list[dict],
    fleet: dict,
    dist_dict: dict,
    assigned_plane_by_flight_id: dict[int, str],
    penalty_per_min: float,
    use_clipping: bool,
) -> tuple[float, dict]:
    total_reward, breakdown_df = replay_schedule_with_policy(
        schedule=schedule,
        fleet=fleet,
        dist_dict=dist_dict,
        action_selector=lambda env, flight: env.planes.index(
            assigned_plane_by_flight_id.get(flight["id"], env.planes[0])
        ),
        penalty_per_min=penalty_per_min,
        use_clipping=use_clipping,
    )
    return float(total_reward), {
        "step_info": breakdown_df.to_dict("records"),
        "breakdown": breakdown_df,
        "env_reward": total_reward,
    }


def evaluate_double_dqn_with_env(
    agent,
    schedule: list[dict],
    fleet: dict,
    dist_dict: dict,
    penalty_per_min: float,
    use_clipping: bool,
):
    if hasattr(agent, "policy_net"):
        agent.policy_net.eval()

    def choose_action(env, _flight):
        state_tuple = env.get_vector_state()
        action_mask = env.get_action_mask() if getattr(agent, "use_action_masking", False) else None
        return agent.choose_action(state_tuple, action_mask=action_mask, use_epsilon=False)

    return replay_schedule_with_policy(
        schedule=schedule,
        fleet=fleet,
        dist_dict=dist_dict,
        action_selector=choose_action,
        penalty_per_min=penalty_per_min,
        use_clipping=use_clipping,
    )


def train_double_dqn_on_schedule(
    schedule: list[dict],
    fleet: dict,
    dist_dict: dict,
    penalty_per_min: float,
    use_clipping: bool,
    n_episodes: int | None = None,
):
    airports = sorted(
        {f["origin"] for f in schedule} | {f["dest"] for f in schedule} | {p["initial_airport"] for p in fleet.values()}
    )
    env = AirlineEnv(
        flights=schedule,
        plane_configs=fleet,
        dist_dict=dist_dict,
        cities=airports,
        penalty_per_min=penalty_per_min,
        use_clipping=use_clipping,
    )
    agent = initialize_agent(env, DoubleDQNAgent, MODEL_HYPERPARAMS["DOUBLE_DQN"])
    episodes = n_episodes or MODEL_TRAINING_PARAMS["DOUBLE_DQN"]["n_episodes"]
    reset_agent_exploration(agent, episodes, MODEL_HYPERPARAMS["DOUBLE_DQN"])
    scores = train_dqn_iteration(
        agent,
        env,
        n_episodes=episodes,
        iteration=1,
        model_name="DOUBLE_DQN",
        training_name="cplex_compare",
        verbose=True,
    )
    return agent, scores


if __name__ == "__main__":
    TRAINING_DATA_DIR = PROJECT_ROOT / "data" / "training"
    flights_df = pd.read_csv(TRAINING_DATA_DIR / "flights.csv")
    itineraries_df = pd.read_csv(TRAINING_DATA_DIR / "itineraries.csv")
    aircraft_df = pd.read_csv(TRAINING_DATA_DIR / "aircraft.csv").drop_duplicates(
        subset="fixed_cost  hourly_cost initial_airport  seats  speed".split(),
    )

    N = 25
    P_N = 4

    C_N = 4
    # flights = build_schedule_pool(
    #     flights_df,
    #     itineraries_df,
    #     n=N,
    #     cities=list(np.random.choice(flights_df.origin.unique(), C_N)),
    # )
    flights = build_flight_pool(flights_df.sample(N).sort_values("start_min", ascending=True), itineraries_df)

    fleet = build_planes(aircraft_df.sample(P_N))
    airports = sorted(
        {f["origin"] for f in flights} | {f["dest"] for f in flights} | {p["initial_airport"] for p in fleet.values()}
    )
    dist_matrix = create_dist_dict_from_airports(airports)

    print_schedule_table(flights, title="PRESCHEDULE")
    print_fleet_table(fleet)

    plan = solve_assignment(
        flights,
        fleet,
        dist_matrix,
        penalty_per_min=REWARD_CONFIG["penalty_per_min"],
        use_clipping=REWARD_CONFIG["train_use_clipping"],
    )
    objective_value = plan.objective
    assigned_plane_by_flight_id = {f["id"]: plane for f, plane in zip(flights, plan.planes)}
    assigned_records = [
        {**f, "assigned_plane": plane, "is_initial_hub_start": from_home}
        for f, plane, from_home in zip(flights, plan.planes, plan.starts_from_home)
    ]

    schedule_df = pd.DataFrame(assigned_records)
    print_pretty_table(
        "FINAL ASSIGNED SCHEDULE",
        schedule_df,
        columns=[
            "id",
            "origin",
            "dest",
            "start",
            "pass",
            "assigned_plane",
            "is_initial_hub_start",
        ],
    )
    print(
        f"\nOptimal CPLEX objective (= -env reward): {objective_value:,.2f}  -> optimal env reward {-objective_value:,.2f}\n"
    )

    env_reward, _ = evaluate_schedule_with_env(
        assigned_records,
        fleet,
        dist_matrix,
        assigned_plane_by_flight_id,
        penalty_per_min=REWARD_CONFIG["penalty_per_min"],
        use_clipping=REWARD_CONFIG["train_use_clipping"],
    )
    print(f"\nEnvironment-style reward for the CPLEX schedule: {env_reward:,.2f}")

    trained_agent, training_scores = train_double_dqn_on_schedule(
        assigned_records,
        fleet,
        dist_matrix,
        penalty_per_min=REWARD_CONFIG["penalty_per_min"],
        use_clipping=REWARD_CONFIG["train_use_clipping"],
        n_episodes=500,
    )
    final_score = float(np.mean(training_scores[-10:])) if training_scores else 0.0
    print(f"Double DQN training score (mean of last 10 eps): {final_score:,.2f}")
    ddqn_reward, ddqn_breakdown = evaluate_double_dqn_with_env(
        trained_agent,
        assigned_records,
        fleet,
        dist_matrix,
        penalty_per_min=REWARD_CONFIG["penalty_per_min"],
        use_clipping=REWARD_CONFIG["train_use_clipping"],
    )

    cplex_reward, cplex_breakdown = evaluate_schedule_with_env(
        assigned_records,
        fleet,
        dist_matrix,
        assigned_plane_by_flight_id,
        penalty_per_min=REWARD_CONFIG["penalty_per_min"],
        use_clipping=REWARD_CONFIG["train_use_clipping"],
    )

    comparison_df = cplex_breakdown["breakdown"][
        ["flight_id", "plane", "reward", "delay_minutes", "actual_start"]
    ].rename(
        columns={
            "plane": "cplex_plane",
            "reward": "cplex_reward",
            "delay_minutes": "cplex_delay_minutes",
        }
    )
    comparison_df = comparison_df.merge(
        ddqn_breakdown[["flight_id", "plane", "reward", "delay_minutes", "actual_start"]].rename(
            columns={
                "plane": "ddqn_plane",
                "reward": "ddqn_reward",
                "delay_minutes": "ddqn_delay_minutes",
            }
        ),
        on="flight_id",
        how="left",
        suffixes=("_cplex", "_ddqn"),
    )
    comparison_df["reward_delta"] = comparison_df["cplex_reward"] - comparison_df["ddqn_reward"]

    print_pretty_table(
        "PER-FLIGHT CPLEX VS DOUBLE DQN COMPARISON",
        comparison_df,
        columns=[
            "flight_id",
            "cplex_plane",
            "ddqn_plane",
            "cplex_reward",
            "ddqn_reward",
            "reward_delta",
            "cplex_delay_minutes",
            "ddqn_delay_minutes",
            "actual_start_cplex",
            "actual_start_ddqn",
        ],
    )
    print(f"\nCPLEX env-style total reward: {cplex_reward:,.2f}")
    print(f"Double DQN env-style total reward: {ddqn_reward:,.2f}")
