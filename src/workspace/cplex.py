"""Standalone CPLEX aircraft assignment pipeline with env-style reward evaluation and Double DQN comparison."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docplex.mp.model import Model

from src.agents.dqn_agent import DoubleDQNAgent
from src.config import MODEL_HYPERPARAMS, MODEL_TRAINING_PARAMS, REWARD_CONFIG
from src.instances import build_flight_pool, build_planes, generate_random_flights
from src.utils.dist import create_dist_dict_from_airports
from src.utils.envs import AirlineEnv
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


TURNAROUND_TIME_MINS = 1
PENALTY_PER_MIN = 150.0
USE_DELAY_CLIPPING = True


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
        columns=["plane", "initial_airport", "seats", "speed", "fixed_cost", "hourly_cost"],
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


def get_flight_duration_mins(dist_km: float, speed_kmh: float) -> float:
    if speed_kmh <= 0:
        return 99999.0
    return (dist_km / speed_kmh) * 60.0


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
        pass_range=(itineraries_df.total_passenger_count.min(), itineraries_df.total_passenger_count.max()),
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


def solve_cplex_schedule(
    flights: list[dict],
    fleet: dict,
    dist_matrix: dict,
    penalty_per_min: float = PENALTY_PER_MIN,
    use_delay_clipping: bool = USE_DELAY_CLIPPING,
):
    flight_costs: dict[tuple[str, int], float | None] = {}
    flight_durations: dict[tuple[str, int], float] = {}
    flight_dict = {f["id"]: f for f in flights}

    for p_name, p_data in fleet.items():
        for f in flights:
            f_id = f["id"]
            dist = dist_matrix.get((f["origin"], f["dest"]), 99999)
            dur_mins = get_flight_duration_mins(dist, p_data["speed"])
            flight_durations[(p_name, f_id)] = dur_mins

            hours = dur_mins / 60.0
            if p_data["seats"] >= f["pass"]:
                flight_costs[(p_name, f_id)] = p_data["fixed_cost"] + (p_data["hourly_cost"] * hours)
            else:
                flight_costs[(p_name, f_id)] = None

    m = Model(name="Aircraft_Assignment_and_Relocation")

    x: dict[tuple[str, int], object] = {}
    for p_name in fleet.keys():
        for f in flights:
            f_id = f["id"]
            x[(p_name, f_id)] = m.binary_var(name=f"x_{p_name}_{f_id}")

    transitions: list[tuple[str, int, int]] = []
    transition_delay_penalties: dict[tuple[str, int, int], float] = {}
    for p_name, p_data in fleet.items():
        for f1 in flights:
            for f2 in flights:
                if f1["id"] != f2["id"] and f2["start"] >= f1["start"]:
                    dist = dist_matrix.get((f1["dest"], f2["origin"]), 99999)
                    travel_mins = get_flight_duration_mins(dist, p_data["speed"])
                    f1_arr_time = f1["start"] + flight_durations[(p_name, f1["id"])]
                    earliest_start = f1_arr_time + TURNAROUND_TIME_MINS + travel_mins
                    actual_start = max(f2["start"], earliest_start)
                    delay_minutes = max(0.0, actual_start - f2["start"])
                    delay_multiplier = 1.0 + min(delay_minutes / 60.0, 2.0)
                    delay_penalty = delay_minutes * f2["pass"] * penalty_per_min * delay_multiplier
                    if use_delay_clipping:
                        delay_penalty = min(delay_penalty, 20000.0)

                    transition_delay_penalties[(p_name, f1["id"], f2["id"])] = delay_penalty
                    transitions.append((p_name, f1["id"], f2["id"]))

    y = m.binary_var_dict(transitions, name="y")

    initial_transitions: list[tuple[str, int]] = []
    initial_delay_penalties: dict[tuple[str, int], float] = {}
    for p_name, p_data in fleet.items():
        init_port = p_data["initial_airport"]
        for f in flights:
            dist = dist_matrix.get((init_port, f["origin"]), 99999)
            travel_mins = get_flight_duration_mins(dist, p_data["speed"])
            earliest_start = travel_mins
            actual_start = max(f["start"], earliest_start)
            delay_minutes = max(0.0, actual_start - f["start"])
            delay_multiplier = 1.0 + min(delay_minutes / 60.0, 2.0)
            delay_penalty = delay_minutes * f["pass"] * penalty_per_min * delay_multiplier
            if use_delay_clipping:
                delay_penalty = min(delay_penalty, 20000.0)

            initial_delay_penalties[(p_name, f["id"])] = delay_penalty
            initial_transitions.append((p_name, f["id"]))

    z = m.binary_var_dict(initial_transitions, name="z")

    for f in flights:
        f_id = f["id"]
        m.add_constraint(
            m.sum(x[(p_name, f_id)] for p_name in fleet.keys() if (p_name, f_id) in x) == 1,
            f"Cover_Flight_{f_id}",
        )

    for p_name in fleet.keys():
        for f2 in flights:
            target_f2_id = f2["id"]
            if (p_name, target_f2_id) in x:
                incoming_initial = z[(p_name, target_f2_id)] if (p_name, target_f2_id) in z else 0
                incoming_transitions = m.sum(
                    y[(p, f1_id, f2_id)] for (p, f1_id, f2_id) in transitions if p == p_name and f2_id == target_f2_id
                )
                m.add_constraint(
                    incoming_initial + incoming_transitions == x[(p_name, target_f2_id)],
                    f"Flow_In_{p_name}_{target_f2_id}",
                )

    for p_name in fleet.keys():
        for f1 in flights:
            f1_id = f1["id"]
            if (p_name, f1_id) in x:
                outgoing_transitions = m.sum(
                    y[(p, src_f1_id, f2_id)]
                    for (p, src_f1_id, f2_id) in transitions
                    if p == p_name and src_f1_id == f1_id
                )
                m.add_constraint(outgoing_transitions <= x[(p_name, f1_id)], f"Flow_Out_{p_name}_{f1_id}")

    assignment_expr = 0.0
    relocation_penalty_expr = 0.0
    delay_penalty_expr = 0.0
    capacity_penalty_expr = 0.0

    for p_name, f_id in x.keys():
        flight = flight_dict[f_id]
        assignment_cost = flight_costs[(p_name, f_id)]
        if assignment_cost is None:
            assignment_cost = 1e8
        capacity_slack = max(0.0, float(fleet[p_name]["seats"]) - float(flight["pass"])) / max(
            1.0, float(flight["pass"])
        )
        capacity_penalty = 500.0 * capacity_slack
        revenue = float(flight.get("total_ticket_price", 0.0))
        assignment_expr += (assignment_cost + capacity_penalty - revenue) * x[(p_name, f_id)]

    for p_name, f1_id, f2_id in transitions:
        f1_dest = flight_dict[f1_id]["dest"]
        f2_orig = flight_dict[f2_id]["origin"]
        dist = dist_matrix.get((f1_dest, f2_orig), 0)
        hours = dist / fleet[p_name]["speed"]
        relocation_penalty = 0.05 * dist
        relocation_penalty_expr += (hours * fleet[p_name]["hourly_cost"] * 0.7 + relocation_penalty) * y[
            (p_name, f1_id, f2_id)
        ]
        delay_penalty_expr += transition_delay_penalties[(p_name, f1_id, f2_id)] * y[(p_name, f1_id, f2_id)]

    for p_name, f_id in initial_transitions:
        delay_penalty_expr += initial_delay_penalties[(p_name, f_id)] * z[(p_name, f_id)]

    m.minimize(assignment_expr + relocation_penalty_expr + delay_penalty_expr + capacity_penalty_expr)

    solution = m.solve()
    if not solution:
        return None, None, None

    assigned_plane_by_flight_id = {}
    for f in flights:
        f_id = f["id"]
        for p_name in fleet.keys():
            if (p_name, f_id) in x and solution.get_value(x[(p_name, f_id)]) > 0.5:
                assigned_plane_by_flight_id[f_id] = p_name
                break

    assigned_records = []
    for f in flights:
        f_id = f["id"]
        assigned_plane = assigned_plane_by_flight_id.get(f_id, "Unassigned")
        is_initial_start = False
        if (
            assigned_plane != "Unassigned"
            and (assigned_plane, f_id) in z
            and solution.get_value(z[(assigned_plane, f_id)]) > 0.5
        ):
            is_initial_start = True
        record = f.copy()
        record["assigned_plane"] = assigned_plane
        record["is_initial_hub_start"] = is_initial_start
        assigned_records.append(record)

    return assigned_records, float(solution.objective_value), assigned_plane_by_flight_id


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

    assigned_records, objective_value, assigned_plane_by_flight_id = solve_cplex_schedule(
        flights,
        fleet,
        dist_matrix,
        penalty_per_min=REWARD_CONFIG["penalty_per_min"],
        use_delay_clipping=REWARD_CONFIG["train_use_clipping"],
    )

    if assigned_records is None or assigned_plane_by_flight_id is None:
        print("No feasible schedule found matching the time/capacity constraints.")
        sys.exit(0)

    schedule_df = pd.DataFrame(assigned_records)
    print_pretty_table(
        "FINAL ASSIGNED SCHEDULE",
        schedule_df,
        columns=["id", "origin", "dest", "start", "pass", "assigned_plane", "is_initial_hub_start"],
    )
    print(f"\nOptimal CPLEX objective: ${objective_value:,.2f}\n")

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
    ].rename(columns={"plane": "cplex_plane", "reward": "cplex_reward", "delay_minutes": "cplex_delay_minutes"})
    comparison_df = comparison_df.merge(
        ddqn_breakdown[["flight_id", "plane", "reward", "delay_minutes", "actual_start"]].rename(
            columns={"plane": "ddqn_plane", "reward": "ddqn_reward", "delay_minutes": "ddqn_delay_minutes"}
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
