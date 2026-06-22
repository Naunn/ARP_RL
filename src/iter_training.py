import copy
import sys
from pathlib import Path

import pandas as pd
import torch

from src.agents.dqn_agent import DoubleDQNAgent
from src.config import (
    DOUBLE_DQN_HYPERPARAMS,
    DOUBLE_DQN_LOG_INTERVAL,
    DQN_LOG_INTERVAL,
    EVAL_USE_CLIPPING,
    FINAL_EVAL_USE_CLIPPING,
    N_DOUBLE_DQN_EPISODES,
    N_DQN_EPISODES,
    N_FLIGHTS,
    N_ITERATIONS,
    PENALTY_PER_MIN,
    TRAIN_USE_CLIPPING,
)
from src.log_config import get_logger
from src.training import (
    TrainingLogger,
    evaluate_final_test,
    evaluate_iteration,
    get_model_filename,
    initialize_dqn_agent,
    log_final_evaluation_start,
    log_training_complete,
    reset_agent_exploration,
    setup_checkpoint_dir,
    train_dqn_iteration,
)
from src.utils.dist import create_dist_dict_from_airports
from src.utils.envs import (
    AirlineEnv,
    ClosestPlaneGreedySolver,
    DQNSolver,
    RandomSolver,
)
from src.utils.schedule import check_global_feasibility


def resolve_project_root() -> Path:
    """Resolve repository root for both script runs and interactive sessions."""
    if "__file__" in globals():
        start = Path(__file__).resolve().parent
    else:
        start = Path.cwd()

    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate

    return start


PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


logger = get_logger("plane_assignment")


def scale_episode_budget(
    base_episodes: int, baseline_flights: int, actual_flights: int, minimum: int
) -> int:
    """Keep total training steps in the same rough budget as the small-schedule setup."""
    if actual_flights <= 0:
        return minimum

    scaled = round(base_episodes * baseline_flights / actual_flights)
    return max(minimum, scaled)


def load_training_tables(data_dir: Path):
    flights_df = pd.read_csv(data_dir / "flights.csv")
    itineraries_df = pd.read_csv(data_dir / "itineraries.csv")
    aircraft_df = pd.read_csv(data_dir / "aircraft.csv")
    return flights_df, itineraries_df, aircraft_df


def build_planes(aircraft_df: pd.DataFrame):
    planes = {}
    for row in aircraft_df.to_dict("records"):
        plane_id = str(row["aircraft_id"])
        planes[plane_id] = {
            "fixed_cost": float(row["fixed_cost"]),
            "hourly_cost": float(row["hourly_cost"]),
            "initial_airport": str(row["initial_airport"]).strip().upper(),
            "seats": int(row["seats"]),
            "speed": float(row["speed"]),
        }
    return planes


def build_flight_pool(flights_df: pd.DataFrame, itineraries_df: pd.DataFrame):
    merged = flights_df.merge(itineraries_df, on="flight_id", how="left")
    merged["total_ticket_price"] = merged["total_ticket_price"].fillna(0.0)
    merged["total_passenger_count"] = merged["total_passenger_count"].fillna(0)
    merged = merged[merged.total_ticket_price > 0]

    pool = []
    for row in merged.to_dict("records"):
        passenger_count = int(max(1, round(float(row["total_passenger_count"]))))
        pool.append(
            {
                "id": int(row["flight_id"]),
                "origin": str(row["origin"]).strip().upper(),
                "dest": str(row["destination"]).strip().upper(),
                "start": int(row["start_min"]),
                "pass": passenger_count,
                "total_passenger_count": passenger_count,
                "total_ticket_price": float(row["total_ticket_price"]),
            }
        )

    return pool


def sample_schedule(flight_pool, n, rng):
    if not flight_pool:
        return []

    replace = len(flight_pool) < n
    selected = (
        rng.choices(flight_pool, k=n) if replace else rng.sample(flight_pool, k=n)
    )
    sampled = [copy.deepcopy(f) for f in selected]
    sampled.sort(key=lambda x: x["start"])
    return sampled


# Setup
setup_checkpoint_dir()

# Load static training data from prepared CSV files
TRAINING_DATA_DIR = PROJECT_ROOT / "data" / "training"
flights_df, itineraries_df, aircraft_df = load_training_tables(TRAINING_DATA_DIR)

aircraft_df = aircraft_df.drop_duplicates(
    subset="fixed_cost  hourly_cost initial_airport  seats  speed".split()
)

PLANES = build_planes(aircraft_df)
num_planes = len(PLANES)

flight_pool = build_flight_pool(flights_df, itineraries_df)
AIRPORTS = sorted(
    {f["origin"] for f in flight_pool}
    | {f["dest"] for f in flight_pool}
    | {p["initial_airport"] for p in PLANES.values()}
)
dist_dict = create_dist_dict_from_airports(airports_list=AIRPORTS)

logger.info(
    f"Loaded {len(flight_pool)} flights, {len(PLANES)} aircraft, {len(AIRPORTS)} airports"
)

effective_dqn_episodes = scale_episode_budget(
    N_DQN_EPISODES,
    baseline_flights=N_FLIGHTS,
    actual_flights=len(flight_pool),
    minimum=1000,
)
effective_double_dqn_episodes = scale_episode_budget(
    N_DOUBLE_DQN_EPISODES,
    baseline_flights=N_FLIGHTS,
    actual_flights=len(flight_pool),
    minimum=1000,
)

effective_dqn_log_interval = (
    100
    if len(flight_pool) >= 200
    else max(1, min(DQN_LOG_INTERVAL, effective_dqn_episodes // 20))
)
effective_double_dqn_log_interval = (
    100
    if len(flight_pool) >= 200
    else max(1, min(DOUBLE_DQN_LOG_INTERVAL, effective_double_dqn_episodes // 20))
)

logger.info(
    "Training budget | "
    f"DQN episodes: {effective_dqn_episodes} | "
    f"Double DQN episodes: {effective_double_dqn_episodes}"
)

# Initialize agents
dummy_env = AirlineEnv(
    flight_pool,
    PLANES,
    dist_dict,
    cities=AIRPORTS,
    penalty_per_min=PENALTY_PER_MIN,
    use_clipping=TRAIN_USE_CLIPPING,
)
dqn_agent = initialize_dqn_agent(dummy_env)
double_dqn_agent = initialize_dqn_agent(
    dummy_env,
    agent_cls=DoubleDQNAgent,
    hyperparams=DOUBLE_DQN_HYPERPARAMS,
)
FLIGHTS = flight_pool

# ===================================================
# --- MAIN TRAINING LOOP ---
# ===================================================
for iteration in range(1, N_ITERATIONS + 1):
    TrainingLogger.log_iteration_start(iteration, N_ITERATIONS)

    # Reset exploration for new iteration
    reset_agent_exploration(dqn_agent, effective_dqn_episodes)
    reset_agent_exploration(double_dqn_agent, effective_double_dqn_episodes)
    TrainingLogger.log_exploration_reset(dqn_agent.epsilon, dqn_agent.epsilon_decay)

    # Solve the full schedule on every iteration.
    FLIGHTS = flight_pool

    # # Generate disrupted variant for robustness testing
    # dg = DisruptionGenerator(CITIES, dist_dict)
    # FLIGHTS_TEST = dg.generate(
    #     FLIGHTS,
    #     actions=[
    #         {
    #             "action": "add_delay",
    #             "target": "random",
    #             "count": max(1, N_FLIGHTS // 4),
    #             "min_delay": 60,
    #             "max_delay": 180,
    #         },
    #         {
    #             "action": "replace_airport",
    #             "target": list(np.random.choice(AIRPORTS)),
    #             "field": "origin",
    #             "method": "closest",
    #         },
    #     ],
    # )

    # Check feasibility
    max_req_planes, utilization = check_global_feasibility(
        FLIGHTS, list(PLANES.keys()), PLANES, dist_dict
    )
    TrainingLogger.log_feasibility(iteration, utilization, max_req_planes, num_planes)

    # Create environments for training and evaluation
    dqn_train_env = AirlineEnv(
        FLIGHTS,
        PLANES,
        dist_dict,
        cities=AIRPORTS,
        penalty_per_min=PENALTY_PER_MIN,
        use_clipping=TRAIN_USE_CLIPPING,
    )
    double_dqn_train_env = AirlineEnv(
        FLIGHTS,
        PLANES,
        dist_dict,
        cities=AIRPORTS,
        penalty_per_min=PENALTY_PER_MIN,
        use_clipping=TRAIN_USE_CLIPPING,
    )
    eval_env = AirlineEnv(
        FLIGHTS,
        PLANES,
        dist_dict,
        cities=AIRPORTS,
        penalty_per_min=PENALTY_PER_MIN,
        use_clipping=EVAL_USE_CLIPPING,
    )

    dqn_scores = train_dqn_iteration(
        dqn_agent,
        dqn_train_env,
        effective_dqn_episodes,
        iteration,
        early_stopping=True,
        log_interval=effective_dqn_log_interval,
        model_name="DQN",
    )
    double_dqn_scores = train_dqn_iteration(
        double_dqn_agent,
        double_dqn_train_env,
        effective_double_dqn_episodes,
        iteration,
        early_stopping=True,
        log_interval=effective_double_dqn_log_interval,
        model_name="DOUBLE_DQN",
    )

    # Save final model weights
    model_path = get_model_filename(
        iteration,
        len(FLIGHTS),
        len(AIRPORTS),
        num_planes,
        effective_dqn_episodes,
        model_tag="DQN",
    )
    double_dqn_path = get_model_filename(
        iteration,
        len(FLIGHTS),
        len(AIRPORTS),
        num_planes,
        effective_double_dqn_episodes,
        model_tag="DOUBLE_DQN",
    )
    torch.save(dqn_agent.policy_net.state_dict(), model_path)
    torch.save(double_dqn_agent.policy_net.state_dict(), double_dqn_path)
    TrainingLogger.log_model_saved(model_path)
    TrainingLogger.log_model_saved(double_dqn_path)

    # Evaluate on both training and disrupted schedules
    solvers = {
        "Random": RandomSolver(),
        "Greedy": ClosestPlaneGreedySolver(),
        "DQN_Agent": DQNSolver(copy.deepcopy(dqn_agent)),
        "Double_DQN": DQNSolver(copy.deepcopy(double_dqn_agent)),
    }
    schedules_eval = {"TRAINING_DATA": FLIGHTS}  # , "DISRUPTION_TEST": FLIGHTS_TEST}

    evaluate_iteration(eval_env, solvers, schedules_eval, iteration)

log_training_complete()


# ===================================================
# --- FINAL EVALUATION ON UNSEEN TEST SET ---
# ===================================================
# Evaluate on the full schedule.
FINAL_TEST_FLIGHTS = flight_pool

final_eval_env = AirlineEnv(
    FINAL_TEST_FLIGHTS,
    PLANES,
    dist_dict,
    cities=AIRPORTS,
    penalty_per_min=PENALTY_PER_MIN,
    use_clipping=FINAL_EVAL_USE_CLIPPING,
)

max_req_planes, utilization = check_global_feasibility(
    FINAL_TEST_FLIGHTS, list(PLANES.keys()), PLANES, dist_dict
)

log_final_evaluation_start(utilization, max_req_planes)

# Prepare model paths for evaluation
model_paths = {
    iteration: get_model_filename(
        iteration,
        len(FLIGHTS),
        len(AIRPORTS),
        num_planes,
        effective_dqn_episodes,
        model_tag="DQN",
    )
    for iteration in range(1, N_ITERATIONS + 1)
}

double_dqn_paths = {
    iteration: get_model_filename(
        iteration,
        len(FLIGHTS),
        len(AIRPORTS),
        num_planes,
        effective_double_dqn_episodes,
        model_tag="DOUBLE_DQN",
    )
    for iteration in range(1, N_ITERATIONS + 1)
}

evaluate_final_test(
    final_eval_env,
    FINAL_TEST_FLIGHTS,
    dqn_agent,
    model_paths,
    double_dqn_agent=double_dqn_agent,
    double_dqn_model_paths=double_dqn_paths,
)
