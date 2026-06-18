import random

import numpy as np

from src.log_config import get_logger

logger = get_logger("plane_assignment")


class AirlineEnv:
    def __init__(
        self,
        flights,
        plane_configs,
        dist_dict,
        cities,
        fuel_price=3,
        penalty_per_min=5,
        use_clipping=True,
    ):
        self.flights = flights
        self.plane_configs = plane_configs
        self.planes = list(plane_configs.keys())
        self.dist_dict = dist_dict
        self.fuel_price = fuel_price
        self.penalty_per_min = penalty_per_min
        self.use_clipping = use_clipping

        # Normalization
        start_times = [f["start"] for f in self.flights]
        self.operation_time = 1440.0
        self.min_start = min(start_times) if start_times else 0.0

        planes_seats = [config["seats"] for config in self.plane_configs.values()]
        self.avg_pass = sum(planes_seats) / len(planes_seats)

        self.max_speed = max(config["speed"] for config in self.plane_configs.values())
        self.max_seats = max(config["seats"] for config in self.plane_configs.values())
        self.max_fuel_use = max(
            config["fuel_use"] for config in self.plane_configs.values()
        )
        self.max_base_fare = max(
            config["base_fare"] for config in self.plane_configs.values()
        )
        self.max_rate_per_km = max(
            config["rate_per_km"] for config in self.plane_configs.values()
        )
        self.max_reloc_dist = max(dist_dict.values()) if dist_dict else 1.0

        # Build stable indexing for strings -> neural arrays
        self.cities = sorted(list(set(cities + ["lodz"])))
        self.city_to_idx = {city: i for i, city in enumerate(self.cities)}

        self.reset()

    def reset(self):
        self.current_f_idx = 0
        self.times = tuple([0.0] * len(self.planes))
        self.locs = tuple(["lodz"] * len(self.planes))
        return (self.times, self.locs, self.current_f_idx)

    def reset_with_schedule(self, new_flights):
        """Swaps the flight list and resets the state."""
        self.flights = new_flights
        return self.reset()

    def get_vector_state(self, raw_state=None):
        """
        Translates raw simulation elements into separate state representations:
        1. Fleet State Vector (Fixed tracking context)
        2. Flight Descriptor Matrix (Sequence length invariant shape)
        """
        if raw_state is None:
            times, locs, f_idx = self.times, self.locs, self.current_f_idx
        else:
            times, locs, f_idx = raw_state

        num_cities = len(self.cities)

        # FLEET CONFIG (Fixed Dimension)
        fleet_elements = []

        active_flight = self.flights[f_idx] if 0 <= f_idx < len(self.flights) else None
        active_origin = active_flight["origin"] if active_flight is not None else None
        active_start_t = (
            float(active_flight["start"]) if active_flight is not None else 0.0
        )

        for plane_idx, (t, loc) in enumerate(zip(times, locs)):
            fleet_elements.append(float(t) / self.operation_time)

            one_hot = [0.0] * num_cities
            one_hot[self.city_to_idx.get(loc, 0)] = 1.0
            fleet_elements.extend(one_hot)

            plane_name = self.planes[plane_idx]
            plane_cfg = self.plane_configs[plane_name]
            fleet_elements.append(float(plane_cfg["seats"]) / self.max_seats)
            fleet_elements.append(float(plane_cfg["speed"]) / self.max_speed)
            fleet_elements.append(float(plane_cfg["fuel_use"]) / self.max_fuel_use)
            fleet_elements.append(
                float(plane_cfg["rate_per_km"]) / self.max_rate_per_km
            )

            if active_origin is not None:
                reloc_dist = self.dist_dict.get(
                    (loc, active_origin), self.max_reloc_dist
                )
                reloc_time = reloc_dist / (plane_cfg["speed"] / 60)
                normalized_reloc_dist = float(reloc_dist) / self.max_reloc_dist
                normalized_arrival = (
                    max(active_start_t, t + reloc_time) / self.operation_time
                )
                fleet_elements.append(normalized_reloc_dist)
                fleet_elements.append(normalized_arrival)
            else:
                fleet_elements.extend([0.0, 0.0])

        # Add a normalized pointer to the active flight index being evaluated
        fleet_elements.append(float(f_idx) / max(1, len(self.flights) - 1))

        # CURRENT FLIGHT CONTEXT
        if active_flight is not None:
            active_orig = float(self.city_to_idx.get(active_flight["origin"], 0)) / max(
                1, num_cities - 1
            )
            active_dest = float(self.city_to_idx.get(active_flight["dest"], 0)) / max(
                1, num_cities - 1
            )
            active_start = float(active_flight["start"]) / self.operation_time
            active_pass = float(active_flight["pass"]) / self.avg_pass
            fleet_elements.extend([active_orig, active_dest, active_start, active_pass])
        else:
            fleet_elements.extend([0.0, 0.0, 0.0, 0.0])

        # FLIGHT LIST (Variable Dimension: N x Features)
        flight_matrix = []
        for f in self.flights:
            f_orig = float(self.city_to_idx.get(f["origin"], 0)) / max(
                1, num_cities - 1
            )
            f_dest = float(self.city_to_idx.get(f["dest"], 0)) / max(1, num_cities - 1)
            f_start = float(f["start"]) / self.operation_time
            f_pass = float(f["pass"]) / self.avg_pass
            flight_matrix.append([f_orig, f_dest, f_start, f_pass])

        # Return them wrapped up as a tuple of numpy arrays
        return (
            np.array(fleet_elements, dtype=np.float32),
            np.array(flight_matrix, dtype=np.float32),
        )

    def get_state_dim(self):
        """
        Dynamically tracks layer widths based on geometry features.
        Returns a tuple of (fleet_state_dim, individual_flight_feature_dim).
        """
        n_planes = len(self.planes)
        n_cities = len(self.cities)

        per_plane_features = (
            1 + n_cities + 6
        )  # time + location one-hot + static and relocation features
        fleet_dim = n_planes * per_plane_features + 1 + 4
        flight_feature_dim = 4  # [origin_idx, dest_idx, norm_start, norm_passengers]

        return fleet_dim, flight_feature_dim

    def simulate_step(self, action_idx):
        """Calculates the reward for an action without updating the environment state."""
        original_times = self.times
        original_locs = self.locs
        original_idx = self.current_f_idx

        res = self.step(action_idx)

        self.times = original_times
        self.locs = original_locs
        self.current_f_idx = original_idx

        return res

    def step(self, action_idx):
        f = self.flights[self.current_f_idx]
        p_name = self.planes[action_idx]
        p_cfg = self.plane_configs[p_name]

        p_free_time = self.times[action_idx]
        p_loc = self.locs[action_idx]

        # Logistics
        reloc_dist = self.dist_dict.get((p_loc, f["origin"]), 500)
        flight_dist = self.dist_dict.get((f["origin"], f["dest"]), 500)
        reloc_time = reloc_dist / (p_cfg["speed"] / 60)

        actual_start = max(f["start"], p_free_time + reloc_time)
        arrival_at_dest = actual_start + (flight_dist / (p_cfg["speed"] / 60))

        # Economics
        ticket_price = p_cfg["base_fare"] + (flight_dist * p_cfg["rate_per_km"])
        revenue = min(f["pass"], p_cfg["seats"]) * ticket_price
        fuel_cost = (
            (reloc_dist + flight_dist) * (p_cfg["fuel_use"] / 100) * self.fuel_price
        )

        # Reward Clipping
        delay_mins = max(0, actual_start - f["start"])
        delay_penalty = delay_mins * f["pass"] * self.penalty_per_min
        if self.use_clipping:
            reward = revenue - fuel_cost - min(delay_penalty, 20000)
            reward = max(-30000, reward)
        else:
            reward = revenue - fuel_cost - delay_penalty

        # State Update
        new_times = list(self.times)
        new_locs = list(self.locs)
        new_times[action_idx] = arrival_at_dest
        new_locs[action_idx] = f["dest"]

        self.times, self.locs = tuple(new_times), tuple(new_locs)
        self.current_f_idx += 1
        done = self.current_f_idx >= len(self.flights)

        info = {
            "actual_start": actual_start,
            "arrival_at_dest": arrival_at_dest,
            "p_curr_loc": p_loc,
        }

        return (self.times, self.locs, self.current_f_idx), reward, done, info


class BaseSolver:
    def choose_action(self, state, env):
        raise NotImplementedError


class RandomSolver(BaseSolver):
    def choose_action(self, state, env):
        return random.randint(0, len(env.planes) - 1)


class ClosestPlaneGreedySolver(BaseSolver):
    """Picks the plane that is physically closest to the flight's origin and can arrive soonest."""

    def choose_action(self, state, env):
        f = env.flights[env.current_f_idx]
        best_action = 0
        min_waste = float("inf")

        for action_idx, p_name in enumerate(env.planes):
            p_cfg = env.plane_configs[p_name]
            p_loc = env.locs[action_idx]
            p_free_time = env.times[action_idx]

            reloc_dist = env.dist_dict.get((p_loc, f["origin"]), 500)
            reloc_time = reloc_dist / (p_cfg["speed"] / 60)

            actual_start_possible = max(f["start"], p_free_time + reloc_time)
            delay_waste = actual_start_possible - f["start"]
            total_waste_score = delay_waste + (reloc_dist / 100)

            if total_waste_score < min_waste:
                min_waste = total_waste_score
                best_action = action_idx

        return best_action


class DQNSolver(BaseSolver):
    """Deep Q-Network sequence-invariant valuation wrapper."""

    def __init__(self, agent):
        self.agent = agent

    def choose_action(self, state, env=None):
        """
        Accepts the execution parameters natively from the test runner.
        Passes the state tuple straight to the agent without splitting it.
        """
        # state is already the tuple package: (fleet_state, flight_matrix)
        # Force use_epsilon=False to ensure clean evaluation scores
        return self.agent.choose_action(state, use_epsilon=False)


def run_unified_execution(env, solver, flights, name="SOLVER"):
    logger.info(f"\n{'=' * 30} {name.upper()} EXECUTION {'=' * 30}")

    # Initialize schedule structure tracking variables
    raw_state = env.reset_with_schedule(flights)
    state = env.get_vector_state(raw_state)
    total_profit = 0
    total_delay_mins = 0
    done = False

    header = (
        f"{'FLIGHT':<8} | {'PAX':<4} | {'PLANE':<10} | {'FROM':<6} | {'ORIGIN':<8} | "
        f"{'DEST':<6} | {'SCHED':>5} | {'ACTUAL':>6} | {'ARRIVE':>6} | {'PROFIT'}"
    )
    line_width = len(header) + 2
    logger.info(header)
    logger.info("-" * line_width)

    while not done:
        f = flights[env.current_f_idx]
        action = solver.choose_action(state, env)
        p_name = env.planes[action]

        plane_start_loc = env.locs[action]

        next_raw_state, reward, done, info = env.step(action)
        next_state = env.get_vector_state(next_raw_state)

        flight_delay = max(0, info["actual_start"] - f["start"])
        total_delay_mins += flight_delay
        total_profit += reward

        delay_flag = "!" if flight_delay > 0 else ""
        act_start_disp = f"{info['actual_start']:>4.0f}{delay_flag}"

        relocated = plane_start_loc != f["origin"]
        from_disp = plane_start_loc[:6] if relocated else "-"
        orig_disp = f"{f['origin']}{'*' if relocated else ''}"

        logger.info(
            f"{f['id']:<8} | {f['pass']:<4} | {p_name.upper():<10} | {from_disp:<6} | "
            f"{orig_disp:<8} | {f['dest']:<6} | {f['start']:>5.0f} | {act_start_disp:>6} | "
            f"{info['arrival_at_dest']:>6.0f} | ${reward:>10,.0f}"
        )
        state = next_state

    logger.info("-" * line_width)
    logger.info(f"TOTAL {name.upper()} PROFIT: ${total_profit:>12,.2f}")
    logger.info(f"TOTAL SYSTEM DELAY: {total_delay_mins:.0f} minutes")
    logger.info(
        "LEGEND: (!) Delayed | (*) Relocated | SCHED/ACTUAL/ARRIVE in minutes from T=0"
    )
    logger.info("=" * line_width + "\n")

    return total_profit, total_delay_mins
