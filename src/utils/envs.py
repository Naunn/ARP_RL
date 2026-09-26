"""High-performance Reinforcement Learning flight scheduling platform and baseline solvers."""

import random
from typing import Any

import numpy as np

from src.utils.logging import get_logger

logger = get_logger("plane_assignment")


class AirlineEnv:
    def __init__(
        self,
        flights,
        plane_configs,
        dist_dict,
        cities,
        penalty_per_min=5,
        use_clipping=True,
        flight_window_size=16,
    ):
        self.flights = flights
        self.plane_configs = plane_configs
        self.planes = list(plane_configs.keys())
        self.dist_dict = dist_dict
        self.penalty_per_min = penalty_per_min
        self.use_clipping = use_clipping
        self.flight_window_size = max(1, int(flight_window_size))
        take_offs = [p["start"] for p in self.flights]
        self.operation_time = max(take_offs) - min(take_offs)

        # Pre-calculate invariant data extensions to avoid loop costs
        planes_seats = [config["seats"] for config in self.plane_configs.values()]
        self.avg_pass = sum(planes_seats) / len(planes_seats) if planes_seats else 1.0

        self.max_speed = max(config["speed"] for config in self.plane_configs.values())
        self.max_seats = max(config["seats"] for config in self.plane_configs.values())
        self.max_hourly_cost = max(config.get("hourly_cost", 0.0) for config in self.plane_configs.values())
        self.max_fixed_cost = max(config.get("fixed_cost", 0.0) for config in self.plane_configs.values())
        self.max_reloc_dist = max(dist_dict.values()) if dist_dict else 1.0

        plane_starts = [cfg["initial_airport"] for cfg in self.plane_configs.values()]
        self.cities = sorted(set(cities) | set(plane_starts))
        self.city_to_idx = {city: i for i, city in enumerate(self.cities)}
        self.num_cities = len(self.cities)

        # Pre-build structural identity mapping matrices for ultra-fast one-hot vector lookups
        self.city_one_hot_eye = np.eye(self.num_cities, dtype=np.float32)  # identity matrix

        # Dense copy of dist_dict for vectorized per-plane lookups. Missing pairs (and cities absent
        # from both the instance and dist_dict, mapped to the extra last slot) keep the 500 km
        # default that dist_dict.get(..., 500) uses elsewhere in this class.
        dist_cities = sorted(set(self.cities) | {a for a, _ in dist_dict} | {b for _, b in dist_dict})
        self._dist_idx = {city: i for i, city in enumerate(dist_cities)}
        self._unknown_city_idx = len(dist_cities)
        self._dist_matrix = np.full((len(dist_cities) + 1, len(dist_cities) + 1), 500.0)
        for (origin, dest), dist in dist_dict.items():
            self._dist_matrix[self._dist_idx[origin], self._dist_idx[dest]] = dist

        self.plane_seats = np.array([float(self.plane_configs[p]["seats"]) for p in self.planes])
        self._plane_speed_per_min = np.array([self.plane_configs[p]["speed"] / 60 for p in self.planes])

        self.reset()

    def reset(self) -> tuple[tuple[float, ...], tuple[str, ...], int]:
        self.current_f_idx = 0
        self.times = tuple([0.0] * len(self.planes))
        self.locs = tuple([self.plane_configs[p]["initial_airport"] for p in self.planes])
        self.used_planes = tuple([False] * len(self.planes))
        return (self.times, self.locs, self.current_f_idx)

    def relocation_estimates(self, times, locs, flight) -> tuple[np.ndarray, np.ndarray]:
        """Per-plane relocation distance to `flight`'s origin and earliest feasible start time."""
        loc_idx = np.fromiter(
            (self._dist_idx.get(loc, self._unknown_city_idx) for loc in locs), dtype=np.int64, count=len(locs)
        )
        origin_idx = self._dist_idx.get(flight["origin"], self._unknown_city_idx)
        reloc_dist = self._dist_matrix[loc_idx, origin_idx]
        actual_start = np.maximum(
            flight["start"], np.asarray(times, dtype=np.float64) + reloc_dist / self._plane_speed_per_min
        )
        return reloc_dist, actual_start

    def reset_with_schedule(self, new_flights) -> tuple[tuple[float, ...], tuple[str, ...], int]:
        """Swaps the current operational flight queue mapping layout structures cleanly."""
        self.flights = new_flights
        return self.reset()

    def get_vector_state(self, raw_state=None) -> tuple[np.ndarray, np.ndarray]:
        """Translates current raw simulation tracking structures into scalable tensor arrays."""
        times, locs, f_idx = raw_state if raw_state is not None else (self.times, self.locs, self.current_f_idx)

        # FLEET SYSTEM VECTOR COMPUTATION
        fleet_list = []
        active_f = self.flights[f_idx] if 0 <= f_idx < len(self.flights) else None  # check if flight is active
        active_origin = active_f["origin"] if active_f else None  # get origin of a flight
        active_start_t = float(active_f["start"]) if active_f else 0.0  # get start time of a flight

        for idx, (t, loc) in enumerate(zip(times, locs)):  # loop through all planes
            p_cfg = self.plane_configs[self.planes[idx]]

            # Fast vectorized one-hot parsing via precomputed identity mapping matrix lookup arrays
            fleet_list.append(float(t) / self.operation_time)
            fleet_list.extend(self.city_one_hot_eye[self.city_to_idx.get(loc, 0)])
            fleet_list.extend(
                [
                    float(p_cfg["seats"]) / self.max_seats,
                    float(p_cfg["speed"]) / self.max_speed,
                    float(p_cfg.get("hourly_cost", 0.0)) / max(1.0, self.max_hourly_cost),
                    float(p_cfg.get("fixed_cost", 0.0)) / max(1.0, self.max_fixed_cost),
                    1.0 if self.used_planes[idx] else 0.0,
                ]
            )

            if active_origin:
                dist = self.dist_dict.get((loc, active_origin), self.max_reloc_dist)
                fleet_list.extend(
                    [
                        float(dist) / self.max_reloc_dist,
                        max(active_start_t, t + (dist / (p_cfg["speed"] / 60))) / self.operation_time,
                    ]
                )
            else:
                fleet_list.extend([0.0, 0.0])

        fleet_list.append(float(f_idx) / max(1, len(self.flights) - 1))

        if active_f:
            fleet_list.extend(
                [
                    float(self.city_to_idx.get(active_f["origin"], 0)) / max(1, self.num_cities - 1),
                    float(self.city_to_idx.get(active_f["dest"], 0)) / max(1, self.num_cities - 1),
                    float(active_f["start"]) / self.operation_time,
                    float(active_f["pass"]) / self.avg_pass,
                ]
            )
        else:
            fleet_list.extend([0.0, 0.0, 0.0, 0.0])

        # SEQUENCE LENGTH INVARIANT LOOKAHEAD WINDOW INTERFACE MATRIX
        matrix_list = []
        if active_f:
            for f in self.flights[f_idx : f_idx + self.flight_window_size]:
                matrix_list.append(
                    [
                        float(self.city_to_idx.get(f["origin"], 0)) / max(1, self.num_cities - 1),
                        float(self.city_to_idx.get(f["dest"], 0)) / max(1, self.num_cities - 1),
                        float(f["start"]) / self.operation_time,
                        float(f["pass"]) / self.avg_pass,
                    ]
                )

        while len(matrix_list) < self.flight_window_size:
            matrix_list.append([0.0, 0.0, 0.0, 0.0])

        return (
            np.array(
                fleet_list, dtype=np.float32
            ),  # 1D flat array containing normalized metrics for every plane in the fleet concatenated with the current flight's requirements
            np.array(
                matrix_list, dtype=np.float32
            ),  # 2D array of shape (flight_window_size, 4) representing a normalized snapshot of upcoming flight demands
        )

    def get_state_dim(self) -> tuple[int, int]:
        """Tracks input dimensions to guide neural sequence structural layer sizes."""
        return (len(self.planes) * (1 + self.num_cities + 7) + 1 + 4, 4)

    def get_action_mask(self, raw_state=None) -> np.ndarray:
        """Returns a boolean mask of legal actions for the currently active flight."""
        times, locs, f_idx = raw_state if raw_state is not None else (self.times, self.locs, self.current_f_idx)

        n_actions = len(self.planes)
        if f_idx >= len(self.flights):
            return np.ones(n_actions, dtype=np.bool_)

        f = self.flights[f_idx]

        # Legal actions are capacity-feasible; delay is handled by reward shaping.
        mask = self.plane_seats >= float(f.get("pass", 0.0))

        # Fallback when no plane is big enough: the largest plane, ties broken by least delay.
        if not mask.any():
            _, actual_start = self.relocation_estimates(times, locs, f)
            delay = np.maximum(0.0, actual_start - f["start"])
            largest = self.plane_seats == self.plane_seats.max()
            mask[int(np.argmin(np.where(largest, delay, np.inf)))] = True

        return mask

    def step(self, action_idx: int) -> tuple[tuple[Any, ...], float, bool, dict]:
        f = self.flights[self.current_f_idx]
        p_name = self.planes[action_idx]
        p_cfg = self.plane_configs[p_name]

        p_free_time, p_loc = self.times[action_idx], self.locs[action_idx]
        reloc_dist = self.dist_dict.get((p_loc, f["origin"]), 500)
        flight_dist = self.dist_dict.get((f["origin"], f["dest"]), 500)

        actual_start = max(f["start"], p_free_time + (reloc_dist / (p_cfg["speed"] / 60)))
        arrival_at_dest = actual_start + (flight_dist / (p_cfg["speed"] / 60))

        # Financial balance computations
        # served_ratio = min(1.0, float(p_cfg["seats"]) / max(1.0, float(f.get("pass", 1))))
        revenue = float(f.get("total_ticket_price", 0.0))  # * served_ratio
        operating_cost = float(p_cfg.get("hourly_cost", 0.0)) * (
            ((reloc_dist + flight_dist) / (p_cfg["speed"] / 60)) / 60
        )
        assignment_cost = operating_cost + (
            0.0 if self.used_planes[action_idx] else float(p_cfg.get("fixed_cost", 0.0))
        )

        delay_minutes = max(0.0, actual_start - f["start"])
        delay_multiplier = 1.0 + min(delay_minutes / 60.0, 2.0)
        delay_penalty = delay_minutes * f["pass"] * self.penalty_per_min * delay_multiplier
        capacity_slack_penalty = (
            500.0 * max(0.0, float(p_cfg["seats"]) - float(f.get("pass", 1))) / max(1.0, float(f.get("pass", 1)))
        )
        relocation_penalty = 0.05 * reloc_dist
        reward = (
            revenue
            - assignment_cost
            - (min(delay_penalty, 20000.0) if self.use_clipping else delay_penalty)
            - relocation_penalty
            - capacity_slack_penalty
        )
        if self.use_clipping:
            reward = max(-30000.0, reward)

        # Apply structural mutations safely across elements
        new_times, new_locs, new_used = (
            list(self.times),
            list(self.locs),
            list(self.used_planes),
        )
        new_times[action_idx] = arrival_at_dest
        new_locs[action_idx] = f["dest"]
        new_used[action_idx] = True

        self.times, self.locs, self.used_planes = (
            tuple(new_times),
            tuple(new_locs),
            tuple(new_used),
        )
        self.current_f_idx += 1

        return (
            (self.times, self.locs, self.current_f_idx),
            reward,
            (self.current_f_idx >= len(self.flights)),
            {
                "actual_start": actual_start,
                "arrival_at_dest": arrival_at_dest,
                "p_curr_loc": p_loc,
                "assignment_cost": assignment_cost,
                "revenue": revenue,
                "delay_minutes": delay_minutes,
                "delay_penalty": delay_penalty,
                # "served_ratio": served_ratio,
                "capacity_slack_penalty": capacity_slack_penalty,
                "relocation_penalty": relocation_penalty,
            },
        )


# --- STRUCTURAL BASELINE SOLVER INTERFACES ---
class BaseSolver:
    def choose_action(self, state, env: AirlineEnv) -> int:
        raise NotImplementedError


class RandomSolver(BaseSolver):
    def choose_action(self, state, env: AirlineEnv) -> int:
        return random.randint(0, len(env.planes) - 1)


class ClosestPlaneGreedySolver(BaseSolver):
    """Assigns the craft that minimizes delay and relocation, strictly prioritizing sufficient passenger capacity."""

    def choose_action(self, state, env: AirlineEnv) -> int:
        f = env.flights[env.current_f_idx]
        reloc_dist, actual_start = env.relocation_estimates(env.times, env.locs, f)
        waste = (actual_start - f["start"]) + reloc_dist / 100.0

        # Least-waste plane among those with enough seats; if none has enough, least-waste overall.
        capable = env.plane_seats >= float(f.get("pass", 1))
        if capable.any():
            waste = np.where(capable, waste, np.inf)
        return int(np.argmin(waste))


class DQNSolver(BaseSolver):
    def __init__(self, agent):
        self.agent = agent

    def choose_action(self, state, env: AirlineEnv | None = None) -> int:
        action_mask = env.get_action_mask() if env is not None else None
        return self.agent.choose_action(
            state,
            action_mask=action_mask,
            use_epsilon=False,
        )


# --- CONSOLIDATED CROSS-VALIDATION UNIFIED TESTING STAGE ENGINE ---
def run_unified_execution(
    env: AirlineEnv,
    solver: BaseSolver,
    flights: list,
    solver_name: str = "SOLVER",
    verbose: bool = False,
) -> tuple[float, float]:
    """Runs `solver` over the full schedule and returns (total_profit, total_delay_minutes).

    verbose=True logs a line per flight -- useful for inspecting a small instance, but slow and
    very noisy on ROADEF-sized ones, hence off by default.
    """
    # Wrap the header logs
    if verbose:
        logger.info(f"\n{'=' * 30} {solver_name.upper()} EXECUTION {'=' * 30}")

    state = env.get_vector_state(env.reset_with_schedule(flights))
    total_profit, total_delay_mins, done = 0.0, 0.0, False

    header = (
        f"{'FLIGHT':<8} | {'PAX':<4} | {'PLANE':<10} | {'FROM':<6} | "
        f"{'ORIGIN':<8} | {'DEST':<6} | {'SCHED':>5} | {'ACTUAL':>6} | "
        f"{'ARRIVE':>6} | {'PROFIT'}"
    )
    line_width = len(header) + 2

    if verbose:
        logger.info(header)
        logger.info("-" * line_width)

    while not done:
        f = flights[env.current_f_idx]

        action = solver.choose_action(state, env)
        plane_start_loc = env.locs[action]
        p_name = env.planes[action]

        next_raw_state, reward, done, info = env.step(action)
        flight_delay = max(0.0, info["actual_start"] - f["start"])
        total_delay_mins += flight_delay
        total_profit += reward

        relocated = plane_start_loc != f["origin"]
        from_display = plane_start_loc[:6] if relocated else "-"
        origin_display = f"{f['origin']}*" if relocated else f["origin"]
        actual_start_display = f"{info['actual_start']:.0f}!" if flight_delay > 0 else f"{info['actual_start']:.0f}"

        # Wrap the per-flight logs
        if verbose:
            logger.info(
                f"{f['id']:<8} | {f['pass']:<4} | {p_name.upper():<10} | "
                f"{from_display:<6} | {origin_display:<8} | "
                f"{f['dest']:<6} | {f['start']:>5.0f} | {actual_start_display:>6} | "
                f"{info['arrival_at_dest']:>6.0f} | ${reward:>10,.0f}"
            )
        state = env.get_vector_state(next_raw_state)

    # Wrap the footer summaries
    if verbose:
        logger.info("-" * line_width)
        logger.info(f"TOTAL {solver_name.upper()} PROFIT: ${total_profit:>12,.2f}")
        logger.info(f"TOTAL SYSTEM DELAY: {total_delay_mins:.0f} minutes")
        logger.info("LEGEND: (!) Delayed | (*) Relocated | SCHED/ACTUAL/ARRIVE in minutes from T=0")
        logger.info("=" * line_width + "\n")

    return total_profit, total_delay_mins
