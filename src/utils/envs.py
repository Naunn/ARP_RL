import random

from src.log_config import get_logger

logger = get_logger("plane_assignment")


class AirlineEnv:
    def __init__(
        self,
        flights,
        plane_configs,
        dist_dict,
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

    def simulate_step(self, action_idx):
        """
        Calculates the reward for an action without updating the environment state.
        Useful for greedy baselines.
        """
        # Store current state
        original_times = self.times
        original_locs = self.locs
        original_idx = self.current_f_idx

        # Get result
        res = self.step(action_idx)

        # Restore state
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

        # --- REWARD CLIPPING ---
        delay_mins = max(0, actual_start - f["start"])
        # We cap the penalty at a maximum to keep the agent learning during disruptions
        delay_penalty = delay_mins * f["pass"] * self.penalty_per_min
        if self.use_clipping:
            # Apply the "Damage Control" caps
            reward = revenue - fuel_cost - min(delay_penalty, 20000)
            reward = max(-30000, reward)
        else:
            # Raw, unshielded financial loss
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
    """
    Naively optimal based on proximity:
    Picks the plane that is physically closest to the flight's origin
    and can arrive soonest.
    """

    def choose_action(self, state, env):
        f = env.flights[env.current_f_idx]
        best_action = 0
        min_waste = float("inf")

        for action_idx, p_name in enumerate(env.planes):
            p_cfg = env.plane_configs[p_name]
            p_loc = env.locs[action_idx]
            p_free_time = env.times[action_idx]

            # Calculate relocation effort
            reloc_dist = env.dist_dict.get((p_loc, f["origin"]), 500)
            reloc_time = reloc_dist / (p_cfg["speed"] / 60)

            # The "Waste" metric: How long after the scheduled start
            # can this plane actually begin?
            actual_start_possible = max(f["start"], p_free_time + reloc_time)
            delay_waste = actual_start_possible - f["start"]

            # Tie-breaker: If two planes can start on time, pick the one that traveled less
            total_waste_score = delay_waste + (reloc_dist / 100)

            if total_waste_score < min_waste:
                min_waste = total_waste_score
                best_action = action_idx

        return best_action


class QLearningSolver(BaseSolver):
    def __init__(self, agent):
        self.agent = agent

    def choose_action(self, state, env):
        # We use the agent's logic but force epsilon=0 for evaluation
        return self.agent.choose_action(state, use_epsilon=False)


def run_unified_execution(env, solver, flights, name="SOLVER"):
    logger.info(f"\n{'=' * 30} {name.upper()} EXECUTION {'=' * 30}")
    state = env.reset_with_schedule(flights)
    total_profit = 0
    total_delay_mins = 0
    done = False

    # Expanded Header with Timeline: SCHED, ACTUAL, ARRIVE
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

        # Location before movement
        plane_start_loc = env.locs[action]

        next_state, reward, done, info = env.step(action)

        flight_delay = max(0, info["actual_start"] - f["start"])
        total_delay_mins += flight_delay
        total_profit += reward

        # Indicator Logic
        delay_flag = "!" if flight_delay > 0 else ""
        # Actual start string with the delay flag
        act_start_disp = f"{info['actual_start']:>4.0f}{delay_flag}"

        # Relocation Logic
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
