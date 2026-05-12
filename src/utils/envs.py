class AirlineEnv:
    """
    Manages airline fleet state, flight logistics, and economic calculations.
    """

    def __init__(
        self, flights, plane_configs, dist_dict, fuel_price=3, penalty_per_min=5
    ):
        """
        Initialize the environment with flights, aircraft, and pricing.
        """
        self.flights = flights
        self.plane_configs = plane_configs
        self.planes = list(plane_configs.keys())
        self.dist_dict = dist_dict
        self.fuel_price = fuel_price
        self.penalty_per_min = penalty_per_min
        self.reset()

    def reset(self):
        """
        Resets the simulation state.
        :return: Initial state (times_tuple, locs_tuple, flight_idx)
        """
        self.current_f_idx = 0
        self.times = tuple([0.0] * len(self.planes))
        self.locs = tuple(["lodz"] * len(self.planes))
        return (self.times, self.locs, self.current_f_idx)

    def step(self, action_idx):
        """
        Assigns a plane to a flight and calculates the outcome.
        :return: (next_state, reward, done, info_dict)
        """
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
        delay_penalty = (
            max(0, actual_start - f["start"]) * f["pass"] * self.penalty_per_min
        )

        reward = revenue - fuel_cost - delay_penalty

        # State Update
        new_times = list(self.times)
        new_locs = list(self.locs)
        new_times[action_idx] = arrival_at_dest
        new_locs[action_idx] = f["dest"]

        self.times, self.locs = tuple(new_times), tuple(new_locs)
        self.current_f_idx += 1

        done = self.current_f_idx >= len(self.flights)

        # Info for logging
        info = {
            "actual_start": actual_start,
            "arrival_at_dest": arrival_at_dest,
            "p_curr_loc": p_loc,
        }

        return (self.times, self.locs, self.current_f_idx), reward, done, info
