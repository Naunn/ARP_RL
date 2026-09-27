"""Synthetic instance generators: fast, parametrized flight schedules for quick iteration and
for stress-testing specific structural patterns (e.g. hub bottlenecks). For benchmarking against
a fixed, real problem instance instead, see roadef.py.
"""

import random


def generate_random_flights(n, cities, start_time_range, pass_range):
    """
    Generates a list of n random flights with constraints.

    :param n: Number of flights to generate.
    :param cities: List of available city names.
    :param start_time_range: Tuple (min_start, max_day_time)
                            e.g., (600, 1440) for 10:00 to 24:00.
    :param pass_range: Tuple (min_pass, max_pass) e.g., (10, 150).
    :return: List of flight dictionaries sorted by start time.
    """
    generated_flights = []

    # We start with the minimum allowed time
    min_time = start_time_range[0]
    max_time = start_time_range[1]

    for i in range(1, n + 1):
        # Ensure origin and destination are not the same
        origin, dest = random.sample(cities, 2)

        # uniformly sample a start time within the allowed range
        start_time = random.randint(min_time, max_time)

        flight = {
            "id": 000,
            "origin": origin,
            "dest": dest,
            "start": start_time,
            "pass": random.randint(pass_range[0], pass_range[1]),
        }
        generated_flights.append(flight)

    # Though generated in order, we sort just to be safe for the RL environment
    generated_flights.sort(key=lambda x: x["start"])
    for i, flight in enumerate(generated_flights, start=100):
        flight["id"] = i

    return generated_flights


def generate_trap_schedule(n, cities, start_time_range, pass_range):
    # Generate base random flight pool
    flights = generate_random_flights(n, cities, start_time_range, pass_range)

    # Dynamic Trap Injection (No hardcoded array indices)
    # Group flights into Early (Yield Trap) and Late (Concurrency Trap) windows
    t_min, t_max = start_time_range
    bottleneck_time = int(t_max * 0.90)

    for i, f in enumerate(flights):
        # Force the first ~20% of flights into a high-capacity hub-to-hub trap
        rand_origin, rand_dest = random.sample(cities, 2)
        if i < max(2, int(n * 0.2)):
            random_direction = random.sample([rand_origin, rand_dest], 2)
            f["origin"], f["dest"] = random_direction[0], random_direction[1]
            f["start"] = random.randint(t_min + 30, t_min + 200)
            f["pass"] = int(
                random.randint(pass_range[0], pass_range[1]) * 1.5
            )  # Increase passenger count to simulate congestion

        # Force the last ~40% of flights to cluster simultaneously at the end
        elif i >= n - max(3, int(n * 0.4)):
            f["origin"], f["dest"] = random.sample(cities[:3], 2)
            f["start"] = random.randint(bottleneck_time - 15, bottleneck_time + 10)

    # Final structural maintenance
    flights.sort(key=lambda x: x["start"])
    for idx, f in enumerate(flights):
        f["id"] = 101 + idx

    return flights


SCHEDULE_GENERATORS = {
    "random": generate_random_flights,
    "trap": generate_trap_schedule,
}


def synthetic_schedule_like(reference_flights: list[dict], kind: str, n_flights: int, n_cities: int) -> list[dict]:
    """Synthetic "random" or "trap" schedule drawn from the same world as `reference_flights`.

    Cities are sampled from the reference schedule's airports, start times and passenger counts span
    the reference ranges, and fares use the reference's passenger-weighted average fare -- so a
    synthetic schedule stays comparable (same airports/distances, same price level) to the real
    one it was derived from. Uses the global `random` RNG, so it follows set_seed().
    """
    if kind not in SCHEDULE_GENERATORS:
        raise ValueError(f"Unknown schedule kind {kind!r}; expected one of {sorted(SCHEDULE_GENERATORS)}")
    min_cities = 3 if kind == "trap" else 2  # the trap generator draws its bottleneck from cities[:3]
    airports = sorted({f["origin"] for f in reference_flights} | {f["dest"] for f in reference_flights})
    if not min_cities <= n_cities <= len(airports):
        raise ValueError(f"n_cities must be between {min_cities} and {len(airports)} for {kind!r}, got {n_cities}")

    starts = [f["start"] for f in reference_flights]
    passengers = [f["pass"] for f in reference_flights]
    fare_per_passenger = sum(f["total_ticket_price"] for f in reference_flights) / sum(passengers)

    flights = SCHEDULE_GENERATORS[kind](
        n=n_flights,
        cities=random.sample(airports, n_cities),
        start_time_range=(min(starts), max(starts)),
        pass_range=(min(passengers), max(passengers)),
    )
    for f in flights:
        f["total_ticket_price"] = f["pass"] * fare_per_passenger
    return flights
