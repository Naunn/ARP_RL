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


def generate_trap_schedule(n, cities, start_time_range, pass_range):
    # Generate base random flight pool
    flights = []
    for i in range(n):
        orig, dest = random.sample(cities, 2)
        flights.append(
            {
                "id": 101 + i,
                "origin": orig,
                "dest": dest,
                "start": random.randint(*start_time_range),
                "pass": random.randint(*pass_range),
            }
        )

    # Dynamic Trap Injection (No hardcoded array indices)
    # Group flights into Early (Yield Trap) and Late (Concurrency Trap) windows
    t_min, t_max = start_time_range
    bottleneck_time = int(t_max * 0.90)

    for i, f in enumerate(flights):
        # Force the first ~20% of flights into a high-capacity hub-to-hub trap
        if i < max(2, int(n * 0.2)):
            f["origin"], f["dest"] = cities[0], cities[1]
            f["start"] = random.randint(t_min + 30, t_min + 200)
            f["pass"] = int(pass_range[1] * 0.95)

        # Force the last ~40% of flights to cluster simultaneously at the end
        elif i >= n - max(3, int(n * 0.4)):
            f["origin"], f["dest"] = random.sample(cities[:3], 2)
            f["start"] = random.randint(bottleneck_time - 15, bottleneck_time + 10)

    # Final structural maintenance
    flights.sort(key=lambda x: x["start"])
    for idx, f in enumerate(flights):
        f["id"] = 101 + idx

    return flights
