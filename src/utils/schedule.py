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
    current_time = start_time_range[0]
    max_time = start_time_range[1]

    for i in range(1, n + 1):
        # Ensure origin and destination are not the same
        origin, dest = random.sample(cities, 2)

        # Ensure start time is not smaller than previous
        # We add a random gap (0 to 60 mins) to make it realistic
        current_time = random.randint(current_time, max_time)

        flight = {
            "id": 100 + i,  # Format like 101, 102, etc.
            "origin": origin,
            "dest": dest,
            "start": current_time,
            "pass": random.randint(pass_range[0], pass_range[1]),
        }
        generated_flights.append(flight)

    # Though generated in order, we sort just to be safe for the RL environment
    generated_flights.sort(key=lambda x: x["start"])
    return generated_flights


def check_global_feasibility(flights, planes, plane_configs, dist_dict):
    """
    Analyzes schedule density and fleet capacity.
    Returns (peak_demand, utilization_percent).
    """
    if not flights or not planes:
        return 0, 0.0

    # Calculate Average Fleet Speed
    avg_speed_km_min = (
        sum(p["speed"] for p in plane_configs.values()) / len(plane_configs)
    ) / 60

    events = []
    total_workload_mins = 0

    # Process Flights into Timeline Events
    for f in flights:
        dist = dist_dict.get((f["origin"], f["dest"]), 500)
        duration = dist / avg_speed_km_min
        total_workload_mins += duration

        # Mark simultaneous demand peaks
        events.append((f["start"], 1))  # Flight starts
        events.append((f["start"] + duration, -1))  # Flight ends

    # Calculate Peak Concurrency (The "Bottleneck")
    # upper bound of the number of planes needed at a single hour
    events.sort()
    max_planes_needed = 0
    current_active = 0
    for _, change in events:
        current_active += change
        max_planes_needed = max(max_planes_needed, current_active)

    # Calculate Global Utilization (The "Time Budget")
    start_time = min(f["start"] for f in flights)
    # Give a 2-hour buffer for the final flight to land
    end_time = max(f["start"] for f in flights) + 120

    # we assume all planes can be used for the whole day (flight time, first start till last start)
    operating_window = end_time - start_time
    total_available_mins = len(planes) * operating_window

    # If I spread all the work out perfectly over 24 hours, how busy are my planes?
    # <20% => not tht busy
    # >80% => solution without delays may be impossible
    utilization = (
        (total_workload_mins / total_available_mins) * 100
        if total_available_mins > 0
        else 0
    )

    return max_planes_needed, utilization
