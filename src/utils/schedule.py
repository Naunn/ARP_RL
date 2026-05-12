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
    total_flight_time = 0
    for f in flights:
        dist = dist_dict.get((f["origin"], f["dest"]), 500)
        # Use average speed of fleet
        avg_speed = sum(p["speed"] for p in plane_configs.values()) / len(plane_configs)
        total_flight_time += dist / (avg_speed / 60)

    # Calculate operating window (from first flight start to last flight end-ish)
    start_time = min(f["start"] for f in flights)
    end_time = max(f["start"] for f in flights) + 120  # Estimate last flight duration

    available_time = len(planes) * (end_time - start_time)

    utilization = (total_flight_time / available_time) * 100
    return utilization  # If > 80-90%, it's likely unsolvable due to relocation needs.


def calculate_max_concurrency(flights, plane_configs, dist_dict):
    events = []
    avg_speed = 900 / 60  # km/min

    for f in flights:
        dist = dist_dict.get((f["origin"], f["dest"]), 500)
        duration = dist / avg_speed

        # Add a "Start" event and an "End" event
        events.append((f["start"], 1))  # Plane becomes busy
        events.append((f["start"] + duration, -1))  # Plane becomes free

    # Sort by time
    events.sort()

    max_planes_needed = 0
    current_planes = 0
    for time, change in events:
        current_planes += change
        max_planes_needed = max(max_planes_needed, current_planes)

    return max_planes_needed
