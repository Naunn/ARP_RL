"""Canonical converters from raw tabular flight/aircraft data into the FLIGHTS/PLANES shapes
AirlineEnv expects. Shared by both real ROADEF instance loading (roadef.py) and synthetic
instance generation (synthetic.py), so every instance source produces identically-shaped output.
"""

import random
from typing import Dict, List, Optional, Tuple

import pandas as pd


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
                "total_ticket_price": float(row["total_ticket_price"]),
            }
        )

    return pool


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


def subsample_instance(
    flights: List[dict],
    planes: Dict[str, dict],
    max_flights: Optional[int] = None,
    max_planes: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[List[dict], Dict[str, dict], List[str]]:
    """Randomly downsizes a loaded instance's flights/planes for faster iteration, and recomputes
    the airport list to match. Pass max_flights/max_planes=None to leave that side untouched.

    The instance's dist_dict does not need rebuilding after this: it stays a (harmless) superset
    of the airports actually used by the returned flights/planes.
    """
    rng = random.Random(seed)

    if max_planes is not None and max_planes < len(planes):
        kept_ids = rng.sample(list(planes), max_planes)
        planes = {plane_id: planes[plane_id] for plane_id in kept_ids}

    if max_flights is not None and max_flights < len(flights):
        flights = rng.sample(flights, max_flights)
    flights = sorted(flights, key=lambda f: f["start"])

    airports = sorted(
        {f["origin"] for f in flights} | {f["dest"] for f in flights} | {p["initial_airport"] for p in planes.values()}
    )
    return flights, planes, airports
