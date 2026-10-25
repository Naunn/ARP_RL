"""Builds the concrete schedule an experiment trains on, from a loaded ROADEF instance."""

from src.instances.common import subsample_instance
from src.instances.roadef import Instance
from src.instances.synthetic import synthetic_schedule_like

SCHEDULE_TYPES = ("sample", "random", "trap")


def build_schedule(
    base: Instance,
    schedule_type: str,
    max_flights: int | None,
    max_planes: int | None,
    n_cities: int,
    seed: int,
) -> Instance:
    """Returns (flights, planes, airports, dist_dict) for one experiment schedule.

    - "sample": max_flights real flights sampled from the base instance
    - "random" / "trap": max_flights synthetic flights between n_cities of the base instance's
      airports, priced like it (see synthetic_schedule_like)

    The fleet is always max_planes aircraft sampled from the base instance, and dist_dict is the
    base instance's (a superset of the airports used). None for max_* keeps the full count.
    """
    flights, planes, _, dist_dict = base
    if schedule_type == "sample":
        flights, planes, airports = subsample_instance(
            flights, planes, max_flights=max_flights, max_planes=max_planes, seed=seed
        )
    elif schedule_type in ("random", "trap"):
        n_flights = max_flights if max_flights is not None else len(flights)
        synthetic = synthetic_schedule_like(flights, schedule_type, n_flights, n_cities)
        flights, planes, airports = subsample_instance(
            synthetic, planes, max_flights=None, max_planes=max_planes, seed=seed
        )
    else:
        raise ValueError(f"Unknown schedule type {schedule_type!r}; expected one of {SCHEDULE_TYPES}")
    return flights, planes, airports, dist_dict
