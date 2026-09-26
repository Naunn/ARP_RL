"""Problem-instance loading: the single source of truth for building the FLIGHTS/PLANES/
AIRPORTS/dist_dict inputs AirlineEnv expects, whether from real ROADEF2009 competition data
(roadef.py) or synthetic generators (synthetic.py). Routing every experiment script through this
module -- rather than ad hoc CSV sampling -- is what makes their results comparable: they end up
solving the identical instance, not just statistically similar ones.
"""

from src.instances.common import build_flight_pool, build_planes, subsample_instance
from src.instances.roadef import discover_roadef_instances, load_roadef_instance
from src.instances.synthetic import generate_random_flights, generate_trap_schedule

__all__ = [
    "build_flight_pool",
    "build_planes",
    "subsample_instance",
    "discover_roadef_instances",
    "load_roadef_instance",
    "generate_random_flights",
    "generate_trap_schedule",
]
