"""Loads the real ROADEF2009 competition instances checked into data/ (sets A1, SA2, SetB1,
SetB2) as fixed benchmark problem instances, so different models/experiments can be compared on
the exact same input instead of statistically-similar random samples -- and so "a bigger
instance" is a real, available axis (A-scale: ~600 flights/84 aircraft; B-scale: ~1400
flights/255 aircraft) rather than something to synthesize.
"""

from pathlib import Path
from typing import Dict, List, Tuple

from src.instances.common import build_flight_pool, build_planes
from src.utils.data_prep import load_all_clean_data
from src.utils.dist import create_dist_dict_from_airports

# Root directories (relative to the project root), each holding one ROADEF instance per
# subdirectory. A1/SA2 are the "A" (smaller) scale; SetB1/SetB2 are the "B" (larger) scale.
INSTANCE_ROOTS = [
    "data/A1_6088570",
    "data/SA2_6088590",
    "data/SetB1_5/SetB1",
    "data/SetB2_5/SetB2",
]

Instance = Tuple[List[dict], Dict[str, dict], List[str], Dict[Tuple[str, str], float]]


def discover_roadef_instances(project_root: Path) -> Dict[str, Path]:
    """Scans the known ROADEF instance roots for subdirectories containing flights.csv.

    Returns {instance_name: instance_dir}, e.g. {"A01_6088570": Path(...), "B_01": Path(...)}.
    """
    instances: Dict[str, Path] = {}
    for root in INSTANCE_ROOTS:
        root_path = project_root / root
        if not root_path.is_dir():
            continue
        for candidate in sorted(root_path.iterdir()):
            if candidate.is_dir() and (candidate / "flights.csv").exists():
                instances[candidate.name] = candidate
    return instances


def load_roadef_instance(instance_dir: Path) -> Instance:
    """Loads one ROADEF instance directory into the FLIGHTS/PLANES/AIRPORTS/dist_dict shapes
    AirlineEnv expects.

    Note: inter-airport distances are computed geodesically via create_dist_dict_from_airports
    (the same distance model synthetic instances use) rather than from the instance's own
    dist.csv nominal_time column, so results stay comparable across real and synthetic instances.
    """
    tables = load_all_clean_data(
        flights_p=instance_dir / "flights.csv",
        dist_p=instance_dir / "dist.csv",
        itineraries_p=instance_dir / "itineraries.csv",
        aircraft_p=instance_dir / "aircraft.csv",
    )

    flights = build_flight_pool(tables["flights"], tables["itineraries"])
    planes = build_planes(tables["aircraft"])
    airports = sorted(
        {f["origin"] for f in flights} | {f["dest"] for f in flights} | {p["initial_airport"] for p in planes.values()}
    )
    dist_dict = create_dist_dict_from_airports(airports_list=airports)

    return flights, planes, airports, dist_dict
