import pandas as pd

from src.config import PLANES_TEMPLATES


def generate_fleet(fleet_counts):
    """
    Creates a full fleet based on template names and counts.

    :param fleet_counts: Dict like {"BOEING": 5, "AIRBUS": 2}
    :return: Flat dictionary of plane instances for the environment.
    """
    full_fleet = {}
    for type_name, count in fleet_counts.items():
        if type_name not in PLANES_TEMPLATES:
            continue

        template = PLANES_TEMPLATES[type_name]
        for i in range(1, count + 1):
            # Creates names like 'boeing_1', 'boeing_2', etc.
            plane_id = f"{type_name.lower()}_{i}"
            full_fleet[plane_id] = template.copy()

    return full_fleet


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
