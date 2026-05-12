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
