"""Distance utility helpers for airport-to-airport matrices."""

from itertools import permutations


def create_dist_dict_from_airports(airports_list: list[str] | None = None) -> dict[tuple[str, str], float]:
    """Create a full directed distance dictionary keyed by (origin, destination) airport IATA codes."""
    if not airports_list:
        return {}

    normalized_codes = [str(code).strip().upper() for code in airports_list if str(code).strip()]
    if not normalized_codes:
        return {}

    unique_codes = list(dict.fromkeys(normalized_codes))

    # Lazy imports keep module import light and avoid blocking unrelated imports.
    import airportsdata
    from geopy.distance import geodesic

    airports_db = airportsdata.load("IATA")
    coords_cache: dict[str, tuple[float, float]] = {}

    for code in unique_codes:
        if code in airports_db:
            airport = airports_db[code]
            coords_cache[code] = (airport["lat"], airport["lon"])

    dist_dict: dict[tuple[str, str], float] = {}
    for code_a, code_b in permutations(unique_codes, 2):
        if code_a in coords_cache and code_b in coords_cache:
            dist = geodesic(coords_cache[code_a], coords_cache[code_b]).km
            dist_dict[(code_a, code_b)] = round(float(dist), 2)

    for code in coords_cache:
        dist_dict[(code, code)] = 0.0

    return dist_dict
