from itertools import permutations

from geopy.distance import geodesic
from geopy.geocoders import Nominatim


def create_dist_dict(cities: list[str] = []):
    """
    Creates a dictionary of distances between pairs of cities using geopy.
    """
    if not cities:
        return {}

    geolocator = Nominatim(user_agent="city_distance_app")

    # Pre-fetch coordinates to avoid redundant API calls
    coords_cache = {}
    for city in cities:
        location = geolocator.geocode(city)
        if location:
            coords_cache[city] = (location.latitude, location.longitude)  # type: ignore
        else:
            print(f"Warning: Could not find coordinates for {city}")

    dist_dict = {}

    # Calculate distances for all unique pairs (A to B)
    # Using permutations(cities, 2) ensures we get (A, B) and (B, A)
    for city_a, city_b in permutations(cities, 2):
        if city_a in coords_cache and city_b in coords_cache:
            dist = geodesic(coords_cache[city_a], coords_cache[city_b]).km
            dist_dict[(city_a, city_b)] = round(dist, 2)

    # Add zero-distance for identity pairs (A to A)
    for city in cities:
        dist_dict[(city, city)] = 0.0

    return dist_dict
