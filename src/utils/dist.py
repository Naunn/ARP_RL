from itertools import permutations

import airportsdata
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


def create_dist_dict_from_airports(airports_list: list[str] = []):
    """Creates a dictionary of distances between pairs of airports using their IATA codes.

    Maintains the exact same input/output structure as the original city
    function.
    """
    if not airports_list:
        return {}

    # Load the airport database indexed by IATA code
    airports_db = airportsdata.load("IATA")

    # Pre-fetch coordinates from the local database to avoid API calls
    coords_cache = {}
    for code in airports_list:
        clean_code = code.upper().strip()

        if clean_code in airports_db:
            airport = airports_db[clean_code]
            coords_cache[code] = (airport["lat"], airport["lon"])
        else:
            print(f"Warning: Could not find coordinates for IATA code: {code}")

    dist_dict = {}

    # Calculate distances for all unique pairs (A to B and B to A)
    for code_a, code_b in permutations(airports_list, 2):
        if code_a in coords_cache and code_b in coords_cache:
            dist = geodesic(coords_cache[code_a], coords_cache[code_b]).km
            dist_dict[(code_a, code_b)] = round(dist, 2)

    # Add zero-distance for identity pairs (A to A)
    for code in airports_list:
        if code in coords_cache:
            dist_dict[(code, code)] = 0.0

    return dist_dict
