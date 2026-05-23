CITIES = ["praga", "milan"]  # , "lodz", "paris", "madryt", "berlin", "barcelona"]
N_FLIGHTS = 5  # 0
FIRST_FLIGHT_HOUR = 5
LAST_FLIGHT_HOUR = 23
MIN_PASS = 100
MAX_PASS = 180

PLANES_TEMPLATES = {
    "BOEING": {
        "fuel_use": 900.0,
        "seats": 150,
        "speed": 900,
        "base_fare": 50,
        "rate_per_km": 0.15,
    },
    "AIRBUS": {
        "fuel_use": 850.0,
        "seats": 120,
        "speed": 850,
        "base_fare": 120,
        "rate_per_km": 0.28,
    },
}
