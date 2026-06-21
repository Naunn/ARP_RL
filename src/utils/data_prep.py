from collections import defaultdict
from pathlib import Path

import pandas as pd

# Structural column definitions for both loading from raw text and downstream usage
DATA_COLUMNS = {
    "aircraft": {
        "load": [
            "aircraft_id",
            "type",
            "family",
            "capacities",
            "fixed_cost",
            "hourly_cost",
            "turnaround_dom",
            "turnaround_int",
            "initial_airport",
            "maintenance",
        ],
        "use": [
            "aircraft_id",
            "fixed_cost",
            "hourly_cost",
            "initial_airport",
            "seats",
            "speed",
        ],
    },
    "dist": {
        "load": ["origin", "destination", "nominal_time", "flight_type"],
        "use": ["origin", "destination", "nominal_time"],
    },
    "flights": {
        "load": [
            "flight_id",
            "origin",
            "destination",
            "departure",
            "arrival",
            "rotation_ref",
        ],
        "use": [
            "flight_id",
            "origin",
            "destination",
            "start_min",
            "arrival_min",
            "duration_min",
        ],
    },
}

# Fixed family-level average cruise speeds in km/h.
FAMILY_AVG_SPEED_KMH = {
    "Airbus": 840,
    "BAE": 750,
    "CRJ": 810,
    "ERJ": 830,
    "Fokker": 845,
    "TranspCom": 0,
}


# --- RESTORED YOUR EXACT ORIGINAL FUNCTION ---
def read_legacy_table(file_path: str | Path, expected_cols: list[str]) -> pd.DataFrame:
    """- Use sep=r'\\s+' to collapse variable whitespace layouts.

    - Use comment='#' to let pandas natively drop trailing metadata lines or
    comments.
    - Slice to expected length to strip away erratic spacing-induced empty
    columns.
    """
    df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:-1, :]
    df.columns = expected_cols
    return df


def parse_time_to_minutes(time_str: str) -> int:
    value = str(time_str).strip()
    day_offset = int(value.split("+")[1]) * 1440 if "+" in value else 0
    hh, mm = value.split("+")[0].split(":")
    return int(hh) * 60 + int(mm) + day_offset


def read_itineraries(file_path: str | Path) -> pd.DataFrame:
    revenue_by_flight = defaultdict(float)
    passenger_by_flight = defaultdict(int)

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.strip().startswith("#"):
                continue

            tokens = line.strip().split()
            ticket_price = float(tokens[2])
            passenger_count = int(tokens[3])
            leg_tokens = tokens[4:]
            leg_count = len(leg_tokens) // 3

            if leg_count == 0:
                continue

            allocated_revenue = (ticket_price * passenger_count) / leg_count
            for idx in range(0, len(leg_tokens), 3):
                flight_id = int(leg_tokens[idx])
                revenue_by_flight[flight_id] += allocated_revenue
                passenger_by_flight[flight_id] += passenger_count

    flight_ids = sorted(revenue_by_flight)
    return pd.DataFrame(
        {
            "flight_id": flight_ids,
            "total_ticket_price": [revenue_by_flight[fid] for fid in flight_ids],
            "total_passenger_count": [passenger_by_flight[fid] for fid in flight_ids],
        }
    )


def load_all_clean_data(
    flights_p, dist_p, itineraries_p, aircraft_p
) -> dict[str, pd.DataFrame]:
    flights = read_legacy_table(flights_p, DATA_COLUMNS["flights"]["load"])
    dist = read_legacy_table(dist_p, DATA_COLUMNS["dist"]["load"])
    aircraft = read_legacy_table(aircraft_p, DATA_COLUMNS["aircraft"]["load"])
    itineraries = read_itineraries(itineraries_p)

    flights["flight_id"] = flights["flight_id"].astype(int)
    flights["start_min"] = flights["departure"].map(parse_time_to_minutes)
    flights["arrival_min"] = flights["arrival"].map(parse_time_to_minutes)
    flights["duration_min"] = (flights["arrival_min"] - flights["start_min"]).clip(
        lower=0
    )

    aircraft["seats"] = (
        aircraft["capacities"]
        .astype(str)
        .apply(lambda x: sum(int(part) for part in x.split("/")))
    )
    aircraft["speed"] = aircraft["family"].map(FAMILY_AVG_SPEED_KMH)

    return {
        "flights": flights[DATA_COLUMNS["flights"]["use"]],
        "distances": dist[DATA_COLUMNS["dist"]["use"]],
        "aircraft": aircraft[aircraft["seats"] > 0][DATA_COLUMNS["aircraft"]["use"]],
        "itineraries": itineraries,
    }


def save_dataframes(data_dict: dict[str, pd.DataFrame], output_dir: str | Path):
    """Creates the output directory if it doesn't exist and saves all dataframes

    as CSV files without index columns.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for name, df in data_dict.items():
        file_path = out_path / f"{name}.csv"
        df.to_csv(file_path, index=False)
        print(f"Saved: {file_path}")


if __name__ == "__main__":
    DATA_DIR = Path("/home/bartosz/repos/ARP_RL/data/A1_6088570/A01_6088570")
    # Setting target save directory to data/training/ relative to your project base
    OUTPUT_DIR = Path("data/training/")

    dfs = load_all_clean_data(
        flights_p=DATA_DIR / "flights.csv",
        dist_p=DATA_DIR / "dist.csv",
        itineraries_p=DATA_DIR / "itineraries.csv",
        aircraft_p=DATA_DIR / "aircraft.csv",
    )

    # Save the processed tables
    save_dataframes(dfs, OUTPUT_DIR)
