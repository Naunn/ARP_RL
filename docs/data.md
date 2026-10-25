# ✈️ Airline Disruption Recovery & Schedule Data Schema

This document provides a clean, comprehensive reference guide for the datasets used in aircraft schedule recovery and passenger disruption management.

---

## 🕒 1. Schedule Data (`schedule.csv`)
Represents the baseline flight legs scheduled across the network, including operational sequences and continuity dependencies.

| Index | Column Name | Type | Example | Description |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `flight_id` | Integer | `2598` | Unique tracking identification code for the flight leg. |
| **2** | `origin` | String (IATA) | `URO` | Departure Airport (3-letter IATA code). |
| **3** | `destination` | String (IATA) | `LYS` | Arrival Airport (3-letter IATA code). |
| **4** | `departure` | Time (HH:MM) | `05:40` | Scheduled time of departure. |
| **5** | `arrival` | Time (HH:MM) | `07:00` or `00:10+1` | Scheduled time of arrival. A `+1` indicates it arrives the next day. |
| **6** | `predecessor_id` | Integer | `2597` or `0` | **Flight Continuity Link:** If `0`, it is independent. If it points to another `flight_id`, it means this flight cannot depart until that predecessor flight lands, forming a mandatory through-flight or aircraft/crew rotation locking constraint. |

---

## 🛩️ 2. Aircraft Data (`aircraft.csv`)
Defines the individual aircraft assets, their seat capacities, cost metrics, and strict operational constraints.

| Index | Column Name | Type | Example | Description |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `aircraft_id` | String | `A319#15` | Unique registration / tail code. |
| **2** | `type` | String | `A319` | Machine variant. |
| **3** | `family` | String | `Airbus` | Group family for allowed swaps. |
| **4** | `capacities` | String | `0/0/138` | Class limits structured as: `First/Business/Economy`. |
| **5** | `fixed_cost` | Integer | `510` | Fixed operational cost metric. |
| **6** | `hourly_cost` | Float | `1850.0` | Cost multiplier per airborne hour. |
| **7** | `turnaround_domestic` | Integer (Mins) | `35` | Ground buffer time required for domestic legs. |
| **8** | `turnaround_intl` | Integer (Mins) | `35` | Ground buffer time required for international legs. |
| **9** | `initial_airport` | String (IATA) | `MPL` | The physical airport where this plane starts the recovery window. |
| **10** | `maintenance` | String | `CDG-...-120` or `NULL` | Scheduled maintenance lock layout: `[Airport]-[StartDate]-[StartTime]-[EndDate]-[EndTime]-[Cost]` |

---

## 🗺️ 3. Flight Distance & Type Data (`fist.csv`)
Contains baseline route properties and geographic categories between airport pairs.

| Index | Column Name | Type | Example | Description |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `origin` | String (IATA) | `AJA` | Departure Airport (3-letter IATA code, e.g., Ajaccio). |
| **2** | `destination` | String (IATA) | `AMS` | Arrival Airport (3-letter IATA code, e.g., Amsterdam). |
| **3** | `nominal_time` | Integer (Mins) | `120` | Baseline flight duration in minutes. |
| **4** | `flight_type` | Character | `C` | **Geographic Category:**<br>• `D` = Domestic<br>• `C` = Continental<br>• `I` = Intercontinental |

---

## 👥 4. Passenger Itinerary Data (`itinerary.csv`)
Tracks passenger bookings, classes, journey directions, and connecting leg sequences used for delay and disruption calculations.

| Column | Column Name | Type | Example | Description |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `Itinerary ID` | Integer | `0` | A unique identifier for this passenger group. |
| **2** | `Cabin Class` | Character | `E` | The service class the group booked:<br>• `F` = First Class<br>• `B` = Business<br>• `E` = Economy<br>*(Crucial because swapping aircraft families changes the available cabin capacities)* |
| **3** | `Trip Type` | Character | `O` | Distinguishes whether this leg is part of an outbound or inbound journey:<br>• `O` = Outbound<br>• `I` = Inbound<br>*(Used to calculate passenger delay penalties; delaying a traveler on their outward journey often carries a different cost weight than a return trip)* |
| **4** | `Number of Passengers` | Integer | `45` | The total volume of individual people bundled into this specific itinerary. |
| **5** | `Number of Flights (N)` | Integer | `2` | The total number of legs/connections in this specific trip (e.g., `1` for direct, `2` or more for connecting itineraries). |
| **6+** | `Flight IDs` | List of Integers | `2598, 2600` | A sequential list of `N` flight numbers that make up the complete journey. For example, if Column 5 is `2`, Columns 6 and 7 will list the two specific Flight ID numbers they must board in order. |