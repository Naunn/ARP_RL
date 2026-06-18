import copy
import random
from typing import Any, Dict, List, Optional


def _find_closest_city(target_city: str, cities: List[str], dist_dict: Dict):
    best = None
    best_d = float("inf")
    for c in cities:
        if c == target_city:
            continue
        d = dist_dict.get((target_city, c), dist_dict.get((c, target_city), None))
        if d is None:
            continue
        if d < best_d:
            best_d = d
            best = c
    return best or (cities[0] if cities else target_city)


class DisruptionGenerator:
    """Generate disruptions on schedules.

    Usage:
      dg = DisruptionGenerator(cities, dist_dict)
      new_schedule = dg.generate(schedule, actions=[{...}, ...])

    Supported actions (initial):
      - add_delay: add (or subtract) minutes to flight start times
        params: target=('random'|'selected'), count=int, min_delay, max_delay, allow_negative

      - replace_airport: replace flight origin or dest
        params: target=('random'|'selected'), field=('origin'|'dest'), method=('random'|'closest'), indices(optional list)
    """

    def __init__(self, cities: List[str], dist_dict: Dict):
        self.cities = list(cities)
        self.dist_dict = dist_dict or {}

    def add_delay(
        self,
        schedule: List[Dict[str, Any]],
        target: str = "random",
        count: int = 1,
        min_delay: int = 5,
        max_delay: int = 30,
        allow_negative: bool = False,
        indices: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        s = copy.deepcopy(schedule)
        n = len(s)
        if indices is None:
            if target == "random":
                indices = random.sample(range(n), min(count, n))
            else:
                indices = list(range(n))[:count]

        for idx in indices:
            shift = random.randint(min_delay, max_delay)
            if allow_negative and random.random() < 0.5:
                shift = -shift
            s[idx]["start"] = max(0, s[idx]["start"] + shift)
        s.sort(key=lambda x: x["start"])
        return s

    def replace_airport(
        self,
        schedule: List[Dict[str, Any]],
        target: str = "random",
        field: str = "origin",
        method: str = "random",
        indices: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        s = copy.deepcopy(schedule)
        n = len(s)
        if indices is None:
            if target == "random":
                indices = [random.randrange(n)]
            else:
                indices = [i for i in range(n)]

        for idx in indices:
            current = s[idx].get(field)
            other_field = "dest" if field == "origin" else "origin"
            other_city = s[idx].get(other_field)

            if method == "random":
                choices = [c for c in self.cities if c != current and c != other_city]
                if choices:
                    s[idx][field] = random.choice(choices)
            elif method == "closest":
                choices = [c for c in self.cities if c != current and c != other_city]
                if choices:
                    s[idx][field] = _find_closest_city(current, choices, self.dist_dict)  # type: ignore
            else:
                # treat method as a specific city name
                if method in self.cities and method != other_city:
                    s[idx][field] = method

        s.sort(key=lambda x: x["start"])
        return s

    def generate(
        self, schedule: List[Dict[str, Any]], actions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply a list of disruption actions sequentially and return the new schedule."""
        out = copy.deepcopy(schedule)
        for act in actions:
            name = act.get("action")
            if name == "add_delay":
                out = self.add_delay(
                    out,
                    target=act.get("target", "random"),
                    count=act.get("count", 1),
                    min_delay=act.get("min_delay", 5),
                    max_delay=act.get("max_delay", 30),
                    allow_negative=act.get("allow_negative", False),
                    indices=act.get("indices", None),
                )
            elif name == "replace_airport":
                out = self.replace_airport(
                    out,
                    target=act.get("target", "random"),
                    field=act.get("field", "origin"),
                    method=act.get("method", "random"),
                    indices=act.get("indices", None),
                )
            else:
                # unknown action: skip
                continue
        return out


def generate_disruptions(
    schedule: List[Dict[str, Any]],
    actions: List[Dict[str, Any]],
    cities: List[str],
    dist_dict: Dict,
) -> List[Dict[str, Any]]:
    dg = DisruptionGenerator(cities, dist_dict)
    return dg.generate(schedule, actions)
