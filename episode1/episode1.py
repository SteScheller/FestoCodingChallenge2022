import re
from dataclasses import dataclass
from typing import List, Dict, Tuple
from functools import reduce
from pathlib import Path

import numpy as np


@dataclass
class PopulationEntry:
    "Class for population.txt entries"
    username: str
    user_id: int
    home_planet: str
    blood_sample: str

    def __hash__(self):
        return hash(self.username + str(self.user_id) + self.home_planet + self.blood_sample)


def read_population(file_path: Path) -> List[PopulationEntry]:
    with open(file_path) as f:
        content = f.read()

    entries = list()
    pattern = (
        r"Name: (.+)\n"
        r"ID: (.+)\n"
        r"Home Planet: (.+)\n"
        r"Blood Sample:\s+\+--------\+\n([pico\s\|]+)\+--------\+"
    )
    for match in re.findall(pattern, content):
        entries.append(
            PopulationEntry(
                username=match[0],
                user_id=int(match[1]),
                home_planet=match[2],
                blood_sample=match[3],
            )
        )

    return entries


def read_lab(file_path: Path) -> List[PopulationEntry]:
    with open(file_path) as f:
        content = f.read()

    entries = list()
    pattern = r"\+--------\+\n([pico\s\|]+)\+--------\+"
    for match in re.findall(pattern, content):
        entries.append(
            PopulationEntry(
                username="John Doe", user_id=42, home_planet="Earth", blood_sample=match
            )
        )

    return entries


def filter_pico(entries: List[PopulationEntry]) -> List[PopulationEntry]:
    filtered = list()

    for item in entries:
        sample = item.blood_sample
        has_picobots = ("pico" in sample) or ("ocip" in sample)
        for i in range(8):
            if has_picobots:
                break
            pattern = (
                rf"\|{i*'.'}p{(7-i)*'.'}\|\s+"
                rf"\|{i*'.'}i{(7-i)*'.'}\|\s+"
                rf"\|{i*'.'}c{(7-i)*'.'}\|\s+"
                rf"\|{i*'.'}o{(7-i)*'.'}\|"
            )
            if re.search(pattern, sample) is not None:
                has_picobots = True
                break
            pattern = (
                rf"\|{i*'.'}o{(7-i)*'.'}\|\s+"
                rf"\|{i*'.'}c{(7-i)*'.'}\|\s+"
                rf"\|{i*'.'}i{(7-i)*'.'}\|\s+"
                rf"\|{i*'.'}p{(7-i)*'.'}\|"
            )
            has_picobots |= re.search(pattern, sample) is not None

        if has_picobots:
            filtered.append(item)

    return filtered


# ---------------------------------------------------------------------------------------------------
@dataclass
class GalaxyEntry:
    "Class for galaxy_map.txt entries"
    planet: str
    coordinates: np.ndarray

    def __hash__(self):
        return hash(self.planet + str(self.coordinates))


def read_galaxy(file_path: Path) -> List[GalaxyEntry]:
    with open(file_path) as f:
        content = f.read()

    entries = list()
    for match in re.findall(r"^(.+\b)\s*:\s*\((.+),(.+),(.+)\)$", content, re.MULTILINE):
        entries.append(
            GalaxyEntry(planet=match[0], coordinates=np.array([int(x) for x in match[1:]]))
        )

    return entries


def fit_galaxy_plane(galaxy: List[GalaxyEntry]) -> np.ndarray:
    X = np.pad(
        np.array([x.coordinates[:-1] for x in galaxy]),
        ((0, 0), (0, 1)),
        "constant",
        constant_values=1,
    )
    y = np.array([x.coordinates[-1] for x in galaxy])
    X_T = np.transpose(X)
    c = np.dot(np.dot(np.linalg.inv(np.dot(X_T, X)), X_T), y)

    return c


def filter_outlier_planets(
    galaxy: List[GalaxyEntry], plane_coeff: np.ndarray, min_distance: int
) -> List[GalaxyEntry]:
    filtered = list()

    for item in galaxy:
        distance = np.dot(np.append(item.coordinates[:-1], 1), plane_coeff) - item.coordinates[-1]
        if abs(distance) >= min_distance:
            filtered.append(item)

    return filtered


# ---------------------------------------------------------------------------------------------------
@dataclass
class SecurityEntries:
    "Class for security_log.txt entries"
    place: str
    events: Dict[str, Tuple[List[str], List[str]]]

    def __hash__(self):
        return hash(self.planet + str(self.coordinates))


def read_security_log(file_path: Path) -> List[SecurityEntries]:
    with open(file_path) as f:
        content = f.read()

    places = content.split("Place: ")[1:]
    entries = list()
    for item in places:
        lines = item.splitlines()
        events = dict()
        for match in re.findall(
            (r"(\d{2}:\d{2})\s+" r"(?:in:\s*([,\w ]+)\s+)?" r"(?:out:\s*([,\w ]+))?$"),
            "\n".join(lines[1:]),
            re.MULTILINE,
        ):
            in_list = [x.strip() for x in match[1].split(",") if match[1]]
            out_list = [x.strip() for x in match[2].split(",") if match[2]]
            events[match[0]] = in_list, out_list
        entries.append(SecurityEntries(place=lines[0].strip(), events=events))
    return entries


@dataclass
class VisitEntry:
    time: str
    place: str
    goes_in: bool


def build_list_of_visited_places(
    people: List[PopulationEntry], log: List[SecurityEntries]
) -> Dict[str, List[VisitEntry]]:
    visits = dict()

    def time_key(item: VisitEntry) -> int:
        return int("".join(item.time.split(":")))

    for p in people:
        name = p.username
        visited = list()
        for entry in log:
            for time, (in_list, out_list) in entry.events.items():
                if name in in_list:
                    visited.append(VisitEntry(time=time, place=entry.place, goes_in=True))
                if name in out_list:
                    visited.append(VisitEntry(time=time, place=entry.place, goes_in=False))
        visited.sort(key=time_key)
        visits[p] = visited

    return visits


def filter_visits(
    people_visits: Dict[PopulationEntry, List[VisitEntry]], sequence: List[str]
) -> List[PopulationEntry]:
    filtered = list()

    pattern = "|".join(sequence)
    for person, visits in people_visits.items():
        visits = visits[::2]
        if pattern in "|".join([x.place for x in visits]):
            filtered.append(person)

    return filtered


if __name__ == "__main__":
    base_path = Path(__file__).parent
    filtered_suspects = list()

    # puzzle 1
    filtered = filter_pico(read_lab(base_path / "lab_blood_clean.txt"))
    print(f"Bots found in clean samples: {len(filtered)}")
    filtered = filter_pico(read_lab(base_path / "lab_blood_gen1.txt"))
    print(f"Bots found in gen1 samples: {len(filtered)}")

    population = read_population(base_path / "population.txt")
    filtered = filter_pico(population)
    sum_ids = reduce(lambda x, y: x + y.user_id, filtered, 0)
    print(f"Solution for puzzle 1: {sum_ids}")
    filtered_suspects.append(filtered)

    # puzzle 2
    galaxy = read_galaxy(base_path / "galaxy_map.txt")
    plane = fit_galaxy_plane(galaxy)
    planets = [x.planet for x in filter_outlier_planets(galaxy, plane, 10)]
    people = [x for x in population if (x.home_planet in planets)]
    sum_ids = reduce(lambda x, y: x + y.user_id, people, 0)
    print(f"Solution for puzzle 2: {sum_ids}")
    filtered_suspects.append(people)

    # puzzle 3
    log = read_security_log(base_path / "security_log.txt")
    people_visits = build_list_of_visited_places(population, log)
    filtered = filter_visits(
        people_visits, ["Junkyard", "Pod Racing Track", "Pod Racing Track", "Palace", "Factory"]
    )
    sum_ids = reduce(lambda x, y: x + y.user_id, filtered, 0)
    print(f"Solution for puzzle 3: {sum_ids}")
    filtered_suspects.append(filtered)

    # list of suspects
    suspects = set(filtered_suspects[0])
    for suspect_list in filtered_suspects:
        suspects = suspects & set(suspect_list)
    print(f"Remaining set of suspects: {suspects}")
