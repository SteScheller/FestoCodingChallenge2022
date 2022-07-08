import math
from pathlib import Path
from typing import List, Tuple, Set, Dict
from functools import reduce
from itertools import product, combinations
import pickle

import numpy as np

from episode1.episode1 import PopulationEntry, read_population, read_lab
from episode1.episode1 import GalaxyEntry, read_galaxy
from episode1.episode1 import SecurityEntries, read_security_log
from episode1.episode1 import VisitEntry, build_list_of_visited_places


def filter_pico_gen2(entries: List[PopulationEntry]) -> List[PopulationEntry]:
    SEQUENCE = "picoico"

    def get_neighborhood(size: Tuple[int, int], row: int, col: int) -> Set[Tuple[int, int]]:
        """get coodinates of neighboring blood cells in horizontal and vertical direction"""
        coords = set()
        if row > 0:
            coords.add((row - 1, col))
        if row < (size[0] - 1):
            coords.add((row + 1, col))
        if col > 0:
            coords.add((row, col - 1))
        if col < (size[1] - 1):
            coords.add((row, col + 1))

        return coords

    def bend_sequence_dfs(
        sample: List[List[str]],
        size: Tuple[int, int],
        sequence: str,
        sequence_index: int,
        row: int,
        col: int,
    ) -> bool:
        """Depth first search for bend sequence of characters in two dimensional array"""
        if sequence_index == len(sequence):
            return True

        if sequence_index == 0:
            neighborhood = {(row, col)}
        else:
            neighborhood = get_neighborhood(size, row, col)
        for (n_row, n_col) in neighborhood:
            if sample[n_row][n_col] == sequence[sequence_index]:
                if bend_sequence_dfs(sample, size, sequence, sequence_index + 1, n_row, n_col):
                    return True

        return False

    filtered = list()
    for p in entries:
        sample = p.blood_sample.splitlines()
        size = (len(sample), len(sample[0]))
        for row, col in product(range(size[0]), range(size[1])):
            if bend_sequence_dfs(sample, size, SEQUENCE, 0, row, col):
                filtered.append(p)
                break

    return filtered


# --------------------------------------------------------------------------------------------------
def compute_min_distance_point_to_line_segment(
    point: np.ndarray, line: Tuple[np.ndarray, np.ndarray]
) -> float:
    vl0p = point - line[0]
    vl1p = point - line[1]
    vl0l1 = line[1] - line[0]
    dl0p = np.linalg.norm(vl0p)
    dl1p = np.linalg.norm(vl1p)

    # Test if the base point of the normal is on the trade route
    if (np.dot(vl0p, vl0l1) > 0) and (np.dot(vl1p, -vl0l1) > 0):
        dlp = math.sqrt(
            dl0p**2 - (np.dot(vl0p / dl0p, vl0l1 / np.linalg.norm(vl0l1)) * dl0p) ** 2
        )
    else:
        dlp = math.inf

    return min(dl0p, dl1p, dlp)


def filter_traderoutes(
    galaxy: List[GalaxyEntry],
    population: List[PopulationEntry],
    routes: Dict[Tuple[str, str], bool],
) -> List[PopulationEntry]:
    filtered = list()

    planets = dict()
    for entry in galaxy:
        planets[entry.planet] = entry.coordinates

    for person in population:
        person_routes = dict()
        planet_coords = planets[person.home_planet]
        for route in routes:
            line_segment = (planets[route[0]], planets[route[1]])
            person_routes[route] = 10 >= compute_min_distance_point_to_line_segment(
                planet_coords, line_segment
            )
        if person_routes == routes:
            filtered.append(person)

    return filtered


# --------------------------------------------------------------------------------------------------
def compute_times(visits: List[VisitEntry]) -> List[int]:
    times = list()

    def minute_difference(t_in: str, t_out: str) -> int:
        h_in, m_in = [int(x) for x in t_in.split(":")]
        h_out, m_out = [int(x) for x in t_out.split(":")]

        dh = 60 * (h_out - h_in)
        dm = m_out - m_in
        if dm < 0:
            dm += 60

        assert dh >= 0
        assert dm < 60
        assert dm >= 0

        return dh + dm

    for idx in range(0, len(visits), 2):
        assert visits[idx].goes_in is True
        assert visits[idx + 1].goes_in is False
        assert visits[idx + 1].place == visits[idx].place
        times.append(minute_difference(visits[idx].time, visits[idx + 1].time))

    return times


def filter_payment(population: List[PopulationEntry], log: List[SecurityEntries]):
    filtered = list()
    PAYMENT = 79

    def compute_possible_payments(times: List[int]) -> Set[int]:
        payments = [0]
        for r in range(1, len(times) + 1):
            payments += [sum(x) for x in combinations(times, r)]

        return set(payments)

    # people_visits = build_list_of_visited_places(population, log)
    # with open("visits.pkl", "wb") as f:
    #    pickle.dump(people_visits, f)

    with open("visits.pkl", "rb") as f:
        people_visits = pickle.load(f)

    for person in population:
        times = compute_times(people_visits[person])
        possible_payments = compute_possible_payments(times)
        if PAYMENT in set(possible_payments):
            print(set(possible_payments))
            filtered.append(person)

    return filtered


if __name__ == "__main__":
    base_path = Path(__file__).parent
    filtered_suspects = list()

    # puzzle 1
    filtered = filter_pico_gen2(read_lab("episode1/lab_blood_clean.txt"))
    print(f"Bots found in clean samples: {len(filtered)}")
    filtered = filter_pico_gen2(read_lab(base_path / "lab_blood_gen2.txt"))
    print(f"Bots found in gen2 samples: {len(filtered)}")

    population = read_population("episode1/population.txt")
    filtered = filter_pico_gen2(population)
    sum_ids = reduce(lambda x, y: x + y.user_id, filtered, 0)
    print(f"Solution for puzzle 1: {sum_ids}")
    filtered_suspects.append(filtered)

    # puzzle 2
    galaxy = read_galaxy("episode1/galaxy_map.txt")
    filtered = filter_traderoutes(
        galaxy,
        population,
        {
            ("Scurus V", "Tauries VII"): True,
            ("Saturus", "Beta Geminus"): False,
            ("Corpeia V", "Menta"): False,
            ("Grux", "Alpha Beron"): False,
            ("Gamma Veni", "Beta Earos"): False,
            ("Alpha Sexta", "Alpha Caprida"): False,
            ("Beta Drado VI", "Uranis"): True,
        },
    )
    sum_ids = reduce(lambda x, y: x + y.user_id, filtered, 0)
    print(f"Solution for puzzle 2: {sum_ids}")
    filtered_suspects.append(filtered)

    # puzzle 3
    security_log = read_security_log("episode1/security_log.txt")
    filtered = filter_payment(population, security_log)
    sum_ids = reduce(lambda x, y: x + y.user_id, filtered, 0)
    print(f"Solution for puzzle 3: {sum_ids}")
    filtered_suspects.append(filtered)

    # list of suspects
    suspects = set(filtered_suspects[0])
    for suspect_list in filtered_suspects:
        suspects = suspects & set(suspect_list)
    print(f"Remaining set of suspects: {suspects}")
