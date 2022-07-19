from copy import copy
from typing import List, Set, Tuple, Iterable, Dict, FrozenSet
from pathlib import Path
from functools import reduce
from itertools import product, combinations

import numpy as np
import networkx as nx

from episode1.episode1 import PopulationEntry, read_population, read_lab
from episode1.episode1 import GalaxyEntry, read_galaxy


def filter_pico_gen3(entries: List[PopulationEntry]) -> List[PopulationEntry]:
    SEQUENCES = ("pic", "opi", "cop", "ico")

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

    def find_single_sequence(
        sample: List[List[str]],
        size: Tuple[int, int],
        used: Set[Tuple[int, int]],
        path: List[Tuple[int, int]],
        solutions: List[List[Tuple[int, int]]],
        sequence: str,
        index: int,
        row: int,
        col: int,
    ) -> None:
        """Depth first search for bend sequence of characters in two dimensional array"""
        if index == len(sequence):
            solutions.append(copy(path))
            return

        if index == 0:
            search_space = {(row, col)}
        else:
            search_space = get_neighborhood(size, row, col)
        for (n_row, n_col) in search_space:
            if (n_row, n_col) in used | set(path):
                continue
            if sample[n_row][n_col] == sequence[index]:
                path.append((n_row, n_col))
                find_single_sequence(
                    sample, size, used, path, solutions, sequence, index + 1, n_row, n_col
                )
                path.pop()

    def find_all_sequences(
        sample: List[List[str]],
        size: Tuple[int, int],
        used: Set[Tuple[int, int]],
        sequences: Iterable[str],
        index: int,
    ) -> bool:
        """Depth first search for multiple bend sequences of characters in two dimensional array
        without overlap"""
        if index == len(sequences):
            return True

        used_old = copy(used)
        for (n_row, n_col) in product(range(size[0]), range(size[1])):
            if (n_row, n_col) in used:
                continue
            path, solutions = list(), list()
            find_single_sequence(
                sample, size, used_old, path, solutions, sequences[index], 0, n_row, n_col
            )
            for solution in solutions:
                used = used_old | set(solution)
                if find_all_sequences(sample, size, used, sequences, index + 1):
                    return True
                else:
                    used = used_old
        return False

    filtered = list()
    for p in entries:
        sample = p.blood_sample.splitlines()
        size = (len(sample), len(sample[0]))
        used = set()
        if find_all_sequences(sample, size, used, SEQUENCES, 0):
            filtered.append(p)

    return filtered


# --------------------------------------------------------------------------------------------------
def read_signal_examples(path: Path) -> Dict[FrozenSet[str], int]:
    with open(path, "r") as f:
        lines = f.readlines()

    distances = dict()
    for entry in lines:
        planets, distance = entry.split(":")
        planets = frozenset([x.strip() for x in planets.split("-")])
        distances[planets] = int(distance.strip())

    return distances


def compute_delay_graph(galaxy: List[GalaxyEntry]) -> nx.Graph:
    MAX_DISTANCE = 50
    graph = nx.Graph()
    for planet in galaxy:
        graph.add_node(planet.planet, coordinates=planet.coordinates)

    for p1, p2 in combinations(galaxy, 2):
        if np.linalg.norm(p1.coordinates - p2.coordinates) <= MAX_DISTANCE:
            graph.add_edge(p1.planet, p2.planet, delay=1)

    return graph


def test_delay_graph(delay_graph: nx.Graph, example_distances: Dict[FrozenSet[str], int]) -> bool:
    result = True
    for planets, distance in example_distances.items():
        p1, p2 = planets
        computed_distance = nx.shortest_path_length(delay_graph, p1, p2)
        if distance != computed_distance:
            result = False
            break

    return result


def filter_signal_delay(
    population: List[PopulationEntry], delay_graph: nx.Graph, delays: Dict[str, int]
) -> List[PopulationEntry]:
    filtered = list()
    return filtered


# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    base_path = Path(__file__).parent
    filtered_suspects = list()

    # puzzle 1
    filtered = filter_pico_gen3(read_lab("episode1/lab_blood_clean.txt"))
    print(f"Bots found in clean samples: {len(filtered)}")
    filtered = filter_pico_gen3(read_lab(base_path / "lab_blood_gen3.txt"))
    print(f"Bots found in gen3 samples: {len(filtered)}")

    population = read_population("episode1/population.txt")
    # filtered = filter_pico_gen3(population)
    sum_ids = reduce(lambda x, y: x + y.user_id, filtered, 0)
    print(f"Solution for puzzle 1: {sum_ids}")
    filtered_suspects.append(filtered)

    # puzzle 2
    galaxy = read_galaxy("episode1/galaxy_map.txt")
    examples = read_signal_examples(base_path / "signal_examples.txt")
    delay_graph = compute_delay_graph(galaxy)
    test_result = "PASS" if test_delay_graph(delay_graph, examples) else "FAIL"
    print(f"Test of delay graph: {test_result}")
    filtered = filter_signal_delay(population, delay_graph, {"Venis": 2, "Cetung": 4, "Phoensa": 9})
    sum_ids = reduce(lambda x, y: x + y.user_id, filtered, 0)
    print(f"Solution for puzzle 2: {sum_ids}")
    filtered_suspects.append(filtered)

    # puzzle 3

    # list of suspects
    suspects = set(filtered_suspects[0])
    for suspect_list in filtered_suspects:
        suspects = suspects & set(suspect_list)
    print(f"Remaining set of suspects: {suspects}")
