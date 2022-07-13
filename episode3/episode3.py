import copy
from typing import List, Set, Tuple, Iterable
from pathlib import Path
from functools import reduce
from itertools import product

from episode1.episode1 import PopulationEntry, read_population, read_lab


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

    def bend_sequence_dfs(
        sample: List[List[str]],
        size: Tuple[int, int],
        used: Set[Tuple[int, int]],
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
            if (n_row, n_col) in used:
                continue
            if sample[n_row][n_col] == sequence[sequence_index]:
                used.add((n_row, n_col))
                if bend_sequence_dfs(
                    sample, size, used, sequence, sequence_index + 1, n_row, n_col
                ):
                    return True
                else:
                    used.remove((n_row, n_col))

        return False

    def outer_dfs(
        sample: List[List[str]],
        size: Tuple[int, int],
        used: Set[Tuple[int, int]],
        sequences: Iterable[str],
        sequences_index: int,
    ) -> bool:
        """Depth first search for sequence of bend sequences of characters in two
        dimensional array without overlap"""
        if sequences_index == len(sequences):
            return True

        for (n_row, n_col) in product(range(size[0]), range(size[1])):
            if (n_row, n_col) in used:
                continue
            used_old = copy.deepcopy(used)
            if bend_sequence_dfs(sample, size, used, sequences[sequences_index], 0, n_row, n_col):
                if outer_dfs(sample, size, used, sequences, sequences_index + 1):
                    return True
            used = used_old
        return False

    filtered = list()
    for idx, p in enumerate(entries):
        sample = p.blood_sample.splitlines()
        size = (len(sample), len(sample[0]))
        used = set()
        if outer_dfs(sample, size, used, SEQUENCES, 0):
            filtered.append(p)

    return filtered


# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    base_path = Path(__file__).parent
    filtered_suspects = list()

    # puzzle 1
    # filtered = filter_pico_gen3(read_lab("episode1/lab_blood_clean.txt"))
    # print(f"Bots found in clean samples: {len(filtered)}")
    filtered = filter_pico_gen3(read_lab(base_path / "lab_blood_gen3.txt"))
    print(f"Bots found in gen3 samples: {len(filtered)}")

    population = read_population("episode1/population.txt")
    # filtered = filter_pico_gen3(population)
    sum_ids = reduce(lambda x, y: x + y.user_id, filtered, 0)
    print(f"Solution for puzzle 1: {sum_ids}")
    filtered_suspects.append(filtered)

    # puzzle 2

    # puzzle 3

    # list of suspects
    suspects = set(filtered_suspects[0])
    for suspect_list in filtered_suspects:
        suspects = suspects & set(suspect_list)
    print(f"Remaining set of suspects: {suspects}")
