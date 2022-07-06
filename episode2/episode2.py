from pathlib import Path
from typing import List
from functools import reduce

from episode1.episode1 import PopulationEntry, read_population, read_lab


def filter_pico_gen2(entries: List[PopulationEntry]) -> List[PopulationEntry]:
    filtered = list()

    for item in entries:
        sample = item.blood_sample
        has_picobots = ("picoico" in sample) or ("ociocip" in sample)
        """
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
        """

        if has_picobots:
            filtered.append(item)

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

    # puzzle 3

    # list of suspects
    suspects = set(filtered_suspects[0])
    for suspect_list in filtered_suspects:
        suspects = suspects & set(suspect_list)
    print(f"Remaining set of suspects: {suspects}")
