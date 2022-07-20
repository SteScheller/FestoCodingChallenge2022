from pathlib import Path
from typing import Tuple

import networkx as nx


def find_max_booty(path: Path) -> Tuple[str, int]:
    with open(path, "r") as f:
        content = f.read()

    booty_texts = content.split("\n\n")

    max_booty_planet, max_booty = "Magrathea", 0
    for booty_text in booty_texts:
        lines = booty_text.splitlines()
        if len(lines) < 3:
            continue
        planet = lines[0].strip()
        upper, lower = lines[1:]
        upper = [int(x.strip()) for x in upper.split(",")]
        lower = [int(x.strip()) for x in lower.split(",")]
        booty_graph = nx.DiGraph()
        for i in range(len(upper) - 1):
            if i == 0:
                booty_graph.add_edge((-1, -1), (0, 0), weight=upper[0])
                booty_graph.add_edge((-1, -1), (1, 0), weight=lower[0])
            booty_graph.add_edge((0, i), (0, i + 1), weight=0)
            booty_graph.add_edge((0, i), (1, i + 1), weight=lower[i + 1])
            booty_graph.add_edge((1, i), (0, i + 1), weight=upper[i + 1])
            booty_graph.add_edge((1, i), (1, i + 1), weight=0)
        booty = nx.dag_longest_path_length(booty_graph)
        if booty > max_booty:
            max_booty_planet = planet
            max_booty = booty

    return max_booty_planet, max_booty


if __name__ == "__main__":
    base_path = Path(__file__).parent

    planet, booty = find_max_booty(base_path / "bunker_gold.txt")
    print(f"Solution for puzzle 2: {planet}{booty}")
