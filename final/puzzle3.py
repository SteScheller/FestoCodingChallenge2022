import re
from typing import List
from pathlib import Path

import networkx as nx


def find_minimum_cut(path: Path) -> List[int]:
    with open(path, "r") as f:
        lines = f.readlines()

    wire_graph = nx.Graph()
    for line in lines:
        match = re.match(r"^(\d+)\s*:\s*(.)\s*\-\s(.):\s(\d+)$", line)
        if match:
            wire_graph.add_edge(match[2], match[3], index=int(match[1]), weight=int(match[4]))

    _, (p1, p2) = nx.minimum_cut(wire_graph, "A", "Z", capacity="weight")

    cutset = set()
    for u, neighbors in ((node, wire_graph[node]) for node in p1):
        cutset.update((u, v) for v in neighbors if v in p2)
    cuts = [wire_graph.edges[u, v]["index"] for (u, v) in cutset]

    return cuts


if __name__ == "__main__":
    base_path = Path(__file__).parent

    cuts = find_minimum_cut(base_path / "machine_room.txt")
    passage_ids = sorted([c for c in cuts])
    print(f"Solution for puzzle 3: {'-'.join([str(c) for c in sorted(cuts)])}")
