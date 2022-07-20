from typing import List
from pathlib import Path
from dataclasses import dataclass

import networkx as nx


@dataclass
class Passage:
    idx: int
    scrap_graph: nx.DiGraph


def find_clearable_passages(path: Path) -> List[Passage]:
    with open(path, "r") as f:
        content = f.read()

    passages = list()
    passage_texts = content.split("Passage")[1:]
    for text in passage_texts:
        lines = text.splitlines()
        idx = int(lines[0].strip())
        scrap_graph = nx.DiGraph()
        for line in lines[1:]:
            if not line.strip():
                continue
            item_node, blocker_nodes = line.split(":")
            item_node = item_node.strip()
            for blocker_node in blocker_nodes.split(","):
                scrap_graph.add_edge(blocker_node.strip(), item_node)
        passages.append(Passage(idx, scrap_graph))

    return [p for p in passages if nx.is_directed_acyclic_graph(p.scrap_graph)]


if __name__ == "__main__":
    base_path = Path(__file__).parent

    passages = find_clearable_passages(base_path / "scrap_scan.txt")
    passage_ids = sorted([p.idx for p in passages])
    print(f"Solution for puzzle 1: {'-'.join([str(pid) for pid in passage_ids])}")
