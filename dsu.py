"""Union-Find / Disjoint Set Union implementation with path compression and union-by-rank."""
from typing import Dict, Any


class DSU:
    def __init__(self):
        # parent and rank are dicts keyed by element (int)
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}

    def make_set(self, x: int) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: int) -> int:
        # lazy make_set
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        # path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return False
        # union by rank
        if self.rank.get(xroot, 0) < self.rank.get(yroot, 0):
            self.parent[xroot] = yroot
        elif self.rank.get(xroot, 0) > self.rank.get(yroot, 0):
            self.parent[yroot] = xroot
        else:
            self.parent[yroot] = xroot
            self.rank[xroot] = self.rank.get(xroot, 0) + 1
        return True

    def components(self) -> Dict[int, list]:
        """Return mapping root -> [members] for known elements."""
        comp: Dict[int, list] = {}
        for v in list(self.parent.keys()):
            r = self.find(v)
            comp.setdefault(r, []).append(v)
        return comp

    def num_components(self) -> int:
        return len(self.components())
