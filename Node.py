import numpy as np
from typing import Optional
# from __future__ import annotations
# import newstuff as G
# from newstuff import Graph
# try:
#     from newstuff import Graph
# except ImportError:
#     import sys
#     Graph = sys.modules[__package__ + '.Graph']


class Node:
    nr: int
    oldlabel: int
    l: int|None
    _neighs: Optional[list["Node"]] = None
    _neigh_L0: list["Node"]             # actually always stores the neighbors to L0
    # _neigh_L0_size: int = 0
    _neigh_L1: Optional[list["Node"]] = None # only exists if l == 0 or if it is memorized
    _neigh_Lge2: Optional[list["Node"]] = None # only exists if memorized

    def __init__(self, nr: int, oldlabel: int, l: int|None = None) -> None:
        self.nr = nr
        self.oldlabel = oldlabel
        self.l = l
        self._neigh_L0 = []

    def addneigh_L0(self, nei: "Node") -> None:
        assert self.l and self.l <= 1
        self._neigh_L0.append(nei)
        # self._neigh_L0_size += 1
    def addneigh_L1(self, nei: "Node") -> None: # only allowed to add if l == 0
        assert self.l == 0
        if self._neigh_L1 is None: self._neigh_L1 = []
        self._neigh_L1.append(nei)

    def getneighs(self, g) -> list["Node"]:
        if self._neighs is None:
            self._neighs = list(map(lambda nr: g.nodes[nr], (g.getneighs(self.nr))))
        return self._neighs
    
    def nr_neighs(self, g) -> int:
        if self._neighs: return len(self._neighs)
        else: return g.nrneighs(self.nr)
    
    def getneighs_L0(self) -> list["Node"]:
        return self._neigh_L0
    # @property
    # def neigh_L0_size(self) -> int:
    #     return self._neigh_L0_size
    
    def getneighs_L1(self, g) -> list["Node"]: ## ! only allowed to call after L0 created
        if self._neigh_L1 is None:
            self._neigh_L1 = list(filter(lambda n: n.l==1, map(lambda nr: g.nodes[nr], (g.getneighs(self.nr)))))
        return self._neigh_L1
    def getneighs_Lge2(self, g) -> list["Node"]: ## ! only allowed to call after L0 created
        if self._neigh_Lge2 is None:
            self._neigh_Lge2 = list(filter(lambda n: n.l>=2, map(lambda nr: g.nodes[nr], (g.getneighs(self.nr)))))
        return self._neigh_Lge2


    @property
    def color(self) -> str:
        return "red" if self.l == 0 else "blue" if self.l == 1 else "gray"
    
    # def __repr__(self) -> str:
    #     return f"{self.nr}({self.oldlabel})"

    def __repr__(self) -> str:
        return f"{self.oldlabel}"
    
    def __hash__(self) -> int:
        return self.nr
    
    def islayerLT(self, other: "Node") -> bool:
        if not self.l: return False
        if not other.l: return True
        return self.l < other.l