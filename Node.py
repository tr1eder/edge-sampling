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
    l: int = 42
    _neigh_L0: list["Node"]             # actually always stores the neighbors to L0
    _neigh_L1: Optional[list["Node"]] = None # only exists if l == 0 or if it is memorized
    _neigh_Lge2: Optional[list["Node"]] = None # only exists if memorized

    def __init__(self, g, nr: int, oldlabel: int) -> None:
        self._neigh_L0 = []
        self.g = g
        self.nr = nr
        self.oldlabel = oldlabel

    def addneigh_L0(self, nei: "Node") -> None:
        assert self.l <= 1
        self._neigh_L0.append(nei)
    def addneigh_L1(self, nei: "Node") -> None: # only allowed to add if l == 0
        assert self.l == 0
        if self._neigh_L1 is None: self._neigh_L1 = []
        self._neigh_L1.append(nei)

    def getneighs(self) -> list["Node"]:
        return list(map(lambda nr: self.g.nodes[nr], (self.g.getneighs(self.nr))))
    
    def getneighs_L0(self) -> list["Node"]:
        return self._neigh_L0
    def getneighs_L1(self) -> list["Node"]:
        if self._neigh_L1 is None:
            self._neigh_L1 = list(filter(lambda n: n.l==1, map(lambda nr: self.g.nodes[nr], (self.g.getneighs(self.nr)))))
        return self._neigh_L1
    def getneighs_Lge2(self) -> list["Node"]:
        if self._neigh_Lge2 is None:
            self._neigh_Lge2 = list(filter(lambda n: n.l>=2, map(lambda nr: self.g.nodes[nr], (self.g.getneighs(self.nr)))))
        return self._neigh_Lge2
    
    # def moveneigh(self, node: "Node", old_l: int) -> None: # the neighbor node is moved to old_l to node.l
    #     if node.l == 0:
    #         if old_l == 0: raise ValueError("old_l cannot be 0")
    #         elif old_l == 1: self.neigh_L1.remove(node) #! expensive
    #         elif old_l == 2: raise ValueError("old_l cannot be 2")
    #         node.neigh_L0.append(self)
    #     else:
    #         raise ValueError("not implemented")


    @property
    def color(self) -> str:
        return "red" if self.l == 0 else "blue" if self.l == 1 else "gray"
    
