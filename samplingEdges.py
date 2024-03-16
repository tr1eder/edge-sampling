import snap, subprocess
import numpy as np
from typing import Set, Tuple, Union, Callable, Optional, NamedTuple
from enum import Enum
from collections import deque
## own imports ##
from Edge import Edge, UEdge, DEdge
from Random import Rand
import Utils


# # create a directed random graph on 100 nodes and 1k edges
# G2 = snap.GenRndGnm(snap.TNGraph, 100, 1000)
# # traverse the nodes
# for NI in G2.Nodes():
#     print("node id %d with out-degree %d and in-degree %d" % (
#         NI.GetId(), NI.GetOutDeg(), NI.GetInDeg()))
# # traverse the edges
# for EI in G2.Edges():
#     print("edge (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId()))
# # traverse the edges by nodes
# for NI in G2.Nodes():
#     for Id in NI.GetOutEdges():
#         print("edge (%d %d)" % (NI.GetId(), Id))

class PlotType(Enum):
    MY = 1
    GraphViz = 2
class VRSC(NamedTuple):
    vertex: int
    comp: Set[int]
    rs: float


class LoopHole():
    def __init__(self, create_graph_from_file: str, use_only_start: Optional[int], use_only_end: Optional[int]) -> None:
        if use_only_start is not None and use_only_end is not None and use_only_start <= use_only_end:
            func = lambda a, b: use_only_start <= a <= use_only_end and use_only_start <= b <= use_only_end
        else:
            func = lambda a, b: True
        self.create_graph_from_file(create_graph_from_file, func)

    def create_graph_from_file(self, filename: str, use_only: Callable[[int, int], bool]) -> None:
        """
        create a graph from a file
        - does remove all self-loops
        """
        self.G = snap.TUNGraph.New() # type: ignore
        self.N = 0
        self.M = 0
        with open(filename) as f:
            for line in f:
                a, b = map(int, line.split())
                if not use_only(a,b): continue
                if a == b: continue
                # add nodes 
                for node in range(self.N, max(a,b)+1): 
                    if use_only(node,node): self.G.AddNode(node)
                if self.N < max(a,b) + 1: self.N = max(a,b) + 1
                # add edge
                self.G.AddEdge(a, b)
                self.M += 1
    
    def generate_L0(self, v0: int = -1, l0_percentage_size: float = 0.01) -> None:
        u = v0 if v0 != -1 else Rand.randint(0, self.N)

        self.L0: Set[int] = {u}
        self.L1: Set[int] = self.neighsOf(u)
        E00: Adj = AdjUndirected()
        E01: Adj = AdjBipartite()
        for v in self.L1:
            E01.add(u, v)

        for i in range(int(self.N*l0_percentage_size)-1):
            u: int = E01.getMaxDegreeNode(E01.adjB)

            # move u from L1 to L1
            self.L1.remove(u)
            self.L0.add(u)

            # remove edges from L1 side
            neighsInL0 = E01.removeRight(u) 
            # add edges back to L0 side
            for v in neighsInL0: E00.add(u, v)

            # add newly discovered nodes to L1 and E01
            for v in self.neighsOf(u).difference(neighsInL0):
                self.L1.add(v)
                E01.add(u, v)

        self.E00: EDGESAMPLER = E00.finalize()
        self.E01: EDGESAMPLER = E01.finalize()

    def estimate_lge2(self, s1, sge2):
        dPlusAvg = 0
        for queryL1 in range(s1):
            v = Rand.choice(list(self.L1)) # make list(set(...)) more efficient
            dPlus = len(self.neighsOf(v).difference(self.L0).difference(self.L1)) ## | neighbors of v in L2 |
            dPlusAvg += dPlus/s1 ## numerically less stable than dividing by s1 at the end
        
        dMinusAvg_ = 0
        trs = 0
        for queryL2 in range(sge2):
            v, _, rs = self.reach_Lge2()
            dMinus = len(self.neighsOf(v).intersection(self.L1)) ## | neighbors of v in L1 |
            dMinusAvg_ += dMinus/rs
            trs += 1/rs
        dMinusAvg = dMinusAvg_/trs
        
        return len(self.L1) * dPlusAvg / dMinusAvg
    
    def estimate_m1(self) -> None:
        self.m1bar = 314
        raise NotImplementedError
    
    def estimate_m2(self) -> None:
        self.m1bar = 314
        raise NotImplementedError
    
    def calculate_rs0(self) -> None:
        self.rs0 = 3.141592
        raise NotImplementedError
    
    def reach_Lge2(self) -> VRSC:
        def find_w() -> int:
            while True: # gets stuck if L2 empty
                edge = self.E01.sample()
                u = edge.a if edge.a in self.L1 else edge.b
                ws = self.neighsOf(u).difference(self.L0).difference(self.L1)
                if len(ws) > 0: return Rand.choice(list(ws))
        def bfs_on_Gge2(w) -> Set[int]:
            ret = set()
            active = deque([w])
            while len(active) > 0:
                v = active.popleft()
                if v in ret: continue
                ret.add(v)
                for u in self.neighsOf(v).difference(self.L1):
                    if in_L2(u) and in_L2(v): continue
                    active.append(u)
            return ret
        def in_L2(v) -> bool:
            return v not in self.L0 and v not in self.L1 and self.neighsOf(v).intersection(self.L1) != set()
        def comp_reachability(C: Set[int]) -> float:
            rsC = 0
            for v in C:
                # if not in_L2(v): continue # not needed as v that are in L>2 will not have neighbors in L1
                rsV = 0
                for u in self.neighsOf(v).intersection(self.L1): # u in N(v) cap L1
                    dMinus = len(self.neighsOf(u).intersection(self.L0))
                    dPlus  = len(self.neighsOf(v).intersection(self.L1))
                    rsU    = dMinus / dPlus
                    rsV    += rsU
                rsC += rsV
            return rsC / len(C)

        w = find_w()
        C = bfs_on_Gge2(w)
        v = Rand.choice(list(C))
        rs = comp_reachability(C)
        return VRSC(v, C, rs)
    






    def sample_edges(self, estimate_E: list[int] = [10000, 70000, 8234] , num_samples = 100) -> list[UEdge]:
        """
        Samples num_samples edges from the graph G
        """
        
        edges: list[UEdge] = []
        for i in range(num_samples):
            bucket = Utils.choose_from_bucket_with_probability(estimate_E)
            if bucket == 0:     # sample from E00
                edges.append(self.sample_edge_E0())
            elif bucket == 1:   # sample from E1
                edges.append(self.sample_edge_E1())
            else:               # sample from E2
                edges.append(self.sample_edge_E2())
        return edges
    
    def sample_edge_E0(self) -> UEdge:
        """
        Samples an edge from E0 = E00 U E01
        """
        chooseFrom = Utils.choose_from_bucket_with_probability([self.E00.size, self.E01.size])
        if chooseFrom == 0:
            return self.E00.sample()
        else:
            return self.E01.sample()
    
    def sample_edge_E1(self) -> UEdge:
        """
        Samples an edge from E1
        """
        def sample_once():
            edge: UEdge = self.E01.sample()
            u = edge.a if edge.a in self.L0 else edge.b
            v = edge.a if edge.a in self.L1 else edge.b
            w = Rand.choice(list(self.neighsOf(v).difference(self.L0)))
            return v, w
        
        v, w = sample_once()
        prob = len(self.neighsOf(v).difference(self.L0)) / len(self.neighsOf(v).intersection(self.L0))
        prob *= self.E01.size() / (2*self.m1bar)
        if Rand.random() < prob:
            return UEdge(v, w)
        else:
            return self.sample_edge_E1()

    
    def sample_edge_E2(self) -> UEdge: # ! WE ARE MISSING OUT ON EDGES THAT CONNECT TWO DISTINCT C COMPONENTS!!
        """
        Samples an edge from E2
        """
        def sample_rejection():
            _, C, rs = self.reach_Lge2()
            # prob being above 1 means we absolutely have to pick it because we come across it so rarely
            # that we are actually underestimating the chances of it being sampled, 
            # which is why these are defined as unreachable, i.e. we can't give the fairness guarantee.
            probability = min(self.rs0 / rs, 1) 
            if Rand.random() < probability:
                return C
            else:
                return sample_rejection()
        
        C = sample_rejection()
        edges = [UEdge(u, v) for u in C
                             for v in self.neighsOf(u).intersection(C)]
        return Rand.choice(edges)
        

    
    def plot(self, type: PlotType = PlotType.MY) -> None:
        if type == PlotType.MY:
            with open("graph.dot", "w") as dot_file:
                dot_file.write("graph G{\n")
                for NI in self.G.Nodes():
                    node_id = NI.GetId()
                    color = "red" if node_id in self.L0 else "blue" if node_id in self.L1 else "black"
                    dot_file.write(f'    {node_id} [label="{node_id}", color="{color}", width=0.3, height=0.2];\n')
                for EI in self.G.Edges():
                    from_id = EI.GetSrcNId()
                    to_id = EI.GetDstNId()
                    color = "red" if from_id in self.L0 or to_id in self.L0 else "blue" if from_id in self.L1 or to_id in self.L1 else "black"
                    dot_file.write(f'{EI.GetSrcNId()} -- {EI.GetDstNId()} [color="{color}"];\n')
                dot_file.write("}\n")

            subprocess.run(["fdp", "-Tpng", "graph.dot", "-o", "graph.png"], check=True)
        elif type == PlotType.GraphViz:
            snap.DrawGViz(self.G, snap.gvlDot, "graph.png", "graph") # type: ignore
        else:
            raise ValueError("Should not happen")
    
    ## helper functions ##
    def neighsOf(self, u: int) -> Set[int]:
        ret = set([v for v in self.G.GetNI(int(u)).GetOutEdges()]) # why does it crash without int()?
        return ret
    
    
class Adj:
    def add(self, a: int, b: int) -> None:
        raise NotImplementedError
    
    def remove(self, a: int) -> None:
        raise NotImplementedError
    
    def getMaxDegreeNode(self, adj: dict[int, Set[int]]) -> int:
        max_degree = -1
        max_degree_node = []
        for node, neighbors in adj.items():
            if len(neighbors) > max_degree:
                max_degree = len(neighbors)
                max_degree_node = [node]
            elif len(neighbors) == max_degree:
                max_degree_node.append(node)
        if max_degree == -1:
            raise ValueError("Should not happen")
        return Rand.choice(max_degree_node)
    
    def sample(self, node: int) -> UEdge:
        raise NotImplementedError
    
    def getAllEdges(self) -> Set[UEdge]:
        raise NotImplementedError
    
class EDGESAMPLER:
    def __init__(self, adj, size, prefix_sum, mapping) -> None:
        self.myadj: Adj = adj
        self.size = size # size != prefix_sum[-1] (only holds for AdjBipartite)
        self.prefix_sum = prefix_sum
        self.mapping = mapping

    def sample(self) -> UEdge:
        bucket = Utils.choose_from_bucket_with_prefix_probability(self.prefix_sum)
        node = self.mapping[bucket]
        return self.myadj.sample(node)

class AdjBipartite(Adj):
    def __init__(self) -> None:
        self.adjA: dict[int, Set[int]] = {}
        self.adjB: dict[int, Set[int]] = {}

    def finalize(self):
        size = 0
        prefix_sum = []
        mapping = []
        for node, neighbors in self.adjA.items():
            size += len(neighbors)
            prefix_sum.append(size)
            mapping.append(node)
        return EDGESAMPLER(self, size, prefix_sum, mapping)

    def add(self, a: int, b: int) -> None:
        if a not in self.adjA:
            self.adjA[a] = set()
        if b not in self.adjB:
            self.adjB[b] = set()
        self.adjA[a].add(b)
        self.adjB[b].add(a)

    def removeRight(self, a: int) -> Set[int]:
        neighs = self.adjB.pop(a)
        for n in neighs:
            self.adjA[n].remove(a)
        return neighs
    
    def sample(self, node: int) -> UEdge:
        """ assumes node is in adjA! """
        return UEdge(node, Rand.choice(list(self.adjA[node]))) # make more efficient by storing adjA as list
    
    def getAllEdges(self) -> Set[UEdge]:
        edges = set()
        for a, neighs in self.adjA.items():
            for b in neighs:
                edges.add(UEdge(a, b))
        return edges

class AdjUndirected(Adj):
    def __init__(self) -> None:
        self.adj: dict[int, Set[int]] = {}
    
    def finalize(self):
        size = 0
        prefix_sum = []
        mapping = []
        for node, neighbors in self.adj.items():
            size += len(neighbors)
            prefix_sum.append(size)
            mapping.append(node)
        return EDGESAMPLER(self, size//2, prefix_sum, mapping)

    def add(self, a: int, b: int) -> None:
        if a not in self.adj:
            self.adj[a] = set()
        if b not in self.adj:
            self.adj[b] = set()
        self.adj[a].add(b)
        self.adj[b].add(a)

    def __getitem__(self, key: int) -> Set[int]:
        return self.adj[key]
    
    def sample(self, node) -> UEdge:
        return UEdge(node, Rand.choice(list(self.adj[node])))
    
    def getAllEdges(self) -> Set[UEdge]:
        edges = set()
        for a, neighs in self.adj.items():
            for b in neighs:
                edges.add(UEdge(a, b))
        return edges
    
if __name__ == "__main__":
    Rand.seed(12345689)
    from_id = 0
    to_id = 100000
    loophole = LoopHole("facebook_combined/facebook_combined.txt", from_id, to_id)
    loophole.generate_L0(v0=from_id, l0_percentage_size=0.04)
    # loophole.sample_edges(estimate_E=[10000, 70000, 8234], num_samples=100)

    # loophole.plot()

    
    # for i in [2, 5, 10, 20, 50, 200, 1000]:
    #     for j in [2, 5, 10, 50, 200, 1000]:
    #         print (f"from (s1={i}, sge2={j}):", loophole.estimate_lge2(i, j))





"""
 - the sampling between E00 and E01 empirically seems to be uniform
"""