import re
import numpy as np
from timeit import timeit
from subprocess import run
from scipy.sparse import csr_matrix
from typing import Callable, Optional, Generic, TypeVar, Set
from enum import Enum
## my ##
from Edge import UEdge
from Node import Node
from Random import Rand
from MinHeap import MaxHeap

## do enumeration
class THEORYMODEL(Enum):
    WEAK = 0
    STRONG = 1

class E_PARTITION(Enum):
    E0 = 0
    E1 = 1
    E2 = 2

class L_PARTITION(Enum):
    L0 = 0
    L1 = 1
    Lge2 = 2

def edge_in(g: "Graph", e: UEdge) -> E_PARTITION:
    val = min(g.nodes[e.a].l,g.nodes[e.b].l)
    return E_PARTITION.E0 if val == 0 else E_PARTITION.E1 if val == 1 else E_PARTITION.E2


class L0Set:
    def __init__(self, g: "Graph") -> None:
        self.g = g
        self.nodes: Set[Node] = set()
    def add(self, node: Node) -> None:
        assert node.l == 1
        node.l = 0

        for nei in node.getneighs():
            if nei.l == 0: # in L0 already, they knew each other
                assert nei._neigh_L1 is not None
                nei._neigh_L1.remove(node) # expensive, costs O(n), but only improves from 26 to 25 seconds
                nei.addneigh_L0(node) #?# change L1-priority ? no, bc. nei is in L0
            elif nei.l == 1: # in L1, they did not know each other
                nei.addneigh_L0(node) #?# change L1-priority ? yes!
                self.g.L1.increase_key(nei)
                node.addneigh_L1(nei)
            else: # was in Lge2 -> L1, they did not know each other
                nei.l = 1
                nei.addneigh_L0(node) #?# change L1-priority ? no, because we add it now
                self.g.L1.add(nei)
                node.addneigh_L1(nei)
        self.nodes.add(node)

# class L1Set:
#     def __init__(self) -> None:
#         self.nodes: set[Node] = set()
#     def add(self, node: Node) -> None:
#         node.l = 1
#         self.nodes.add(node)
#     def remove(self, node: Node) -> None:
#         self.nodes.remove(node)
        
class L1Set:
    def __init__(self) -> None:
        self.nodes: MaxHeap = MaxHeap()
    def add(self, node: Node) -> None:
        node.l = 1
        self.nodes.insert(node, len(node.getneighs_L0()))
    def increase_key(self, node: Node) -> None:
        self.nodes.increase_key(node, len(node.getneighs_L0()))
    def extract_max(self) -> Node:
        return self.nodes.extract_max()


class Graph: 
    edges: list[UEdge]
    nodes: list[Node]
    csr: csr_matrix
    m: int
    n: int

    L0: L0Set
    L1: L1Set

    def __init__(self, filename: str, use_only: Callable[[UEdge], bool] = lambda x: True) -> None:
        """
        Read a graph from a file and return the graph
        """
        def match_edge_line(line:str) -> Optional[re.Match]:
            return re.match(r"\s*(?P<a>\d+)\s*(,| )\s*(?P<b>\d+)\s*", line)
        with open(filename,"r") as f:
            edges = list(map(match_edge_line, f))
            edges = [UEdge(*map(int, [line.group('a'), line.group('b')])) for line in edges if line
                                                                                            if not line.group('a') == line.group('b')] # filtered self-loops!
            edges = list(filter(use_only, edges))

            ## rename nodes to 0...n-1
            edgeset = set(e.a for e in edges) | set(e.b for e in edges)
            nodetoint = {node: i for i, node in enumerate(edgeset)}
            self.nodes = list(map(lambda item: Node(self, item[1], item[0]), nodetoint.items()))
            self.edges = list(map(lambda e: UEdge(nodetoint[e.a], nodetoint[e.b]), edges))
            self.m = len(self.edges)
            self.n = len(edgeset)

            # Convert edges to indices
            indices = [(e.a, e.b) for e in self.edges] + [(e.b, e.a) for e in self.edges]
            # Create data and row indices for csr_matrix
            data = np.ones(len(indices), dtype=int)
            rows, cols = zip(*indices)

            self.csr = csr_matrix((data, (rows, cols)), shape=(self.n, self.n))

    def generate_L0(self, v0: Optional[Node] = None, l0_percentage_size: float = 0.01, theoretical_model: THEORYMODEL = THEORYMODEL.WEAK) -> None:
        u = v0 if v0 is not None else Rand.choice(self.nodes)

        self.L0 = L0Set(self)
        self.L1 = L1Set()

        self.L1.add(u)

        for i in range(int(self.n * l0_percentage_size)):
            if len(self.L1.nodes) == 0: 
                print("L1 is empty, stopping") # should not happen
                print("L0: ", self.L0.nodes)
                for node in self.L0.nodes: print(node.nr, node.oldlabel, node.getneighs())
                break
            
            # u = max(self.L1.nodes, key=lambda v: (len(v.getneighs_L0()) if theoretical_model==THEORYMODEL.WEAK else len(v.getneighs()))) # get node from L1 with most neighbors in L0
            # self.L1.remove(u)
            u = self.L1.extract_max()

            print ("chosen node: ", u.nr, u.oldlabel, len(u.getneighs_L0()))
            self.L0.add(u) # sets l -> 0 and moves Lge2 neighbors to L1

    
    def getneighs(self, i: int) -> np.ndarray:
        return self.csr.indices[self.csr.indptr[i]:self.csr.indptr[i+1]]
    
    def plot(self) -> None:
        with open("graph.dot", "w") as dot_file:
            dot_file.write("graph G {\n")
            for n in self.nodes:
                dot_file.write(f'    {n.nr} [label="{n.oldlabel}", color="{n.color}", width=0.3, height=0.2, style=filled];\n')
            for e in self.edges:
                dot_file.write(f'    {e.a} -- {e.b} [color="{self.nodes[e.a].color if self.nodes[e.a].l<self.nodes[e.b].l else self.nodes[e.b].color}"];\n')
            dot_file.write("}\n")
        run(["fdp", "-Tpng", "graph.dot", "-o", "graph.png"], check=True)











def make_graph(filename: str, l0_size: float, seed: int = 42) -> Graph:
    Rand.seed(seed)
    g = Graph(filename)
    if l0_size>0: g.generate_L0(None, l0_size)
    print (g.L0.nodes)
    return g




if __name__ == "__main__":
    Rand.seed(42)

    filename1 = "testgraph/small.txt"
    filename2 = "soc-hamsterster/soc-hamsterster.edges"
    filename3 = "facebook_combined/facebook_combined.txt"
    filename4 = "git_web_ml/git_web_ml/musae_git_edges.csv"
    filename5 = "gemsec_deezer_dataset/deezer_clean_data/HR_edges.csv"
    filename6 = "gemsec_facebook_dataset/facebook_clean_data/artist_edges.csv"

    only_use1 = lambda e: True
    only_use2 = lambda e: e.inrange(0, 6)
    only_use3 = lambda e: e.inrange(10, 20)

    # g = Graph(filename6, only_use1)

    # generate = lambda size: g.generate_L0(l0_percentage_size=size)
    for filename in [filename2]:
        for i in range(1,2): 
            print(f"l0_size:{0.02*i}", timeit(lambda: make_graph(filename, 0.02*i), number=1),"s") 


    # print(g.n, g.m)
    # g.plot()
