import re, pickle, time
import numpy as np
# from timeit import timeit
from subprocess import run
from scipy.sparse import csr_matrix
from typing import Callable, Optional, Set, NamedTuple, Tuple
from collections import deque
from functools import wraps
from enum import Enum
## my ##
from Edge import UEdge
from Node import Node
from Random import Rand
from MinHeap import MaxHeap

## CONSTANTS ##
filename1 = ("testgraph/small.txt", "test")
filename2 = ("soc-hamsterster/soc-hamsterster.edges", "soc-hamsterster")
filename3 = ("facebook_combined/facebook_combined.txt", "ego-facebook")
filename4 = ("git_web_ml/git_web_ml/musae_git_edges.csv", "musae-git")
filename5 = ("gemsec_deezer_dataset/deezer_clean_data/HR_edges.csv", "gemsec-deezer")
filename6 = ("gemsec_facebook_dataset/facebook_clean_data/artist_edges.csv", "gemsec-facebook")
filename7 = ("twitch_gamers/large_twitch_edges.csv", "twitch-gamers")
filename8 = ("as-skitter/as-skitter.txt", "as-skitter")

only_use1 = lambda e: True
only_use2 = lambda e: e.inrange(0, 6)
only_use3 = lambda e: e.inrange(10, 20)

## do enumeration
class THEORYMODEL(Enum):
    WEAK = "knows all neighbors"
    MEDIUM = "knows all neighbors and their degrees"
    STRONG = "knows all neighbors and knows highest-degree nodes"

class VCERS(NamedTuple):
    vertex: Node
    comp: Set[Node]
    edges: Set[UEdge]
    rs_n: float
    rs_e: float


class E_PARTITION(Enum):
    E00 = "00"
    E01 = "01"
    E11 = "11"
    E12 = "12"
    E22 = "22"          # ! not safe to use because l=2 is not automatically set
    # E23 = "23"
    Eg2 = "33"

    @property
    def is_E0(self) -> bool: return self in [E_PARTITION.E00, E_PARTITION.E01]
    
    @property
    def is_E1(self) -> bool: return self in [E_PARTITION.E11, E_PARTITION.E12]

    @property
    def is_E2(self) -> bool: return self in [E_PARTITION.E22, E_PARTITION.Eg2]

class L_PARTITION(Enum):
    L0 = 0
    L1 = 1
    Lge2 = 2

def edge_in(g: "LoopHole", e: UEdge) -> E_PARTITION:
    l_a = g.nodes[e.a].l
    l_b = g.nodes[e.b].l
    valMin = min(l_a if l_a else 42, l_b if l_b else 42)
    valMax = max(l_a if l_a else 42, l_b if l_b else 42)
    if valMin == 0: return E_PARTITION.E00 if valMax == 0 else E_PARTITION.E01
    elif valMin == 1: return E_PARTITION.E11 if valMax == 1 else E_PARTITION.E12
    return E_PARTITION.E22 if valMin == 2 else E_PARTITION.Eg2


class L0Set:
    def __init__(self) -> None:
        # self.g = g
        self.nodes: Set[Node] = set()
    def add(self, g: "LoopHole", node: Node) -> None:
        assert node.l == 1
        node.l = 0

        for nei in node.getneighs(g):
            if nei.l == 0: # in L0 already, they knew each other
                assert nei._neigh_L1 is not None
                nei._neigh_L1.remove(node) # expensive, costs O(n), but only improves from 26 to 25 seconds
                nei.addneigh_L0(node) # change L1-priority ? no, bc. nei is in L0
            elif nei.l == 1: # in L1, they did not know each other
                nei.addneigh_L0(node) # change L1-priority ? yes!
                g.L1.increase_key(nei)
                node.addneigh_L1(nei)
            else: # was in Lge2 -> L1, they did not know each other
                nei.l = 1
                nei.addneigh_L0(node) #change L1-priority ? no, because we add it now
                g.L1.add(nei)
                node.addneigh_L1(nei)
        self.nodes.add(node)

    @classmethod
    def deserialize(cls, l0nodes):
        l0 = cls()
        l0.nodes = set(l0nodes)
        return l0
        
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
    
class L1Set_Medium(L1Set):
    def __init__(self, g) -> None:
        super().__init__()
        self.g = g
    def add(self, node: Node) -> None:
        node.l = 1
        self.nodes.insert(node, self.g.degrees[node.nr])
    def increase_key(self, node: Node) -> None:
        return
    
class L1Set_Strong(L1Set):
    def __init__(self, g) -> None:
        super().__init__()
        self.g = g
        self.sorted_degrees_nodes = sorted(zip(g.degrees, g.nodes), key=lambda x: x[0], reverse=False)
    def extract_max(self) -> Node:
        node = self.sorted_degrees_nodes.pop()[1]
        node.l = 1
        self.add(node)
        return node
    


class LoopHole: 
    edges: list[UEdge]
    nodes: list[Node]
    degrees: list[int]
    csr: csr_matrix
    m: int
    n: int

    L0: L0Set
    L1: L1Set
    L1List: list[Node]

    def __init__(self, filename: str, use_only: Callable[[UEdge], bool] = lambda x: True) -> None:
        """
        Read a graph from a file and return the graph
        """
        def match_edge_line(line:str) -> Optional[re.Match]:
            return re.match(r"\s*(?P<a>\d+)\s*(,| |\t)\s*(?P<b>\d+)\s*", line)
        with open(filename,"r") as f:
            edges = list(map(match_edge_line, f))
            edges = [UEdge(*map(int, [line.group('a'), line.group('b')])) for line in edges if line
                                                                                            if not line.group('a') == line.group('b')] # filtered self-loops!
            edges = list(filter(use_only, edges))

            ## rename nodes to 0...n-1
            edgeset = set(e.a for e in edges) | set(e.b for e in edges)
            nodetoint = {node: i for i, node in enumerate(edgeset)}
            self.nodes = list(map(lambda item: Node(item[1], item[0]), nodetoint.items()))
            self.edges = list(map(lambda e: UEdge(nodetoint[e.a], nodetoint[e.b]), edges))
            self.m = len(self.edges)
            self.n = len(edgeset)
            # Convert edges to indices
            indices = [(e.a, e.b) for e in self.edges] + [(e.b, e.a) for e in self.edges]
            # Create data and row indices for csr_matrix
            data = np.ones(len(indices), dtype=int)
            rows, cols = zip(*indices)

            self.csr = csr_matrix((data, (rows, cols)), shape=(self.n, self.n))
            self.degrees = list(map(lambda nr: self.getneighs(nr).size, range(self.n)))

    def generate_L0(self, v0: Optional[Node] = None, l0_percentage_size: float = 0.01, theoretical_model: THEORYMODEL = THEORYMODEL.WEAK) -> None:
        u = v0 if v0 is not None else Rand.choice(self.nodes)

        self.L0 = L0Set()
        self.L1 = L1Set() if theoretical_model == THEORYMODEL.WEAK else L1Set_Medium(self) if theoretical_model == THEORYMODEL.MEDIUM else L1Set_Strong(self)

        self.L1.add(u)

        for i in range(int(self.n * l0_percentage_size)):
            if len(self.L1.nodes) == 0: self.exit("L1 is empty, stopping", self.L0.nodes)
            
            u = self.L1.extract_max()
            self.L0.add(self, u) # sets l -> 0 and moves Lge2 neighbors to L1


        ## finish up the setup ##
        self.L1List = self.L1.nodes.toList()


    def init_m0(self) -> None:
        ret = 0
        for n in self.L0.nodes:
            for neigh in n.getneighs(self):
                if neigh.l == 0: ret += 1/2
                else: ret += 1
        self.m0 = ret

    def init_m1bar(self, nroftests: int) -> None:
        avg = 0
        for _ in range(nroftests):
            v = Rand.choice(self.L1List)
            avg += v.nr_neighs(self) - len(v.getneighs_L0())/nroftests # neighs in L1 and L2

        self.m1bar = avg

    def init_m2bar(self, nroftests: int) -> None:
        avg = 0
        trs = 0
        for _ in range(nroftests):
            v, _, _, rs, _ = self.reach_Lge2()
            avg += len(v.getneighs_Lge2(self)) / rs
            trs += 1 / rs
        
        self.m2bar = avg / trs

    def reach_Lge2(self) -> VCERS:
        def find_w() -> Node:
            while True: # gets stuck if L2 empty
                v = Rand.choice(self.L1List) # ! why must we not choose u.a.r?
                ws = v.getneighs_Lge2(self)
                if len(ws) > 0: return Rand.choice(ws)
        def bfs_on_Gge2(w) -> Tuple[Set[Node], Set[UEdge]]:
            retC: Set[Node] = set()
            retE: Set[UEdge] = set()
            active = deque([w])
            while len(active) > 0:
                v = active.popleft()
                if v in retC: continue
                retC.add(v)
                for u in v.getneighs_Lge2(self):
                    if in_Leq2(u) and in_Leq2(v): 
                        if u.nr < v.nr: # do tiebreaking
                            retE.add(UEdge(u.nr, v.nr))
                        continue
                    retE.add(UEdge(u.nr, v.nr))
                    active.append(u)
            return retC, retE
        def in_Leq2(v: Node) -> bool:
            if not v.l: 
                v.l = 2 if len(v.getneighs_L1(self)) > 0 else 42
            return v.l == 2
        def comp_reachability(C: Set[Node], E: Set[UEdge]) -> Tuple[float, float]:
            rsC = 0
            for v in C:
                if not in_Leq2(v): continue
                rsV = 0
                for u in v.getneighs_L1(self):
                    dMinus = len(u.getneighs_L0())
                    dPlus  = len(u.getneighs_Lge2(self))
                    rsU    = dMinus / dPlus
                    rsV    += rsU
                rsC += rsV

            rs_n = 1/len(C) * rsC
            rs_e = 1/len(E) * rsC
            return rs_n, rs_e

        w = find_w()
        C, E = bfs_on_Gge2(w)
        v = Rand.choice(list(C))
        rs_n, rs_e = comp_reachability(C, E)
        return VCERS(v, C, E, rs_n, rs_e)





    def getneighs(self, i: int) -> np.ndarray:
        return self.csr.indices[self.csr.indptr[i]:self.csr.indptr[i+1]]
    
    def nrneighs(self, i: int) -> int:
        return self.csr.indptr[i+1] - self.csr.indptr[i]
    
    def plot(self) -> None:
        with open("graph.dot", "w") as dot_file:
            dot_file.write("graph G {\n")
            for n in self.nodes:
                dot_file.write(f'    {n.nr} [label="{n.oldlabel}", color="{n.color}", width=0.3, height=0.2, style=filled];\n')
            for e in self.edges:
                dot_file.write(f'    {e.a} -- {e.b} [color="{self.nodes[e.a].color if self.nodes[e.a].islayerLT(self.nodes[e.b]) else self.nodes[e.b].color}"];\n')
            dot_file.write("}\n")
        run(["fdp", "-Tpng", "graph.dot", "-o", "graph.png"], check=True)

    def export(self, filename: str) -> None:
        with open(filename, "wb") as f:
            # json.dump(self.__dict__, f) #, default=lambda o: o.__dict__)
            # sys.setrecursionlimit(1000000)
            pickle.dump(self.serialize(), f)

    @staticmethod
    def import_graph(filename: str) -> "LoopHole":
        # raise FileNotFoundError("Not implemented")
        with open(filename, "rb") as f:
            # return json.load(f) #, object_hook=lambda d: LoopHole.__dict__.update(d) or LoopHole(d["filename"], only_use1))
            return LoopHole.deserialize(pickle.load(f))
        
    def exit(self, *args) -> None:
        print(*args)
        raise RuntimeError("Exiting", *args)
    
    def serialize(self):
        ser = {
            "edges": [(e.a, e.b) for e in self.edges],
            "nodes": [(n.nr, n.oldlabel, n.l) for n in self.nodes],
            "csr_shape": self.csr.shape,
            "csr_data": self.csr.data.tolist(),
            "csr_indices": self.csr.indices.tolist(),
            "csr_indptr": self.csr.indptr.tolist(),
            "m": self.m,
            "n": self.n,
            "L0": [n.nr for n in self.L0.nodes],
            "L1": [n.nr for n in self.L1.nodes.toList()]
        }
        return ser
    
    @classmethod
    def deserialize(cls, ser): 
        g = cls.__new__(cls)
        g.edges = [UEdge(*e) for e in ser["edges"]]
        g.nodes = [Node(*n) for n in ser["nodes"]]
        g.csr = csr_matrix((ser["csr_data"], ser["csr_indices"], ser["csr_indptr"]), shape=ser["csr_shape"])
        g.m = ser["m"]
        g.n = ser["n"]
        g.L0 = L0Set.deserialize(map(lambda n: g.nodes[n], ser["L0"]))
        # set _neighs_L0
        for n in g.L0.nodes:
            for neigh in n.getneighs(g):
                if neigh.l == 0: n.addneigh_L0(neigh)
                elif neigh.l == 1: n.addneigh_L1(neigh)
                else: raise ValueError("neigh.l >= 2")

                neigh.addneigh_L0(n)
            

        g.L1 = L1Set()
        for n in ser["L1"]: g.L1.add(g.nodes[n])
        return g



def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"--> {func.__name__} took {end-start} seconds")
        return result
    return wrapper



@timed
def LOOPHOLE_FACTORY(filename: str, name: str, l0_size: float, seed: int = 42) -> LoopHole:
    try:
        # raise FileNotFoundError
        g = _load_graph(name, l0_size, seed)
        print ("--- Graph loaded ---")
        return g
    except FileNotFoundError:
        print ("--- Graph not yet created, creating it now ---")
        return _store_graph(filename, name, l0_size, seed)

def _make_graph(filename: str, name: str, l0_size: float, seed: int = 42, v0 = None) -> LoopHole:
    Rand.seed(seed)
    g = LoopHole(filename)
    if l0_size>0: g.generate_L0(v0, l0_size)
    return g

def _store_graph(filename: str, name: str, l0_size: float, seed: int = 42) -> LoopHole:
    g = _make_graph(filename, name, l0_size, seed)
    g.export("bin/"+name+str(l0_size)+"_"+str(seed)+".bin")
    return g

def _load_graph(name: str, l0_size: float, seed: int = 42) -> LoopHole:
    return LoopHole.import_graph("bin/"+name+str(l0_size)+"_"+str(seed)+".bin")




if __name__ == "__main__":
    Rand.seed(42)


    # for filename, name in [filename6]:
    #     for i in range(0,6): 
    #         print(f"l0_size:{0.02*i}", timeit(lambda: make_graph(filename, 0.02*i), number=1),"s") 

    g = LOOPHOLE_FACTORY(filename8[0], filename8[1], 0)
    print(g.n, g.m)
    # g.plot()
