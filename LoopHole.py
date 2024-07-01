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
import Utils

class HighestKDegreeNodes:
    def __init__(self, g: "LoopHole", k: int) -> None:
        self.g = g
        self.k = k
        self.nodes: list[Node] = []
    def add(self, node: Node) -> None:
        if len(self.nodes) < self.k:
            self.nodes.append(node)
            self.nodes.sort(key=lambda n: len(n.getneighs(self.g)), reverse=True)
        else:
            if len(node.getneighs(self.g)) > len(self.nodes[-1].getneighs(self.g)):
                self.nodes[-1] = node
                self.nodes.sort(key=lambda n: len(n.getneighs(self.g)), reverse=True)
    def __str__(self) -> str:
        return f"Highest{self.k}DegreeNodes({[len(n.getneighs(self.g)) for n in self.nodes]})"

## type A = "VAL1" | "VAL2" | "VAL3"+Int
class HWEnum(Enum):
    UndersamplingE1 = 1,
    UndersamplingE2 = 2,
    DiscardingE1 = 3,
    DiscardingE2 = 4,
    DiscardingE1AndUseForE2 = 5
class HoleWarning: 
    name: HWEnum
    value: float|None
    def __init__(self, name: HWEnum, value: float|None = None) -> None:
        self.name = name
        self.value = value
def HWcount(entry: HWEnum) -> int:
    return len(list(filter(lambda x: x.name == entry, printed)))

DEBUG = False
WARNINGS = False
printed = []
def myprint(arg: HoleWarning):
    if WARNINGS: printed.append(arg)

## CONSTANTS ##
filename1 = ("loopgraph/loopgraph.txt", "loopgraph")
filename2 = ("soc-hamsterster/soc-hamsterster.edges", "soc-hamsterster") # is not connected !!
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

class VCRS(NamedTuple):
    vertex: Node
    comp: Set[Node]
    rs_n: float

class ECRS(NamedTuple):
    edge: UEdge
    comp: Set[UEdge]
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
    valMin = min(l_a if l_a is not None else 42, l_b if l_b is not None else 42)
    valMax = max(l_a if l_a is not None else 42, l_b if l_b is not None else 42)
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
    def set_key(self, node: Node, key: int) -> None:
        self.nodes.increase_key(node, key)
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
            self.L0 = L0Set()
            self.L1 = L1Set()
            
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
        self._l0_finish_setup()
        # self.L0List = list(self.L0.nodes)
        # self.L1List = self.L1.nodes.toList()

        # self.l0_created = True
        # self._L00cumsum = list(np.cumsum(list(map(lambda n: len(n.getneighs_L0()), self.L0List))))
        # self._L01cumsum = list(np.cumsum(list(map(lambda n: len(n.getneighs_L1(self)), self.L0List))))

        # self.D00 = lambda: (node := self.L0List[Utils.choose_from_bucket_with_prefix_probability(self._L00cumsum)], Rand.choice(node.getneighs_L0()))
        # self.D01 = lambda: (node := self.L0List[Utils.choose_from_bucket_with_prefix_probability(self._L01cumsum)], Rand.choice(node.getneighs_L1(self)))

    def load_L0(self, L0: list[int]) -> None:
        self.L0 = L0Set() #.deserialize(map(lambda n: self.nodes[n], L0))
        self.L1 = L1Set()
        for n in L0:
            self.L1.add(self.nodes[n])
        for n in L0:
            self.L1.set_key(self.nodes[n], 100)
            u = self.L1.extract_max()
            assert u == self.nodes[n]
            self.L0.add(self, u)

        self._l0_finish_setup()

    def _l0_finish_setup(self) -> None:
        self.L0List = list(self.L0.nodes)
        self.L1List = self.L1.nodes.toList()

        self.l0_created = True
        self._L00cumsum = list(np.cumsum(list(map(lambda n: len(n.getneighs_L0()), self.L0List))))
        self._L01cumsum = list(np.cumsum(list(map(lambda n: len(n.getneighs_L1(self)), self.L0List))))

        self.D00 = lambda: (node := self.L0List[Utils.choose_from_bucket_with_prefix_probability(self._L00cumsum)], Rand.choice(node.getneighs_L0()))
        self.D01 = lambda: (node := self.L0List[Utils.choose_from_bucket_with_prefix_probability(self._L01cumsum)], Rand.choice(node.getneighs_L1(self)))

    def init_m0(self) -> None:
        ret = 0
        for n in self.L0.nodes:
            for neigh in n.getneighs(self):
                if neigh.l == 0 and neigh.nr < n.nr: continue
                else: ret += 1
        self.m0 = ret

    @property 
    def getE00size(self) -> int: return self.m0 - self.getE01size
    @property
    def getE01size(self) -> int: return self._L01cumsum[-1]

    def init_m1bar(self, nroftests: int) -> None:
        avg = 0
        for _ in range(nroftests):
            v = Rand.choice(self.L1List)
            # avg += v.nr_neighs(self) - len(v.getneighs_L0()) #/nroftests # neighs in L1 and L2
            avg += len(v.getneighs_L1(self))/2 + len(v.getneighs_Lge2(self)) #! slow

        self.m1bar = round(avg*len(self.L1List)/nroftests)

    def init_l2bar(self, s1: int, sge2: int) -> None:
        dPlus = 0
        for _ in range(s1):
            v      = Rand.choice(self.L1List)
            dPlus += len(v.getneighs_Lge2(self))
        dPlusAvg = dPlus/s1

        dMinus, trs = 0, 0
        for _ in range(sge2):
            v, _, rs_n = self.reach_Lge2()
            dMinus    += len(v.getneighs_L1(self)) / rs_n ## can be implemented by Bloom-Filter
            trs       += 1 / rs_n
        dMinusAvg = dMinus / trs

        self.l2bar = round(len(self.L1List) * dPlusAvg / dMinusAvg)

    def init_m2bar(self, nroftests: int) -> None:
        assert hasattr(self, "l2bar"), "l2bar not initialized, call init_l2bar"
        avg = 0
        trs = 0
        for _ in range(nroftests):
            v, _, rs = self.reach_Lge2()
            avg += len(v.getneighs_Lge2(self)) / rs / 2
            trs += 1 / rs
        
        self.m2bar = round(self.l2bar * avg / trs)
        
    def reach_E1(self: "LoopHole") -> Tuple[Node, Node, float]:
        _, v = self.D01()
        choosefrom = list(filter(lambda neigh: neigh.nr < v.nr, v.getneighs_L1(self))) + v.getneighs_Lge2(self) # tiebreaking, getneighs_L1 can be implemented by Bloom-Filter
        if len(choosefrom) == 0: return self.reach_E1() # could be empty
        w    = Rand.choice(choosefrom)
        rs   = len(v.getneighs_L0()) / len(choosefrom)
        return v, w, rs
    
    def _find_w(self) -> Node:
        while True: # gets stuck if L2 empty
            u, v = self.D01()
            ws = v.getneighs_Lge2(self)
            if len(ws) > 0: return Rand.choice(ws)
    def _bfs_on_Gge2(self, w) -> Tuple[Set[Node], Set[UEdge]]:
        retC: Set[Node] = set()
        retE: Set[UEdge] = set()
        active = deque([w])
        while len(active) > 0:
            v = active.popleft()
            if v in retC: continue
            retC.add(v)
            for u in v.getneighs_Lge2(self):
                if self._in_Leq2(u) and self._in_Leq2(v): 
                    if u.nr < v.nr: # do tiebreaking, give edge to bigger node
                        retE.add(UEdge(u.nr, v.nr))
                    continue
                retE.add(UEdge(u.nr, v.nr))
                active.append(u)
        if WARNINGS and len(retC) > 200:
            ### & have found a huge component !!
            print ("BFS on C-size:", len(retC), "E-size:", len(retE), "rsC", self.comp_reachability(retC, retE)/len(retC), "rsE:", self.comp_reachability(retC, retE)/len(retE),  "vs", "rs0_E2:", self.rs0_E2 if hasattr(self, "rs0_E2") else "not initialized")
        
            highestC = HighestKDegreeNodes(self, 10)
            for n in retC: highestC.add(n)
            print("Nodes in C:", highestC)

            hightestTot = HighestKDegreeNodes(self, 10)
            for n in self.nodes: hightestTot.add(n)
            print("Nodes in Total:", hightestTot)

            assert False, "huge component found"

        return retC, retE
    def _in_Leq2(self, v: Node) -> bool:
        if v.l is None: v.l = 2 if len(v.getneighs_L1(self)) > 0 else 42 # getneighs_L1 can be implemented by Bloom-Filter
        return v.l == 2
    def comp_reachability(self, C: Set[Node], E: Set[UEdge]) -> float:
        rsC = 0
        for v in C:
            if not self._in_Leq2(v): continue
            rsV = 0
            for u in v.getneighs_L1(self): # getneighs_L1 can be implemented by Bloom-Filter
                dMinus = len(u.getneighs_L0())
                dPlus  = len(u.getneighs_Lge2(self))
                rsU    = dMinus / dPlus
                rsV    += rsU
            rsC += rsV
        return rsC
    def reach_Lge2(self: "LoopHole") -> VCRS:
        C, E = self._bfs_on_Gge2(self._find_w())
        rsC = self.comp_reachability(C, E)
        v = Rand.choice(list(C))
        return VCRS(v, C, rsC/len(C))
    def reach_Ege2(self) -> ECRS:
        C, E = self._bfs_on_Gge2(self._find_w())
        if len(E) == 0: return self.reach_Ege2() # could be empty
        rsC = self.comp_reachability(C, E)
        e = Rand.choice(list(E))
        return ECRS(e, E, rsC/len(E))

    def calculate_rs0_E1(self, nroftests: int, eps: float) -> list[float]:
        """ Calculate baseline reachability for E1, should be ~1/15*avg(rs0_E1)? for decent performance """
        rss = []
        for _ in range(nroftests):
            _, _, rs_e = self.reach_E1()
            rss.append(rs_e)
        
        self.rs0_E1 = self.estimate_baseline_reachability(rss, eps)
        return rss
    
    def calculate_rs0_E2(self, nroftests: int, eps: float) -> list[float]:
        """ Calculate baseline reachability for E2, should be ~1/100*avg(rs0_E2)? for decent performance """
        rss = []
        for _ in range(nroftests):
            _, _, rs_e = self.reach_Ege2()
            rss.append(rs_e)
            
        self.rs0_E2 = self.estimate_baseline_reachability(rss, eps)
        return rss

    def estimate_baseline_reachability(self, rss: list[float], eps: float) -> float:
        """ A higher baseline reachability means a faster algorithms that undersamples more often """
        np_rss = np.array(sorted(rss))
        w = np_rss[0] / np_rss
        cumsum = np.cumsum(w)
        cw = cumsum / cumsum[-1]
        ri = np_rss[(np.argmax(cw >= eps))]
        return ri


    def sample_edges(self: "LoopHole", num_samples: int) -> list[UEdge]:
        """
        Samples num_samples edges from the graph G
        """

        assert hasattr(self, "l0_created") and self.l0_created, "L0 not initialized, call generate_L0"
        assert hasattr(self, "m0"), "m0 not initialized, call init_m0"
        assert hasattr(self, "m1bar"), "m1bar not initialized, call init_m1bar"
        assert hasattr(self, "m2bar"), "m2bar not initialized, call init_m2bar"
        assert hasattr(self, "rs0_E2"), "rs0_E2 not initialized, call calculate_rs0_E2"
        assert hasattr(self, "rs0_E1"), "rs0_E1 not initialized, call calculate_rs0_E1"
        e_estimates = [self.m0, self.m1bar, self.m2bar]

        edges: list[UEdge] = []
        for _ in range(num_samples):
            bucket = Utils.choose_from_bucket_with_probability(e_estimates)
            if bucket == 0:
                edges.append(self.sample_edge_E0())
            elif bucket == 1:
                edges.append(self.sample_edge_E1())
            else:
                edges.append(self.sample_edge_E2())
        return edges
    
    def sample_edge_E0(self: "LoopHole") -> UEdge:
        chooseFrom = Utils.choose_from_bucket_with_probability([self.getE00size, self.getE01size])
        if chooseFrom == 0: return UEdge.factory(self.D00())
        else: return UEdge.factory(self.D01())

    def sample_edge_E1(self: "LoopHole") -> UEdge:
        def sample_once():
            return self.reach_E1()
        
        v, w, rs = sample_once()
        prob = self.rs0_E1 / rs
        if DEBUG and prob > 1: 
            print(f"undersampling E1 where prob > 1: {prob}")
        if WARNINGS and prob > 1:
            myprint(HoleWarning(HWEnum.UndersamplingE1, prob))

        if Rand.random() < prob: return UEdge.factory((v, w))
        else: 
            if DEBUG and (w.l is None or w.l > 1): 
                print(f"discarding E1, could reuse it in reach_LEge2")
            if WARNINGS and (w.l is None or w.l > 1):
                myprint(HoleWarning(HWEnum.DiscardingE1AndUseForE2))
            if WARNINGS:
                myprint(HoleWarning(HWEnum.DiscardingE1))
            return self.sample_edge_E1()

    def sample_edge_E2(self: "LoopHole") -> UEdge:
        def sample_once():
            e, _, rs_e = self.reach_Ege2()
            return e, self.rs0_E2 / rs_e
        

        while True:
            e, prob = sample_once()
            if prob > 1:
                if DEBUG: print(f"undersampling E2 where prob > 1: {prob}")
                if WARNINGS: myprint(HoleWarning(HWEnum.UndersamplingE2, prob))
            if Rand.random() < prob: return e

            if WARNINGS: myprint(HoleWarning(HWEnum.DiscardingE2))
            continue
        
        # e, prob = sample_once()
        # if DEBUG and prob > 1: 
        #     print(f"undersampling E2 where prob > 1: {prob}")
        # if WARNINGS and prob > 1:
        #     myprint(HoleWarning(HWEnum.UndersamplingE2, prob))

        # if Rand.random() < prob: return e
        # else: 
        #     if WARNINGS:
        #         myprint(HoleWarning(HWEnum.DiscardingE2))
        #     return self.sample_edge_E2()



    def getneighs(self, i: int) -> np.ndarray:
        return self.csr.indices[self.csr.indptr[i]:self.csr.indptr[i+1]]
    
    def nrneighs(self, i: int) -> int:
        return self.csr.indptr[i+1] - self.csr.indptr[i]
    
    def bfs(self, start: Node) -> list[bool]:
        visited = [False] * self.n
        active = deque([start])
        while len(active) > 0:
            v = active.popleft()
            if visited[v.nr]: continue

            visited[v.nr] = True
            for u in v.getneighs(self):
                active.append(u)
        return visited
    
    def plot(self) -> None:
        with open("graph.dot", "w") as dot_file:
            dot_file.write("graph G {\n")
            for n in self.nodes:
                if DEBUG: print(n.nr, n.oldlabel, n.l, n.color)
                dot_file.write(f'    {n.nr} [label="{n.oldlabel}", color="{n.color}", width=0.3, height=0.2, style=filled];\n')
            for e in self.edges:
                dot_file.write(f'    {e.a} -- {e.b} [color="{self.nodes[e.a].color if self.nodes[e.a].islayerLT(self.nodes[e.b]) else self.nodes[e.b].color}"];\n')
            dot_file.write("}\n")
        run(["fdp", "-Tpng", "graph.dot", "-o", "graph.png"], check=True)

    def export(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self.serialize(), f)

    @staticmethod
    def import_graph(filename: str) -> "LoopHole":
        with open(filename, "rb") as f:
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

        # g.load_L0(ser["L0"])
        # set _neighs_L0
        for n in g.L0.nodes:
            for neigh in n.getneighs(g):
                if neigh.l == 0: n.addneigh_L0(neigh)
                elif neigh.l == 1: n.addneigh_L1(neigh)
                else: raise ValueError("neigh.l >= 2")

                neigh.addneigh_L0(n)
            

        g.L1 = L1Set()
        for n in ser["L1"]: g.L1.add(g.nodes[n])
        g._l0_finish_setup()
        for i, n in enumerate(g.nodes): assert n.nr == i
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

def LOOPHOLE_FACTORY_CONNECTED(filename: str, name: str, l0_size: float, seed: int = 42, theorymodel = THEORYMODEL.WEAK) -> LoopHole:
    _g = LOOPHOLE_FACTORY(filename, name, l0_size, seed, theorymodel=theorymodel)
    _visited_nr = _g.bfs(_g.nodes[0])
    if all(_visited_nr): return _g

    _visited_oldlabel = [False] * (_g.n+1)
    for ind, v in enumerate(_visited_nr): 
        if v: _visited_oldlabel[_g.nodes[ind].oldlabel] = True
    
    only_use = lambda e: _visited_oldlabel[e.a] and _visited_oldlabel[e.b]
    return LOOPHOLE_FACTORY(filename, name, l0_size, seed, use_only=only_use, theorymodel=theorymodel)

@timed
def LOOPHOLE_FACTORY(filename: str, name: str, l0_size: float, seed: int = 42, use_only = lambda x: True, theorymodel = THEORYMODEL.WEAK) -> LoopHole:
    Rand.seed(seed)
    try:
        raise FileNotFoundError # storing files does not support different theory models
        g = _load_graph(name, l0_size, seed)
        print ("--- Graph loaded ---")
        return g
    except FileNotFoundError:
        print ("--- Graph not yet created, creating it now ---")
        return _store_graph(filename, name, l0_size, seed, use_only=use_only, theorymodel=theorymodel)

def _make_graph(filename: str, name: str, l0_size: float, seed: int = 42, v0 = None, use_only = lambda x: True, theorymodel = THEORYMODEL.WEAK) -> LoopHole:
    g = LoopHole(filename, use_only=use_only)
    if l0_size>0: g.generate_L0(v0, l0_size, theoretical_model=theorymodel)
    return g

def _store_graph(filename: str, name: str, l0_size: float, seed: int = 42, use_only = lambda x: True, theorymodel = THEORYMODEL.WEAK) -> LoopHole:
    g = _make_graph(filename, name, l0_size, seed, use_only=use_only, theorymodel=theorymodel)
    g.export("bin/"+name+str(l0_size)+"_"+str(seed)+".bin")
    return g

def _load_graph(name: str, l0_size: float, seed: int = 42) -> LoopHole:
    return LoopHole.import_graph("bin/"+name+str(l0_size)+"_"+str(seed)+".bin")




if __name__ == "__main__":
    Rand.seed(42)
    print ("--- Testing ---")
    file = filename7
    g = LOOPHOLE_FACTORY_CONNECTED(file[0], file[1], 0.05)

    g.init_m0()
    g.init_m1bar(1000)
    g.init_l2bar(1000, 1000)
    g.init_m2bar(1000)
    g.calculate_rs0_E1(200, 0.4)
    g.calculate_rs0_E2(200, 0.3)

    print ("Nodes:", g.n, "Edges:", g.m)
    # print (f"m0: {g.m0}, m1bar: {g.m1bar}, m2bar: {g.m2bar}, l2bar: {g.l2bar}")
    # print (f"m0: {len(list(filter(lambda e: edge_in(g, e).is_E0, g.edges)))}, m1_is: {len(list(filter(lambda e: edge_in(g, e).is_E1, g.edges)))}, m2_is: {len(list(filter(lambda e: edge_in(g, e).is_E2, g.edges)))}, l2_is: {len(list(filter(lambda n: n.l is None or n.l >= 2, g.nodes)))}")
    # print ("rs0_E1:", g.rs0_E1, "rs0_E2:", g.rs0_E2)

    # g.sample_edges(10)

    print ("--- DONE ---")
    # g.plot()
