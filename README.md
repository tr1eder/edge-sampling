# Edge-sampling
A work-in-progress on sampling edges or other structures in a large social network by exploiting the core-periphery structure.

---

## Pseudocode
```python
def sample_E0:
	choose (P=m00/m0): return E00.sample() # from undirected subgraph L0
	choose (P=m01/m0): return E01.sample() # from bipartite  subgraph L0-L1

def sample_E1:
	edge e(u,v) = E01.sample()                # s.t. u in L0, v in L1
	node w      = (v.neighbors()\L0).sample() # or ?retry if w in L0?
	
	choose (P=d_L1cup2(v)/d_L0(v)*m01/2/m1): return {v,w}       # success
	choose (otherwise):                      return sample_E1() # retry

def sample_E2:
	u, C, rs    = reach_Lge2()
	edge e{v,w} = C.sample()
	
	choose (P=rs0/rs):  return {v,w}     # success
	choose (otherwise): return sample_E2 # retry
```

## Usage
- samplingEdges.py contains the main program
- `g = LoopHole(filename)` generates a new Graph object
- `g.generate_L0()` generates the $L_0$, $L_1$ and $\mathcal D$ (here: $E_{00} \cup E_{01}$)
- `estimate_lge2(s1, sge2)` estimates the size of $L_{\ge 2}$
- `reach_Lge2()` finds a vertex $v \in L_{\ge 2}$, its component and reach. score
- `sample_edges(estimate_E)` should sample from $E_0, E_1, E_{\ge2}$

## Visuals
- `graph.png` shows a visual of a small graph, where
  - red nodes/edges are of $L_0 / E_0$,
  - blue nodes/edges are of $L_1 / E_1$,
  - black nodes/edges are of $L_{\ge2} / E_{\ge2}$
