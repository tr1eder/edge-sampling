# Edge-sampling
A work-in-progress on sampling edges/triangles or other structures in a large social network by exploiting the core-periphery structure.

---

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


## TODO
- sampling from $E_1, E_2$, estimating $E_1, E_2$
