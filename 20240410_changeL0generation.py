import sys
import loophole, module_20240315_plot_degree_dist_L1 as degreedist
from loophole import filename1, filename2, filename3, filename4, filename5, filename6, filename7, filename8
import matplotlib.pyplot as plt
import numpy as np

def namedToString(values, names):
    return ", ".join([f"{name}: {value}" for name, value in zip(names, values)])


if __name__ == "__main__":

    filename, name = globals()[sys.argv[1]]
    l0_size = float(sys.argv[2])
    seed = int(sys.argv[3])

    g_normal = loophole.LOOPHOLE_FACTORY(filename, name, l0_size, seed)

    g_betterV0 = loophole._make_graph(filename, name, 0, seed)
    v0 = g_betterV0.nodes[np.argmax([g_betterV0.getneighs(n).size for n in range(g_betterV0.n)])]
    g_betterV0.generate_L0(v0, l0_size, loophole.THEORYMODEL.MEDIUM)

    g_highestDegreeL0 = loophole._make_graph(filename, name, 0, seed)
    g_highestDegreeL0.generate_L0(l0_percentage_size=l0_size, theoretical_model=loophole.THEORYMODEL.STRONG)

    print (f"--- Created filename: {filename}, name: {name}, l0_size: {l0_size} ---")
    print ("--- Comparisons ---")
    print (f"L0-sizes are normal {len(g_normal.L0.nodes)} vs betterV0 {len(g_betterV0.L0.nodes)} vs highestDegreeL0 {len(g_highestDegreeL0.L0.nodes)}")
    print (f"L1-sizes are normal {len(g_normal.L1.nodes)} vs betterV0 {len(g_betterV0.L1.nodes)} vs highestDegreeL0 {len(g_highestDegreeL0.L1.nodes)}")
    print (f"L1-sizes of total graph are normal {len(g_normal.L1.nodes)/g_normal.n*100:2.0f}% vs betterV0 {len(g_betterV0.L1.nodes)/g_betterV0.n*100:2.0f}% vs highestDegreeL0 {len(g_highestDegreeL0.L1.nodes)/g_highestDegreeL0.n*100:2.0f}%")
    print (f"L1-size improvement: betterV0 {len(g_betterV0.L1.nodes)/len(g_normal.L1.nodes):4.2f}x vs highestDegreeL0 {len(g_highestDegreeL0.L1.nodes)/len(g_normal.L1.nodes):4.2f}x")

    # --

    print (f"Edge distribution: normal {namedToString(degreedist.get_E0_E1_E2(g_normal), ["e0", "e1", "e2", "e11", "e12"])} vs betterV0 {namedToString(degreedist.get_E0_E1_E2(g_betterV0), ["e0", "e1", "e2", "e11", "e12"])} vs highestDegreeL0 {namedToString(degreedist.get_E0_E1_E2(g_highestDegreeL0), ["e0", "e1", "e2", "e11", "e12"])}")

    # --

    print ("--- Done ---")