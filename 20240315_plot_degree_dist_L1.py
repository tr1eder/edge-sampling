from LoopHole import Graph
import matplotlib.pyplot as plt
from Random import Rand



if __name__ == "__main__":
    Rand.seed(32)
    print ("--- Plotting degree distribution of L1 ---")



    filename1 = "testgraph/small.txt"
    filename2 = "facebook_combined/facebook_combined.txt"
    filename3 = "soc-hamsterster/soc-hamsterster.edges"
    filename4 = "git_web_ml/git_web_ml/musae_git_edges.csv"

    only_use1 = lambda e: True
    only_use2 = lambda e: e.inrange(0, 6)
    only_use3 = lambda e: e.inrange(10, 20)

    g = Graph(filename4, only_use1)
    g.generate_L0(l0_percentage_size=0.01)
    # g.plot()

    nodes =     list(g.L1.nodes)
    d_L0s =     [len(node.getneighs_L0()) for node in nodes]
    d_L1cup2s = [len(node.getneighs_L1()) + len(node.getneighs_Lge2()) for node in nodes]
    d_labels =  [str(node.oldlabel) for node in nodes]

    plt.scatter(d_L0s, d_L1cup2s)
    for i, txt in enumerate(d_labels): plt.annotate(txt, (d_L0s[i], d_L1cup2s[i]))
    plt.xlabel("d_L0")
    plt.ylabel("d_L1cup2")
    # #// linlog plot
    # plt.xscale('log')
    # plt.yscale('log')
    plt.show()

    print("--- DONE ---")