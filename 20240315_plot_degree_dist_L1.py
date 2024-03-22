from LoopHole import Graph, edge_in, E_PARTITION
import matplotlib.pyplot as plt
import numpy as np
from Random import Rand


def plot_scatter(nodes, d_L0s, d_L1cup2s):
    """ Plotting d_L0 vs d_L1cup2 as a scatter-plot """
    d_labels =  [str(node.oldlabel) for node in nodes]
    plt.scatter(d_L0s, d_L1cup2s)
    for i, txt in enumerate(d_labels): plt.annotate(txt, (d_L0s[i], d_L1cup2s[i]))
    plt.xlabel("d_L0")
    plt.ylabel("d_L1cup2")
    plt.show()

def plot_histogram(d_L0s, d_L1cup2s):
    """ Plotting d_L1cup2/d_L0 as a histogram """
    d_ratios = [0]*int(max(d_L1cup2s/d_L0s)+1)
    for ratio in d_L1cup2s/d_L0s: d_ratios[int(ratio)] += 1
    plt.bar(range(len(d_ratios)), d_ratios)
    plt.xlabel("d_L1cup2/d_L0")
    plt.ylabel("# of nodes")
    plt.yscale('log')
    plt.show()

def print_percentiles(text, d_ratios):
    """ Print the percentiles of the d_L1cup2/d_L0 ratios """
    print(f"Percentiles of d_L1cup2/d_L0 ratios for {text}: ", end="")
    for p in [50, 80, 90, 95, 99, 100]: print(f"{p}%: {np.percentile(d_ratios, p):<5.1f}", end="   ")
    print()

def get_E0_E1_E2(g):
    e0, e1, e2 = 0, 0, 0
    for edge in g.edges:
        if edge_in(g, edge) == E_PARTITION.E0: e0 += 1
        elif edge_in(g, edge) == E_PARTITION.E1: e1 += 1
        else: e2 += 1
    return e0, e1, e2

def print_E0_E1_E2(text, g):
    """ Print the number of edges in E0, E1 and E2 """
    e0, e1, e2 = get_E0_E1_E2(g)
    print(f"{text} | E0: {e0:<5}, E1: {e1:<5}, E2: {e2:<5}")

def print_lost_edges(text, g, nodes, degrees_L12, ratios):
    """ Print the number of lost edges for 95th percentile """
    percentile = 95
    val = np.percentile(ratios, percentile)

    e0, e1, e2 = get_E0_E1_E2(g)
    e11_lost = 0
    e12_lost = 0

    for node, degree, ratio in zip(nodes, degrees_L12, ratios):
        if ratio >= val:
            e11_lost += len(node.getneighs_L1())/2 # divide by 2 because of double counting
            e12_lost += len(node.getneighs_Lge2()) 

    print(f"{text} | E0: {e0:<6}, E1: {e1-e11_lost-e12_lost:>6.0f}/{e1:<6}, E2: {e2:<6} | Lost edges for {percentile:.0f}th percentile: E11{{{e11_lost:>6.0f} }}, E12{{{e12_lost:>6.0f} }}")

def plot_histogram_with_without_tiebreaks(text, nodes): 
    def tiebreak(this, other) -> bool:
        ratioThis: float = float(d_L1s_wout_tiebreaks[indexin[this]]/d_L0s[indexin[this]])
        ratioOther: float = float(d_L1s_wout_tiebreaks[indexin[other]]/d_L0s[indexin[other]])
        return ratioThis < ratioOther or (ratioThis == ratioOther and this.nr < other.nr)
    indexin = {node: i for i, node in enumerate(nodes)}
    d_L0s = np.array([len(node.getneighs_L0()) for node in nodes])
    d_L1s_wout_tiebreaks = np.array([len(node.getneighs_L1())/2 + len(node.getneighs_Lge2()) for node in nodes])
    d_L1s_with_tiebreaks = np.array([len(list(filter(lambda other: tiebreak(node, other), node.getneighs_L1()))) + len(node.getneighs_Lge2()) for ind, node in enumerate(nodes)])

    # plot two plots side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(text+" Histogram of d_L1cup2/d_L0 with and without tiebreaks")
    axs[0].hist(d_L1s_wout_tiebreaks/d_L0s, bins=100)
    axs[0].set_title("Without tiebreaks")
    axs[0].set_xlabel("d_L1cup2/d_L0")
    axs[0].set_ylabel("# of nodes")
    # axs[0].set_yscale('log')

    axs[1].hist(d_L1s_with_tiebreaks/d_L0s, bins=100)
    axs[1].set_title("With tiebreaks")
    axs[1].set_xlabel("d_L1cup2/d_L0")
    axs[1].set_ylabel("# of nodes")
    # axs[1].set_yscale('log')

    plt.show()




if __name__ == "__main__":
    print ("--- Plotting degree distribution of L1 ---")

    filename1 = ("testgraph/small.txt", "test")
    filename2 = ("soc-hamsterster/soc-hamsterster.edges", "soc-hamsterster")
    filename3 = ("facebook_combined/facebook_combined.txt", "ego-facebook")
    filename4 = ("git_web_ml/git_web_ml/musae_git_edges.csv", "musae-git")
    filename5 = ("gemsec_deezer_dataset/deezer_clean_data/HR_edges.csv", "gemsec_deezer")
    filename6 = ("gemsec_facebook_dataset/facebook_clean_data/artist_edges.csv", "gemsec_facebook")

    only_use1 = lambda e: True
    only_use2 = lambda e: e.inrange(0, 6)
    only_use3 = lambda e: e.inrange(10, 20)


    for filename, name in [filename5, filename6]:
        for l0_size in [0.02, 0.05, 0.10]:
            Rand.seed(42)
            g = Graph(filename, only_use1)
            g.generate_L0(l0_percentage_size=l0_size)

            nodes =     list(g.L1.nodes)
            # d_L0s =     np.array([len(node.getneighs_L0()) for node in nodes])
            # d_L1cup2s = np.array([len(node.getneighs_L1()) + len(node.getneighs_Lge2()) for node in nodes])

            # plot_scatter(nodes, d_L0s, d_L1cup2s)
            # plot_histogram(d_L0s, d_L1cup2s)
            # print_percentiles(f"{name.upper()} with l0_size={l0_size:4.2f}", d_L1cup2s/d_L0s)
            # print_E0_E1_E2(f"{name.upper()} with l0_size={l0_size:4.2f}", g)
            # print_lost_edges(f"{name.upper()} with l0_size={l0_size:4.2f}", g, nodes, d_L1cup2s, d_L1cup2s/d_L0s)
            plot_histogram_with_without_tiebreaks(f"{name.upper()} with l0_size={l0_size:4.2f}", nodes)

    # def func() -> str: 
    #     Rand.seed(42)
    #     g = Graph(filename2[0], only_use1)
    #     g.generate_L0(l0_percentage_size=0.05)
    #     nodes = list(g.L1.nodes)
    #     nodes.sort(key=lambda x: x.oldlabel)
    #     edges = list(g.edges)
    #     print (str(nodes))
    #     return str(nodes) + str(edges)
    
    # print (Rand.isDeterministic(func, 5))
    










    # #// linlog plot
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    print("--- DONE ---")


# arr = [1,2,3]
# arr.index