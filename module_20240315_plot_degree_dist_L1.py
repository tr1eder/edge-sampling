from loophole import LoopHole, edge_in, E_PARTITION
import loophole, timeit
import matplotlib.pyplot as plt
import numpy as np
import bisect, sys
from Random import Rand

import sys
sys.setrecursionlimit(1000000)  # Set a higher recursion limit


# def plot_scatter(nodes, d_L0s, d_L1cup2s):
#     """ Plotting d_L0 vs d_L1cup2 as a scatter-plot """
#     d_labels =  [str(node.oldlabel) for node in nodes]
#     plt.scatter(d_L0s, d_L1cup2s)
#     for i, txt in enumerate(d_labels): plt.annotate(txt, (d_L0s[i], d_L1cup2s[i]))
#     plt.xlabel("d_L0")
#     plt.ylabel("d_L1cup2")
#     plt.show()

# def plot_histogram(d_L0s, d_L1cup2s):
#     """ Plotting d_L1cup2/d_L0 as a histogram """
#     d_ratios = [0]*int(max(d_L1cup2s/d_L0s)+1)
#     for ratio in d_L1cup2s/d_L0s: d_ratios[int(ratio)] += 1
#     plt.bar(range(len(d_ratios)), d_ratios)
#     plt.xlabel("d_L1cup2/d_L0")
#     plt.ylabel("# of nodes")
#     plt.yscale('log')
#     plt.show()

# def print_percentiles(text, d_ratios):
#     """ Print the percentiles of the d_L1cup2/d_L0 ratios """
#     print(f"Percentiles of d_L1cup2/d_L0 ratios for {text}: ", end="")
#     for p in [50, 80, 90, 95, 99, 100]: print(f"{p}%: {np.percentile(d_ratios, p):<5.1f}", end="   ")
#     print()

def get_E0_E1_E2(g):
    e0, e1, e2 = 0, 0, 0
    e11, e12 = 0, 0
    for edge in g.edges:
        if edge_in(g, edge).is_E0: e0 += 1
        elif edge_in(g, edge).is_E1: e1 += 1
        else: e2 += 1

        if edge_in(g, edge) == E_PARTITION.E11: e11 += 1
        if edge_in(g, edge) == E_PARTITION.E12: e12 += 1
    return e0, e1, e2, e11, e12

def print_lost_edges(text, g, nodes, degrees_L12, ratios, percentile, threshold):
    """ Print the number of lost edges for 95th percentile """
    # percentile = 95
    # val = np.percentile(ratios, percentile)

    e0, e1, e2, e11, e12 = get_E0_E1_E2(g)

    # e11_lost = 0
    # e12_lost = 0
    # for node, degree, ratio in zip(nodes, degrees_L12, ratios):
    #     if ratio >= threshold:
    #         e11_lost += len(node.getneighs_L1())/2 # divide by 2 because of double counting
    #         e12_lost += len(node.getneighs_Lge2()) 

    # print(f"{text} | E0: {e0:>6} E11: {e11-e11_lost:>6.0f}/{e11:<6} E12: {e12-e12_lost:>6.0f}/{e12:<6} E2: {e2:>6} | Lost edges for {percentile:.0f}th percentile: E11{{{e11_lost:>6.0f} }}, E12{{{e12_lost:>6.0f} }}")
    print(f"{text} | E0: {e0:>6} E11: {e11:>6} E12: {e12:>6} E2: {e2:>6}")

def plot_histogram_with_without_tiebreaks(text, g, nodes, save=False): 
    def tiebreak(this, other) -> bool:
        ratioThis: float = float(d_L1s_wout_tiebreaks[indexin[this]]/d_L0s[indexin[this]])
        ratioOther: float = float(d_L1s_wout_tiebreaks[indexin[other]]/d_L0s[indexin[other]])
        return ratioThis < ratioOther or (ratioThis == ratioOther and this.nr < other.nr)
    indexin = {node: i for i, node in enumerate(nodes)}
    d_L0s = np.array([len(node.getneighs_L0()) for node in nodes])
    d_L1s_wout_tiebreaks = np.array([len(node.getneighs_L1(g))/2 + len(node.getneighs_Lge2(g)) for node in nodes])
    d_L1s_with_tiebreaks = np.array([len(list(filter(lambda other: tiebreak(node, other), node.getneighs_L1(g)))) + len(node.getneighs_Lge2(g)) for node in nodes])

    ratioWO = d_L1s_wout_tiebreaks/d_L0s
    ratioW = d_L1s_with_tiebreaks/d_L0s
    xmax: int = max(max(ratioW), max(ratioWO))
    bins: int = 80

    myfunc =    lambda xs: np.where(xs < 40, xs, 30+xs/4)
    myfuncinv = lambda xs: np.where(xs < 40, xs, 4*xs-120)

    # plot two plots side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(text+" Histogram of d_L1cup2/d_L0 with and without tiebreaks")
    countsWO, binsWO, _ = axs[0].hist(ratioWO, bins=int(max(ratioWO)/xmax*bins), rwidth=0.9)
    axs[0].set_title("Without tiebreaks")
    axs[0].set_xlabel("d_L1cup2/d_L0")
    axs[0].set_ylabel("# of nodes")
    axs[0].set_xscale('function', functions=[myfunc, myfuncinv])

    countsW, binsW, _ = axs[1].hist(ratioW, bins=int(max(ratioW)/xmax*bins), rwidth=0.9)
    axs[1].set_title("With tiebreaks")
    axs[1].set_xlabel("d_L1cup2/d_L0")
    axs[1].set_ylabel("# of nodes")
    axs[1].set_xscale('function', functions=[myfunc, myfuncinv])

    ymax = max(max(countsWO), max(countsW))
    axs[0].set_xlim(-.5, xmax)
    axs[0].set_ylim(-ymax/200, ymax)
    axs[1].set_xlim(-.5, xmax)
    axs[1].set_ylim(-ymax/200, ymax)

    # plot 95th percentile (where 5% of edges are lost)
    percentile = 95
    def get_percentile_value(ratio, percentile):
        sortedRatio = sorted(ratio)
        prefixsum = np.cumsum(sortedRatio)
        return sortedRatio[bisect.bisect_left(prefixsum, percentile/100*prefixsum[-1])]
    # compute prefix sum and then the value at the 95th percentile
    valueWO = get_percentile_value(ratioWO, percentile)
    axs[0].axvline(x=valueWO, color='r', linestyle='--', label=f"{percentile}% ({valueWO})")
    axs[0].legend()

    valueW = get_percentile_value(ratioW, percentile)
    axs[1].axvline(x=valueW, color='r', linestyle='--', label=f"{percentile}% ({valueW})")
    axs[1].legend()


    print_lost_edges(text, g, nodes, d_L1s_wout_tiebreaks, ratioWO, percentile, valueWO)
    
    
    if save:
        plt.savefig(f"PLOT_{text}.png")
    else:
        plt.show()


filename1 = ("testgraph/small.txt", "test")
filename2 = ("soc-hamsterster/soc-hamsterster.edges", "soc-hamsterster")
filename3 = ("facebook_combined/facebook_combined.txt", "ego-facebook")
filename4 = ("git_web_ml/git_web_ml/musae_git_edges.csv", "musae-git")
filename5 = ("gemsec_deezer_dataset/deezer_clean_data/HR_edges.csv", "gemsec-deezer")
filename6 = ("gemsec_facebook_dataset/facebook_clean_data/artist_edges.csv", "gemsec-facebook")
filename7 = ("twitch_gamers/large_twitch_edges.csv", "twitch-gamers")
filename8 = ("as-skitter/as-skitter.txt", "as-skitter")

if __name__ == "__main__":
    print ("--- Plotting degree distribution of L1 ---")

    # for filename, name in [loophole.filename4]:
    #     for l0_size in [.05]:
    #         for seed in [42]:
                # Rand.seed(42)
                # g = LoopHole(filename, loophole.only_use1)
                # g.generate_L0(l0_percentage_size=l0_size)


                # plot_scatter(nodes, d_L0s, d_L1cup2s)
                # plot_histogram(d_L0s, d_L1cup2s)
                # print_percentiles(f"{name.upper()} with l0_size={l0_size:4.2f}", d_L1cup2s/d_L0s)
                # print_E0_E1_E2(f"{name.upper()} with l0_size={l0_size:4.2f}", g)
                # print_lost_edges(f"{name.upper()} with l0_size={l0_size:4.2f}", g, nodes, d_L1cup2s, d_L1cup2s/d_L0s)
                
                
                
    filename, name = globals()[sys.argv[1]]
    l0_size = float(sys.argv[2])
    seed = int(sys.argv[3])
    print (f"filename: {filename}, name: {name}, l0_size: {l0_size}")
    g = loophole.LOOPHOLE_FACTORY(filename, name, l0_size, seed)
    nodes =     g.L1.nodes.toList()
    plot_histogram_with_without_tiebreaks(f"{name.upper()} with l0_size={l0_size:4.2f}", g, nodes, True)






    print("--- DONE ---")

