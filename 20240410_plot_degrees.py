import sys
import loophole
from loophole import filename1, filename2, filename3, filename4, filename5, filename6, filename7, filename8
import matplotlib.pyplot as plt
import numpy as np

def plot_Lx_degree_distribution(text, g: loophole.LoopHole, axs, nodes):
    axs.set_title(text)

    degrees_L0 =   [len(n.getneighs_L0()) for n in nodes]
    degrees_L1 =   [len(n.getneighs_L1(g)) for n in nodes]
    degrees_Lge2 = [len(n.getneighs_Lge2(g)) for n in nodes]

    zipped = list(zip(degrees_L0, degrees_L1, degrees_Lge2))
    zipped.sort(key=lambda x: sum(x))
    zipped = np.array(zipped)
    zipped = zipped[:int(len(zipped)*0.95)]

    axs.bar(np.arange(len(zipped)), zipped[:,0], color='r', label='L0')
    axs.bar(np.arange(len(zipped)), zipped[:,1], bottom=zipped[:,0], color='g', label='L1')
    axs.bar(np.arange(len(zipped)), zipped[:,2], bottom=zipped[:,0]+zipped[:,1], color='b', label='Lge2')

    axs.set_xlabel("Nodes")
    axs.set_ylabel("Degree")
    axs.legend()

if __name__ == "__main__":
    print ("--- Plotting degree distribution of L1 ---")

    filename, name = globals()[sys.argv[1]]
    seed = int(sys.argv[2])
    l0_sizes = [.01, .02, .05]

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    for i, l0_size in enumerate(l0_sizes):
        print (f"filename: {filename}, name: {name}, l0_size: {l0_size}")
        g = loophole.LOOPHOLE_FACTORY(filename, name, l0_size, seed)

        nodes_L1 = g.L1.nodes.toList()
        nodes_Lge2 = list(filter(lambda n: n.l is None or n.l > 1, map(lambda nr: g.nodes[nr], np.arange(g.n))))

        plot_Lx_degree_distribution(f"L1 with l0_size={l0_size:4.2f}", g, axs[0, i], nodes_L1)
        plot_Lx_degree_distribution(f"L2 with l0_size={l0_size:4.2f}", g, axs[1, i], nodes_Lge2)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Degree distribution of L1 and L2 of {name.upper()} (n={g.n})", fontsize=18)
    plt.savefig(f"PLOT_degreedist_{name}_L1_L2.png")
    # plt.show()
    
    print ("--- DONE ---")