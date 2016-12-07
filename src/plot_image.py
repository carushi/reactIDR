import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import utility
from idr_wrapper import *


def plot_score_idr_scatter(s1, s2, i12, head, log):
    for rank in [False, True]:
        if rank:
            rank1 = get_rank_dictionary(s1)
            s1 = [rank1[x] for x in s1]
            rank2 = get_rank_dictionary(s2)
            s2 = [rank2[x] for x in s2]
            tail = '_rank_'
        else:
            tail = ''
            if log:
                plt.xscale('log')
                plt.yscale('log')
                s1 = [x+1 for x in s1]
                s2 = [x+1 for x in s2]
        # colors = cm.rainbow(np.linspace(0, 1, len(IDR)))
        # plt.scatter(s1, s2, c=colors)
        plt.scatter(s1, s2, c=i12)
        plt.gray()
        plt.colorbar()
        # for i in range(4):
        #     start = [0, 10, 20, 1000][i]
        #     end = [10, 20, 1000, -1][i]
        #     col = ['black', 'grey', 'pink', 'red'][i]
        #     index = [idx for idx in range(len(i12)) if i12[idx] >= start and (end < 0 or i12[idx] < end) ]
        #     if len(index) == 0: continue
            # plt.scatter(list(map(lambda x: int(s1[x]), index)), list(map(lambda x: int(s2[x]), index)), c=col, label=str(start)+"~"+str(end-1))
        plt.legend(loc='upper left', numpoints=1)
        ofile = head+tail+"_gray.png"
        plt.savefig(ofile)
        plt.clf()
        utility.log("# "+ofile+" min="+str(min(i12))+" max="+str(max(i12)))

def plot_density(IDR, ofile):
    sns.set_style('whitegrid')
    sns_plot = sns.kdeplot(np.array(IDR), bw=0.1)
    fig = sns_plot.get_figure()
    fig.savefig(ofile)
    fig.clf()
