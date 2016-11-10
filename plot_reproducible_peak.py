import utility
import numpy as np
import random
import subprocess
import os
from statsmodels.tsa.stattools import acf
import pylab
import pvalue
import matplotlib.pyplot as plt


def plot_profile(self, xmin, xmax):
    sidx = pvalue.check_dist_model(self.options.dist_model)
    score_file, idr_file = self.output_file_name(cidx, sidx)
    self.dicts = list(utility.parse_score(score_file))
    if os.path.exists(idr_file):
        IDR = utility.parse_idr(idr_file)
        self.get_profile(IDR)
    else:
        IDR = self.no_idr()
    s1, s2 = get_concatenated_scores(IDR.keys(), dict1, dict2)
    i12 = get_concatenated_score(IDR.keys(), IDR)
    if sidx <= 0:
        plt.xscale('log')
        plt.yscale('log')
    for rank in [False, True]:
        if rank:
            s1 = rankdata(s1)
            s2 = rankdata(s2)
            tail = '_rank_'
        else:
            tail = ''
        for i in range(3):
            start = [0, 1, 2][i]
            end = [1, 2, 101, 1001][i]
            col = ['black', 'grey', 'pink', 'red'][i]
            index = [idx for idx in range(len(i12)) if i12[idx] >= start and i12[idx] < end ]
            plt.scatter(list(map(lambda x: x+1, s1[index])), list(map(lambda x: x+1, s2[index])), c=col, label=str(start)+"~"+str(end-1))
        plt.legend(loc='upper left', numpoints=1)
        plt.savefig(self.options.ofile+'_score_idr_scatter_'+self.cnames[cidx]+"_"+str(self.options.dist_model)+tail+'.png')
        plt.clf()
