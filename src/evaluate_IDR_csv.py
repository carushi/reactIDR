import os.path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

# rfile = "/Users/cawa/Research/RiboSNitch/data/rRNA/rRNA_18S_ref.fa"
# ref = ''
# with open(rfile) as f:
#     ref = f.readlines()[1].rstrip('\n')


def plot_heatmap(file):
    lines = []
    with open(file) as f:
        lines = f.readlines()
    prefix = os.path.basename(file)
    transcripts = list(set([line.split('\t')[1] for line in lines]))
    for cond in ['cond', 'case']:
        for t in transcripts:
            fprefix = prefix+"_"+t+"_"+str(cond)
            tlines = [line.rstrip('\n').replace(':', '\t').split('\t') for line in lines if t in line]
            tlines = [contents for contents in tlines if contents[2] == str(cond)]
            if len(tlines) == 0:    break
            if len(tlines)-2 > 10:
                tlines = [contents for i, contents in enumerate(tlines) if i == 0 or i == 1 or (i-2)%max(1, math.floor((len(tlines)-2)/10)) == 0]
            row_labels = [contents[0] for contents in tlines][1:]
            step = 500
            skip = 3
            for i in range(0, len(tlines[0][skip:]), step):
                data = [list(map(float, contents[i+skip:i+step+skip])) for contents in tlines[1:]]
                df = pd.DataFrame(data=np.array(data, dtype=float))
                print(np.array(data).shape)
                # plt.imshow(df, aspect='auto')
                # plt.yticks(range(len(row_labels)), row_labels)
                # plt.colorbar(orientation='horizontal')
                # plt.savefig(t+"_"+str(i)+"_"+str(cond)+".pdf")
                # plt.clf()
            data = [list(map(float, contents[skip:])) for contents in tlines[1:]]
            df = pd.DataFrame(data=np.array(data, dtype=float))
            plt.imshow(df, aspect='auto')
            plt.yticks(range(len(row_labels)), row_labels)
            plt.colorbar(orientation='horizontal')
            plt.savefig(fprefix+"_full_length.pdf")
            plt.clf()

def calc_precision():


if __name__ == '__main__':
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            plot_heatmap(f)
    else:
        file = "idr_output_profile.csv"
        plot_heatmap(f)
    tfile = "temp.csv"
