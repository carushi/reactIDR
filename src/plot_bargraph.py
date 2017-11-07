import sys
import os
import argparse
from utility import *
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from scipy.stats import rankdata, kde

def moving_average(interval, window_size):
    return [np.nanmean(interval[i:min(len(interval), i+window_size)]) for i in range(0, len(interval))]
    # window = np.ones(int(window_size))/float(window_size)
    # return np.convolve(interval, window, 'same')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='+', type=str, help="input csv", metavar="INPUT")
    parser.add_argument("--idr", dest="idr", action="store_true", help="Input is reactIDR output.", required=False)
    parser.add_argument("--header", dest="header", action="store_true", help="skip header.", required=False)
    parser.add_argument("--window", dest="window", type=int, help="window size for averaging", default=5, required=False)
    parser.add_argument("--output", dest="output", type=str, help="output name", default="out", required=False)
    parser.add_argument("--bed", dest="bed", type=str, help="File name prefix for bed files about a diversity area.", default="", required=False)
    # parser.add_argument("--case", dest="case", type=str, help="read replicate data from FILE1, FILE2, ...", metavar="CASE", required=True)
    # parser.add_argument("--control", dest="control", type=str, help="read replicate data from FILE1, FILE2, ...", metavar="CONTROL", required=False)
    # # parser.add_argument("--coverage", dest="coverage", nargs='+', type=str, help="use the ratio of read enrichment based on coverage", metavar="COVERAGE", required=False)
    # parser.add_argument("-n", "--sample", dest="sample_size", type=int, help="sample size for test (default: all)", default=-1, metavar="SAMPLESIZE", required=False)
    parser.set_defaults(idr=False)
    return parser

def one_minus_nan_float(x):
    if x == 'nan' or x == 'None' or x == 'Nan':
        return float('nan')
    else:
        return 1.0-float(x)


def nan_float(x):
    if x == 'nan' or x == 'None' or x == 'Nan':
        return float('nan')
    else:
        return float(x)

def get_dict_idr(input, header=False):
    lines = []
    with open(input) as f:
        start = 0 if not header else 1
        for line in f.readlines()[start:]:
            lines.append[line.rstrip('\n').split('\t')]
    lines = [line for line in lines if line[0] != "type"]
    return lines

def get_dict(input, header=False):
    dict = {}
    with open(input) as f:
        start = 0 if not header else 1
        for line in f.readlines()[start:]:
            contents = line.rstrip('\n').split('\t')
            if len(contents) == 2:
                name, data = contents
                data = list(map(nan_float, data.split(';')))
            else:
                name, data = contents[0], contents[1:]
                # print(data[1:])
                data = np.nanmean([[nan_float(x) for x in tdata.split(';')] for tdata in data], axis=0)
            dict[name] = data
    return dict

class DiffVis:
    """docstring for DiffVis"""
    def __init__(self, arg):
        self.arg = arg

    def extract_variable_region(self, prefix, method, x, y, sample):
        key = ''
        colors=['red', 'blue', 'black']
        vecs = []
        for temp in [x, y]:
            key, lines = temp
            if "IDR" in method:
                vecs.append(list(map(one_minus_nan_float, lines[0][3].split(';'))))
            else:
                vecs.append(list(map(nan_float, lines[0][3].split(';'))))
        # vecs = [sample_data[i][3].split(';') for i in range(2)]
        max_y = -1
        min_y = 0
        for c in range(3):
            if c < 2:
                vec = vecs[c]
            else:
                vec = [math.log(x/y) if x == x and y == y and x*y > 0.0 else float('nan') for x, y in zip(vecs[0], vecs[1])]
                for i, name in enumerate(['vivo', 'vitro']):
                    vivo = vecs[i]
                    EPS = 0.01
                    ten = max(EPS, np.percentile([x for x in vivo if x == x], 90.0))
                    quarter = max(EPS, np.percentile([x for x in vivo if x == x], 75.0))
                    print(max([x for x in vivo]))
                    print('quarter', name, ten, quarter)
                    pvec = ['n' if x < quarter or x != x else 'q' if x < ten else 't' for x in vivo]
                    print(name, key, prefix, method, sample, " ".join(list(map(str, pvec))))
                diff = [math.log((x+EPS)/(y+EPS)) if x == x and y == y and x*y > 0.0 else float('nan') for x, y in zip(vecs[0], vecs[1])]
                ten = np.percentile([x for x in diff if x == x], 90.0)
                quarter = np.percentile([x for x in diff if x == x], 75.0)
                mten = np.percentile([x for x in diff if x == x], 10.0)
                mquarter = np.percentile([x for x in diff if x == x], 25.0)
                print('quarter diff', ten, quarter, mten, mquarter)
                pvec = ['n' if x != x or y != y else 'tv' if d >= ten else 'qv' if d >= quarter else 'tt' if d <= mten else 'qt' if d <= mquarter else 'n' for x, y, d in zip(vecs[0], vecs[1], diff)]
                # pvec = ['n' if x != x or y != y or x == y else 'v' if x > y else 't' for x, y in zip(vecs[0], vecs[1])]
                print('diff', key, prefix, method, sample, " ".join(list(map(str, pvec))))
                print(len(pvec))
                print(len(vec))
                break
            max_value = len(vec)
            mov = moving_average(vec, self.arg.window)
            max_y = max(max_y, max(mov))
            min_y = min(min_y, min(mov))
            plt.plot(np.linspace(0, max_value, max_value), mov, color=colors[c])
        min_y = min(min_y, max_y)
        if self.arg.window > 50:
            plt.ylim((min_y, max_y+max_y*0.1))
        else:
            plt.ylim((min_y, max_y+max_y*0.3))
        plt.legend([fname.split('.')[0] for fname in self.arg.input], loc='upper right')
        plt.savefig(key+'_'+method+'_'+str(self.arg.window)+self.arg.output+'_'+sample+'_difference.pdf')
        plt.clf()

    def write_bed_files(self, prefix, method, x, y, sample, thres):
        chrom = x[0]
        diff_list = []
        x_vec = ([line[3] for line in x[1] if line[0] == method and line[2] == sample])[0].split(';')
        y_vec = ([line[3] for line in x[1] if line[0] == method and line[2] == sample])[0].split(';')
        for i, (l, r) in enumerate(zip(x_vec, y_vec)):
            l, r = float(l), float(r)
            if l != l:  l = 0.0
            if r != r:  r = 0.0
            diff = l-r
            print(l, r, diff, diff_list)
            if abs(diff) >= thres:
                if (len(diff_list) == 0 or np.sign(diff) == np.sign(diff_list[0])):
                    diff_list.append(diff)
                    continue
            if len(diff_list) > 0:
                chromstart = i-len(diff_list)
                chromend = i
                yield '\t'.join([prefix, chromstart, chromend, '', mean(diff_list), '+'])+'\n'
                diff_list = []
            if abs(diff) >= thres:
                diff_list.append(diff)
        if len(diff_list) > 0:
            chromstart = i-len(diff_list)
            chromend = i
            yield '\t'.join([prefix, chromstart, chromend, sample, mean(diff_list), '+'])+'\n'

    def read_data(self, method, fname, sample):
        data = []
        name = ""
        with open(fname) as f:
            while True:
                line = f.readline()
                if line == "":
                    break
                contents = line.rstrip('\n').split('\t')
                if method != contents[0] or contents[2] != sample:
                    continue
                if name != contents[1]:
                    if len(name) > 0:
                        # print(name, data)
                        yield name, data
                    name = contents[1]
                    data = []
                data.append(contents)
        if len(name) > 0:
            yield name, data


    def visualize_parameter(self):
        lines = []
        for method in ["IDR-HMM-final", "count"]:
            if method == 'count':
                continue
            thres = 0.5
            for sample in ['case', 'cont']:
                if sample != 'cont':
                    continue
                transcript = []
                prefix = os.path.basename(self.arg.input[0])
                append = False
                data = []
                for input in self.arg.input:
                    data.append(self.read_data(method, input, sample))
                # print(data)
                try:
                    if len(self.arg.bed) > 0:
                        with open(self.arg.bed+'_'+method+'.bed', 'w') as f:
                            while True:
                                x, y = data[0].__next__(), data[1].__next__()
                                assert x[0] == y[0]
                                if x[0][-1] == '-':
                                    continue
                                for line in self.write_bed_files(prefix, method, x, y, sample, thres):
                                    f.write(line)
                    else:
                        while True:
                            x, y = data[0].__next__(), data[1].__next__()
                            assert x[0] == y[0]
                            if x[0][-1] == '-':
                                continue
                            self.extract_variable_region(prefix, method, x, y, sample)
                except StopIteration:
                    pass
                else:
                    pass

def plot_bar(options):
    if options.idr:
        param = DiffVis(options)
        param.visualize_parameter()
    else:
        dict_list = [get_dict(fname, options.header) for fname in options.input]
        all = len(dict_list)
        width = 1./all
        assert all <= 3
        colors=['black', 'red', 'blue']
        for transcript in dict_list[0]:
            max_y = -1
            min_y = 0
            ps = []
            for c, dict in enumerate(dict_list):
                max_value = len(dict[transcript])
                mov = moving_average(dict[transcript], options.window)
                max_y = max(max_y, max(mov))
                min_y = min(min_y, min(mov))
                # ps.append(plt.bar(np.linspace(0, len(dict[transcript]), len(dict[transcript]))+width*c, dict[transcript], color=str(c/float(all)), width=width))
                plt.plot(np.linspace(0, max_value, max_value), mov, color=colors[c])
            min_y = min(min_y, max_y)
            if options.window > 50:
                plt.ylim((min_y, max_y+0.1))
            else:
                plt.ylim((min_y, max_y+0.3))
            plt.legend([fname.split('.')[0] for fname in options.input], loc='upper right')
            plt.savefig(transcript+'_'+str(options.window)+options.output+'_bar.pdf')
            plt.clf()


if __name__ == '__main__':
    parser = get_parser()
    options = parser.parse_args()
    plot_bar(options)
