import sys
import os
import argparse
from reactIDR.utility import *
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from scipy.stats import rankdata, kde

def moving_average(interval, window_size):
    return [np.nanmean(interval[i:min(len(interval), i+window_size)]) for i in range(0, len(interval)-window_size+1)]
    # window = np.ones(int(window_size))/float(window_size)
    # return np.convolve(interval, window, 'same')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='+', type=str, help="Input csv files (one or two are allowed)", metavar="INPUT")
    parser.add_argument("--output", dest="output", type=str, help="Output file prefix.", default="output", required=False)
    parser.add_argument("--idr", dest="idr", action="store_true", help="Add when an input is obtained by reactIDR with the idr output option.", required=False)
    parser.add_argument("--window", dest="window", type=int, help="Set a window size for averaging", default=5, required=False)
    parser.add_argument("--struct", dest="struct", type=float, help="Output secondary structures in dot-blacket style depending on the threshold.", metavar="THRESHOLD", required=False, default=None)
    parser.add_argument("--bed", dest="bed", type=str, help="File name prefix for bed files about a diversity area.", default="", required=False)
    parser.add_argument("--segmentation", dest="segmentation", action="store_true", help="Do not skip the first line.", required=False)

    parser.add_argument("--threshold", dest="threshold", type=float, help="The minimum difference to be written in bed files.", metavar="THRESHOLD", required=False, default=0.5)
    parser.add_argument("--keep_header", dest="header", action="store_false", help="Do not skip the first line.", required=False)
    parser.add_argument("--ignore", dest="ignore_ms", action="store_true", help="Ignore the data of transcripts on the minus strand.", required=False)
    parser.set_defaults()
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

    def print_difference(self, diff):
        for i, name in enumerate([fname.split('.')[0] for fname in self.arg.input]):
            vivo = vecs[i]
            EPS = 0.01
            ten = max(EPS, np.percentile([x for x in vivo if x == x], 90.0))
            quarter = max(EPS, np.percentile([x for x in vivo if x == x], 75.0))
            pvec = ['n' if x < quarter or x != x else 'q' if x < ten else 't' for x in vivo]
            if verbose:
                print(max([x for x in vivo]))
                print('quarter', name, ten, quarter)
                print(name, key, prefix, method, sample, " ".join(list(map(str, pvec))))
        ten = np.percentile([x for x in diff if x == x], 90.0)
        quarter = np.percentile([x for x in diff if x == x], 75.0)
        mten = np.percentile([x for x in diff if x == x], 10.0)
        mquarter = np.percentile([x for x in diff if x == x], 25.0)
        pvec = ['n' if x != x or y != y else 'tv' if d >= ten else 'qv' if d >= quarter else 'tt' if d <= mten else 'qt' if d <= mquarter else 'n' for x, y, d in zip(vecs[0], vecs[1], diff)]
        print('quarter diff', ten, quarter, mten, mquarter)
        print('diff', key, prefix, method, sample, " ".join(list(map(str, pvec))))
        print(len(pvec))
        print(len(vec))

    def extract_variable_region(self, method, x, y, sample, verbose=False):
        key = ''
        colors=['red', 'blue', 'black']
        vecs = []
        for temp in [p for p in [x, y] if p is not None]:
            key, lines = temp
            if "IDR" in method:
                vecs.append(list(map(one_minus_nan_float, lines[0][3].split(';'))))
            else:
                vecs.append(list(map(nan_float, lines[0][3].split(';'))))
        # vecs = [sample_data[i][3].split(';') for i in range(2)]
        max_y = -1
        min_y = 0
        for c in range(min(len(vecs), 3)):
            if c < 2 and c < len(vecs):
                vec = vecs[c]
                if self.arg.idr and "IDR" in method:
                    vec = [1.-x for x in vec]
            else:
                vec = [math.log(x/y) if x == x and y == y and x*y > 0.0 else float('nan') for x, y in zip(vecs[0], vecs[1])]
                if self.arg.idr and "IDR" in method:
                    vec = [-x for x in vec]
                if verbose:
                    self.print_difference(vec)
                break
            mov = moving_average(vec, self.arg.window)
            max_y = max(max_y, max(mov))
            min_y = min(min_y, min(mov))
            plt.plot(np.linspace(0, len(mov), len(mov)), mov, color=colors[c])
        min_y = min(min_y, max_y)
        if self.arg.window > 50:
            plt.ylim((min_y, max_y+max_y*0.1))
        else:
            plt.ylim((min_y, max_y+max_y*0.3))
        plt.legend([fname.split('.')[0] for fname in self.arg.input], loc='upper right')
        plt.savefig(self.arg.output+'_'+key+'_'+method+'_'+str(self.arg.window)+'_'+self.arg.output+'_'+sample+'_difference.pdf')
        plt.clf()

    def write_bed_files(self, prefix, method, x, y, sample, segment=False):
        thres = self.arg.threshold
        chrom = x[0]
        diff_list = []
        x_vec = ([line[3] for line in x[1] if line[0] == method and line[2] == sample])[0].split(';')
        y_vec = ([line[3] for line in y[1] if line[0] == method and line[2] == sample])[0].split(';')
        for i, (l, r) in enumerate(zip(x_vec, y_vec)):
            l, r = float(l), float(r)
            if l != l:  l = 0.0
            if r != r:  r = 0.0
            if self.arg.idr and "IDR" in method:
                l = 1.0-l
                r = 1.0-r
            diff = l-r
            # print(l, r, diff, diff_list)
            if abs(diff) >= thres:
                if (len(diff_list) == 0 or np.sign(diff) == np.sign(diff_list[0])):
                    diff_list.append(diff)
                    if segment:
                        continue
            if len(diff_list) > 0:
                chromstart = i-len(diff_list)
                chromend = i
                yield '\t'.join([x[0].rstrip('+').rstrip('-'), str(chromstart), str(chromend), '', str(np.mean(diff_list)), x[0][-1]])+'\n'
                diff_list = []
            if abs(diff) >= thres:
                diff_list.append(diff)
        if len(diff_list) > 0:
            chromstart = i-len(diff_list)
            chromend = i
            yield '\t'.join([prefix, str(chromstart), str(chromend), sample, str(np.mean(diff_list)), '+'])+'\n'

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
        for method in ["IDR-HMM-final", "noHMMIDR", "IDR", "count"]:
            for sample in ['case', 'cont']:
                if sample == 'cont' and method != 'count':
                    continue
                transcript = []
                append = False
                data = []
                if self.arg.struct is not None:
                    if "IDR" not in method: continue
                    self.seek_and_write_struct(method)
                    continue
                for input in self.arg.input:
                    data.append(self.read_data(method, input, sample))
                if len(data) == 0 or (len(data) == 1 and len(self.arg.bed) > 0):
                    continue
                if len(self.arg.bed) > 0:
                    if "IDR" in method: self.seek_and_write_bed_files(method, data, sample)
                else:
                    try:
                        while True:
                            x, y = data[0].__next__(), (data[1].__next__() if len(data) > 1 else None)
                            if len(data) > 1: assert x[0] == y[0]
                            if x[0][-1] == '-' and self.arg.ignore_ms:
                                continue
                            self.extract_variable_region(method, x, y, sample)
                    except StopIteration:
                        pass

    def seek_and_write_struct(self, method):
        for input in self.arg.input:
            data = []
            for t in ['case', 'cont']:
                data.append(self.read_data(method, input, t))
            self.print_struct(method, input.split('.')[0], data)

    def print_struct(self, method, head, data):
        for i in range(len(data)):
            for line in data[i]:
                name, x = line[0], line[1][0][3].split(';')
                if name[-1] == '-' and self.arg.ignore_ms:
                    continue
                vec = [float(p) for p in x]
                if self.arg.idr:
                    vec = [1.0-p for p in vec]
                print("#", name, ['case', 'cont'][i], method, head, )
                # print(name, data)
                print('>'+name)
                if i == 0:
                    print(''.join(['a' if p > self.arg.struct else '.' for p in vec]))
                else:
                    print(''.join(['s' if p > self.arg.struct else '.' for p in vec]))


    def seek_and_write_bed_files(self, method, data, sample):
        prefix = os.path.basename(self.arg.input[0])
        with open(self.arg.bed+'_'+method+'.bed', 'w') as f:
            try:
                while True:
                    x, y = data[0].__next__(), data[1].__next__()
                    assert x[0] == y[0]
                    if x[0][-1] == '-' and self.arg.ignore_ms:
                        continue
                    for line in self.write_bed_files(prefix, method, x, y, sample, self.arg.segmentation):
                        f.write(line)
            except StopIteration:
                pass

def plot_bar(options):
    dict_list = [get_dict(fname, options.header) for fname in options.input]
    all = len(dict_list)
    width = 1./all
    assert all <= 3
    colors=['black', 'red', 'blue']
    for transcript in dict_list[0]:
        max_y = -1
        min_y = 0
        for c, dict in enumerate(dict_list):
            max_value = len(dict[transcript])
            mov = moving_average(dict[transcript], options.window)
            max_y = max(max_y, max(mov))
            min_y = min(min_y, min(mov))
            plt.plot(np.linspace(0, max_value, max_value), mov, color=colors[c])
        min_y = min(min_y, max_y)
        if options.window > 50:
            plt.ylim((min_y, max_y+0.1))
        else:
            plt.ylim((min_y, max_y+0.3))
        plt.ylabel("Accessible class probability")
        plt.legend([fname.split('.')[0] for fname in options.input], loc='upper right')
        plt.savefig(transcript+'_'+str(options.window)+'_'+options.output+'_bar.pdf')
        plt.clf()

def print_struct(options):
    dict_list = [get_dict(fname, options.header) for fname in options.input]
    all = len(dict_list)
    width = 1./all
    assert all <= 3
    colors=['black', 'red', 'blue']
    for transcript in dict_list[0]:
        max_y = -1
        min_y = 0
        for c, dict in enumerate(dict_list):
            max_value = len(dict[transcript])
            mov = moving_average(dict[transcript], options.window)
            max_y = max(max_y, max(mov))
            min_y = min(min_y, min(mov))
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
    param = DiffVis(options)
    param.visualize_parameter()
