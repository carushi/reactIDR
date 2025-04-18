import sys
import os.path
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from scipy.stats import rankdata, kde
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.model_selection import cross_val_score
import seaborn as sns
from seaborn import JointGrid
import pandas as pd
import copy
from reactIDR.AUC import *
from collections import Counter
import argparse

REMOVE = True
def scale_vector(vec):
    max_vec, min_vec = np.nanmax(vec), np.nanmin(vec)
    return [float(x-min_vec)/(max_vec-min_vec) if x == x else float('nan') for x in vec ]

def parse_score_tri_iterator(file, header=True):
    with open(file) as f:
        while True:
            line = f.readline()
            if line == "":  break
            if header or ';' not in line:
                header = False
                continue
            contents = line.rstrip('\n').split('\t')
            key = contents[0].split('|')[0]
            if '|' in contents[0]:  key += contents[0].split('|')[-1]
            yield key, [list(map(float, score.split(';'))) for score in contents[1:]]

def read_data_pars_format_multi(fnames, verbose=True):
    data = []
    count = 0
    for index in range(len(fnames)):
        if len(fnames[index]) == 0:
            continue
        data.append([])
        for key, tdata in parse_score_tri_iterator(fnames[index]):
            count += 1
            if count%1000 == 0 and verbose:
                print("# Reading lines (%s lines)..." % count)
            if len(data[index]) == 0:
                data[index] = [{chr(0):[0]} for t in tdata]
            for i, temp in enumerate(tdata):
                data[index][i][key] = temp
    return data

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def remove_nan(X, Y):
    index = [i for i in range(len(Y)) if len([ j for j in range(X.shape[0]) if not np.isnan(X[j,i])]) == X.shape[0]]
    X = [[x[i] for i in index] for x in X]
    Y = [Y[i] for i in index]
    # else:
    #     print(X)
    #     X, Y = list([(x, y) for x, y in zip(X, Y) if not np.isnan(x)])
    return X, Y

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='+', type=str, help="input csv", metavar="INPUT")
    parser.add_argument("--parameter", dest="parameter", action="store_true", help="Visualize parameters", required=False)
    parser.add_argument("--fix", dest="fix", action="store_true", help="To fix y-axis.", required=False)
    parser.add_argument("--reverse", dest="reverse", action="store_true", help="Consider case and control in the opposite way. (Stem enriched in case)", required=False)
    # parser.add_argument("--filter", dest="filter", action="store_true", help="Compute AUC with filtering dataset according to IDR.", required=False)
    parser.add_argument("--auc", dest="auc", action="store_true", help="Compute AUC.", required=False)
    parser.add_argument("--covauc", dest="covauc", action="store_true", help="Compute AUC with coverage-based filtering.", required=False)
    parser.add_argument("--mauc", dest="mauc", action="store_true", help="Compute AUC with reactivities from multiple dataset.", required=False)
    parser.add_argument("--score", dest="score_file", nargs='+', type=str, help="Score file such as global PARS, icshape score (should include 'pars' or 'icshape' in the name)", required=False)
    parser.add_argument("--scatter", dest="scatter", action="store_true", help="Draw scatter plot.", required=False)
    parser.add_argument("--dscatter", dest="dscatter", action="store_true", help="Draw scatter plot.", required=False)
    parser.add_argument("--annotation", dest="annotation", type=str, help="GTF file to output bed files", required=False)
    parser.add_argument("--case", dest="case", type=str, help="Count file for case.", required=False)
    parser.add_argument("--cont", dest="cont", type=str, help="Count file for cont.", required=False)
    parser.add_argument("--dir", dest="dir", type=str, help="Directory for count data.", required=False)
    parser.set_defaults(score_file=[], case="", cont="", dir="", annotation="", covauc=False, mauc=False)
    return parser

def heatmap_routine(df, row_labels, fname):
    plt.imshow(df, aspect='auto', interpolation='nearest')
    plt.yticks(range(len(row_labels)), row_labels)
    plt.colorbar(orientation='horizontal')
    plt.savefig(fname)
    plt.clf()

def normalized_vec(vec):
    temp = [nan_float(x) for x in vec]
    min_t, max_t = min(temp), max(temp)
    return [ (float(x)-min_t)/max(0.0000000001, max_t-min_t) if x != "nan" else float('nan') for x in vec]

def get_lines(input, skip, sep):
    lines = []
    with open(input) as f:
        lines = [ line.rstrip('\n').split('\t') for line in f.readlines() ]
    prefix = os.path.basename(input)
    both_flag = False
    if len([line for line in lines if line[2] == 'cont' and line[0] != 'score']) > 0 and \
       len([line for line in lines if line[2] == 'case' and line[0] != 'score']) > 0:
        both_flag = True
    lines = [line[:skip]+line[skip].split(sep) for line in lines]
    # for i in range(len(lines)):
    #     print(lines[i][0:10])
    return lines, prefix, both_flag

def get_sample_dict(tlines, skip):
    dict = [{}, {}]
    order = {'case': 0, 'cont': 1}
    for contents in tlines:
        dict[order[contents[2]]][contents[0]] =  list(map(nan_float, contents[skip:]))
    return dict

def append_integrated_transcript(lines, skip, seta=None, minus_strand=False, boundary=False):
    if seta is None:
        trans_list = list(set([contents[1] for contents in lines if contents[0] != "score"]))
        if minus_strand:
            trans_list = [x for x in trans_list if x[-1] != '-']
    else:
        trans_list = list(set(seta))
    score_list = list(set([contents[0] for contents in lines]))
    all_concat = []
    for score_type in score_list:
        for sample in ['case', 'cont']:
            tlines = [line for line in lines if line[0] == score_type and line[2] == sample]
            if not all([x in [contents[1] for contents in tlines] for x in trans_list]):
                continue
            for flag_18S, name in zip([True, False], ["all_unknown", "all_concat"]):
                concat_data = []
                for x in trans_list:
                    if flag_18S and x == "RNA18S5+":    continue
                    index = [i for i in range(len(tlines)) if tlines[i][1] == x]
                    assert len(index) == 1
                    concat_data.extend(tlines[index[0]][skip:])
                    if boundary:
                        concat_data.extend([float("-inf")])
                # print(score_type, sample, len(concat_data))
                if len(concat_data) > 0:
                    all_concat.append([score_type, name, sample]+concat_data)
    # print([len(x) for x in all_concat], all_concat)
    return trans_list, all_concat

def append_integrated_transcript_count(lines, skip, seta=None, boundary=False):
    return append_integrated_transcript(lines, skip, seta=seta, boundary=boundary)

class AccEval:
    """docstring for AccEval"""
    def __init__(self, arg):
        self.sep=";"
        self.skip = 3 # first index of probability
        self.blty={'count': 'o', 'IDR-HMM-final': '--', 'IDR':'-.', 'score':'-', 'BUMHMM':':', 'noHMMIDR':'-.', 'PROBer':'x'}
        self.lty={'count': '--', 'IDR-HMM-final': '-', 'IDR':'-', 'score':'--', 'BUMHMM':':', 'noHMMIDR':'-', 'PROBer': '-.'}
        self.color={'count': '0', 'IDR-HMM-final': 'red', 'IDR': 'violet', 'score':'b', 'BUMHMM':'g', 'noHMMIDR': 'orange', 'PROBer': 'gray'}
        # , '.', ',', 'o', 'v', '^', '<', '>', 's']
        self.comp_sample = False
        self.arg = arg

    def convert_negative(self, score, negative):
        score = list(map(lambda x: float("nan") if x == "None" else float(x), score.split(self.sep)))
        if negative:
            score = [-x for x in score]
        return score
        # return (self.sep).join(map(str, score))


    def append_simple_score(self, index=0): # reading converted scores
        lines = []
        if self.arg.score_file is None or len(self.arg.score_file) == 0:
            return lines
        with open(self.arg.score_file[index]) as f:
            pars_flag = ("pars" in self.arg.score_file)
            head = ["cont", "case"]
            for line in f.readlines():
                if len(line) > 0 and line[0] != "#":
                    transcript, score = line.rstrip('\n').split('\t')
                    lines.append(["score", transcript, head[1]]+self.convert_negative(score, pars_flag))
                    lines.append(["score", transcript, head[0]]+self.convert_negative(score, not pars_flag))
        return lines

    def reduce_dataset(self, tlines):
        primary_type = ['type', 'ref', 'count', 'IDR', 'score', 'IDR-HMM_0', 'IDR-HMM-final', 'BUMHMM', 'PROBer']
        index = [i for i, contents in enumerate(tlines) if contents[0] in primary_type]
        return index

    def get_lines(self, index=0):
        lines, prefix, both = get_lines(self.arg.input[index], self.skip, self.sep)
        if both:    self.comp_sample = both
        if len([contents[0] for contents in lines if contents[0] == 'score']) == 0:
            lines.extend(self.append_simple_score(index))
        return lines, prefix

    def plot_heatmap(self):
        if self.arg.mauc:
            results = [self.get_lines(i) for i in range(len(self.arg.input))]
            for i, (lines, prefix) in enumerate(results):
                trans_list, alines = append_integrated_transcript(lines, self.skip, ["RNA18S5+", "RNA5-8S5+", "RNA28S5+", "ENSG00000201321|ENST00000364451+"])
                results[i] = (lines+alines, prefix)
            lines, prefix = results[0][0], results[0][1]
            #print(prefix, lines)
        else:
            lines, prefix = self.get_lines()
            if self.arg.auc or self.arg.covauc:
                trans_list, alines = append_integrated_transcript(lines, self.skip, ["RNA18S5+", "RNA5-8S5+", "RNA28S5+", "ENSG00000201321|ENST00000364451+"])
                lines += alines
                if self.arg.covauc:
                    clines = self.append_count_score(self.arg.case, self.arg.cont)
                    clines += append_integrated_transcript_count(clines, self.skip, trans_list)[1]
        for trans in list(set([contents[1] for contents in lines])):
            fprefix = prefix+"_"+trans
            print(trans)
            if trans != 'all_unknown' and trans != 'RNA18S5+':
                continue
            data = [contents for contents in lines if contents[1] == trans]
            if len(data) == 0:
                continue
            dict_count = get_sample_dict(data, self.skip)
            if 'type' not in dict_count[0]:
                continue
            if self.arg.dscatter:
                self.plot_double_scatter(trans, dict_count, fprefix)
            elif self.arg.auc:
                self.plot_all_auc(trans, dict_count, fprefix)
            elif self.arg.covauc:
                cdata = [contents for contents in clines if contents[1] == trans]
                self.comp_cov_auc(trans, dict_count, fprefix, cdata)
            elif self.arg.mauc:
                if trans == 'all_unknown':
                    cdata = [get_sample_dict([contents for contents in clines if contents[1] == trans], self.skip) for clines, prefix in results]
                    self.comp_multi_auc(trans, fprefix, cdata, [prefix for clines, prefix in results])
            else:
                self.plot_single_heatmap(trans, dict_count, fprefix)
                if self.comp_sample:
                    self.plot_double_heatmap(trans, dict_count, fprefix)
                    self.plot_count_heatmap(trans, dict_count, fprefix)

    def all_data_mat(self, dict_count, normalize=True):
        mat, row_labels = [], []
        if len(dict_count) == 0:
            return mat, row_labels
        keys = list(dict_count[0].keys())+list(dict_count[1].keys())
        keys = [key for key in ['ref', 'count', 'score', 'IDR', 'IDR-HMM-final', 'BUMHMM', 'PROBer'] if key in keys]
        if len(keys) == 1 and keys[0] == 'score':
            return [], []
        for i, key in enumerate(keys):
            if key not in dict_count[0] and key not in dict_count[1]:
                keys.remove(key)
                continue
            for j in [0, 1]:
                if key not in dict_count[j]:
                    continue
                score = dict_count[j][key]
                if key == 'count':
                    if normalize:
                        max_count = max([math.log10(float(x)+1) for x in score if x != "nan"])
                        mat.append([ math.log10(float(x)+1)/max_count if x != "nan" else 0. for x in score])
                    else:
                        mat.append([ log_count(x) for x in score ])
                elif key == 'score':
                    temp = ([float(x) for x in score if x != "nan"])
                    max_count, min_count = max(temp), min(temp)
                    if max_count-min_count > 0:
                        print("no data", key, i, j, min_count, max_count)
                    mat.append([ (float(x)-min_count)/max(0.0000000001, max_count-min_count) if x != "nan" else 0. for x in score])
                elif key != 'type':
                    mat.append(list(map(nan_float, score)))
                else:
                    continue
        row_labels = keys
        return mat, row_labels

    def plot_count_heatmap(self, tname, dict_count, prefix):
        data = [{'count': tdict['count']} if 'count' in tdict else {} for tdict in dict_count ]
        mat, row_labels = self.all_data_mat(data, normalize=False)
        row_labels = ['case', 'cont']
        if len(mat) == 0:
            return
        df = pd.DataFrame(data=np.array(mat, dtype=float))
        heatmap_routine(df, row_labels, prefix+"_count.pdf")

    def plot_single_heatmap(self, tname, dict_count, prefix):
        for i, cond in enumerate(['case', 'cont']):
            if len(dict_count[i]) == 0:
                continue
            fprefix = prefix+"_"+str(cond)
            tdict_count = [{}, {}]
            tdict_count[i] = dict_count[i]
            mat, row_labels = self.all_data_mat(tdict_count)
            if len(mat) == 0:
                continue
            df = pd.DataFrame(data=np.array(mat, dtype=float))
            heatmap_routine(df, row_labels, fprefix+cond+"_full.pdf")

    def set_rgb_data(self, tlines, row_labels):
        length = len(tlines[0][self.skip:])
        data = np.zeros(shape=(len(row_labels), length, 3))
        # for i in range(2, len(tlines), 2): #[cont, case], [cont, case], ...
        for i in range(0, len(tlines), 2): #[case, cont], [case, cont], ...
            for j in range(length):
                x = int(i/2)
                if row_labels[int(i/2)] == 'ref':
                    prob = float(tlines[i][j+self.skip])
                    if prob == 0.0:
                        data[x,j,0] = 1.0
                    elif prob == 1.0:
                        data[x,j,2] = 1.0
                    else:
                        data[x,j,0] = prob
                        data[x,j,2] = prob
                else:
                    data[x,j,0] = 1.0-float(tlines[i+1][j+self.skip])
                    data[x,j,2] = 1.0-float(tlines[i][j+self.skip])
                    # data[x,j,1] = 1.0-data[x,j,0]+1.0-data[x,j,2]
        return data

    def get_pos(self, sample_cond, name):
        if sample_cond == 'case':
            pos = 0
            if self.arg.reverse and name != 'score': pos = 1
        else:
            pos = 1
            if self.arg.reverse and name != 'score': pos = 0
        return pos

    def plot_boxplot_class(self, dict_count, prefix):
        for sp in ["all", "canonical"]:
            answer = set_positive_negative(dict_count[0]['ref'], sp)
            fig=plt.figure(1)
            count = 0
            # for score_type in ["count", "score", "IDR", "IDR-HMM-final", "BUMHMM", "noHMMIDR", 'PROBer']:
            for score_type in ["count", "score", "IDR-HMM-final"]:
                all_data = []
                row_labels = []
                for i, sample in enumerate(["case", "cont"]):
                    if score_type == "score" and sample == "cont":  continue
                    if score_type not in dict_count[i]:
                        continue
                    pos = self.get_pos(sample, score_type)
                    if score_type == "score":
                        pred = [float(val) for val in dict_count[i][score_type] if val == val]
                    elif score_type == "count":
                        pred = [log_count(val) for val in dict_count[i][score_type]]
                    else:
                        pred = [one_minus_nan_float(val) if "IDR" in score_type else nan_float(val) for val in dict_count[i][score_type]]
                    # pred = normalized_vec(pred)
                    if len(pred) != len(answer):
                        print('Length error:', score_type)
                        print('score length=', len(answer), score_type+" length=", len(pred))
                        return
                    all_data.append([pred[i] for i in range(len(pred)) if answer[i] == pos and pred[i] == pred[i]])
                    all_data.append([pred[i] for i in range(len(pred)) if answer[i] != pos and pred[i] == pred[i]])
                    row_labels.extend([sample+"_"+["a", "s"][i]+"\n"+score_type[0:7], sample+"_"+["s", "a"][i]+"\n"+score_type[0:7]])
                plt.figure()
                # print(prefix, score_type, len(all_data), len(row_labels), row_labels)
                if len(all_data) == 0:
                    continue
                assert len(all_data) == len(row_labels)
                plt.boxplot(all_data, labels=row_labels, showmeans=True)
                plt.savefig(prefix+"_"+sp+"_"+score_type+"_boxplot_idr.pdf")
                plt.clf()

    def compute_and_plot_curves(self, pos, answer, pred, score_type, prefix, sample, curve, top, fig, assym):
        assert "assym" not in prefix or assym
        if curve == "auc":
            tpr, fpr, auc = calc_tp_fp(answer, pred, pos, assym=assym)
            xlab, ylab = "1-Specificity", "Sensitivity"
        else:
            tpr, fpr, auc = calc_precision_recall(answer, pred, pos, assym=assym)
            xlab, ylab = "Recall", "Precision"
        print(curve, score_type, prefix, sample, auc, len([x for x in answer if x == pos]), len([x for x in answer if x == 1-pos]))
        if top:
            if curve == 'prc' and min([x for x in fpr if x > 0.0]) > 0.1: # no points in selected region
                index = [i  for i, x in enumerate(fpr) if x == min([x for x in fpr if x > 0.0])][0]
                fig.add_subplot(111).plot((-1, 1), (tpr[index], tpr[index]), ':', color=self.color[score_type], label=score_type)
            else:
                fig.add_subplot(111).plot(fpr, tpr, 'o', markersize=5, color=self.color[score_type], label=score_type, markeredgewidth=0)
        else:
            fig.add_subplot(111).plot(fpr, tpr, 'o', markersize=3, color=self.color[score_type], label=score_type, markeredgewidth=0)
        fig.add_subplot(111).set_xlabel(xlab)
        fig.add_subplot(111).set_ylabel(ylab)
        return fig, min(tpr)


    def calc_all_auc(self, dict_count, prefix, curve="auc", top=False):
        if top:
            prefix = prefix+"_top"
        loc = 2
        if curve == "prc":
            if top:
                loc = 4
            else:
                loc = 3
        for sp in ["all", "canonical"]:
            if sp == 'canonical':
                break
            answer = set_positive_negative(dict_count[0]['ref'], sp)
            for i, sample in enumerate(["case", "cont"]):
                for assym in [True, False]:
                    if assym and sample == "case":  continue
                    if top:
                        fig = plt.figure(1)
                        fig.set_size_inches((6, 9))
                    else:
                        fig = plt.figure(1)
                        fig.set_size_inches((8, 6))
                    fprefix = prefix+"_"+sp+sample+("_assym" if assym else "")
                    ymin = 1
                    for score_type in ["count", "score", "IDR-HMM-final", "noHMMIDR", "BUMHMM", 'PROBer']:
                        if score_type not in dict_count[i]:
                            print("No data!", score_type, file=sys.stderr)
                            continue
                        pos = self.get_pos(sample, score_type)
                        pred = [one_minus_nan_float(val) if "IDR" in score_type else nan_float(val) for val in dict_count[i][score_type]]
                        # if method == 'score_type == 'noHMMIDR':
                        if 'all_unknown' in prefix and curve == 'auc' and (assym or sample == "case"):
                            tpred, tanswer = pred_conversion_assym(pred, True, False, pos, answer)
                            #print("pred", score_type, sample, pos, ","+",".join(map(str, tpred)))
                            #print("answer", score_type, sample, pos, ","+",".join(map(str, tanswer)))
                        if curve == "auc":
                            tpr, fpr, auc = calc_tp_fp(answer, pred, pos, assym=assym)
                            xlab, ylab = "1-Specificity", "Sensitivity"
                        #self.check_print_debug(fprefix, score_type, pred, pred, answer)
                        fig, tymin = self.compute_and_plot_curves(pos, answer, pred, score_type, fprefix, sample, curve, top, fig, assym)
                        ymin = min(ymin, tymin)
                    if top:
                        fig.add_subplot(111).set_ylim([max(-0.01, ymin-0.1), 1.01])
                        fig.add_subplot(111).set_xlim([-0.001, 0.1])
                    else:
                        fig.add_subplot(111).set_ylim([-0.01, 1.01])
                        fig.add_subplot(111).set_xlim([-0.01, 1.01])
                    fig.add_subplot(111).legend(loc=loc, numpoints=1)
                    fig.savefig(fprefix+"_"+curve+"_idr.pdf")
                    fig.clf()


    def calc_ratio_auc(self, dict_count, prefix):
        length = len(dict_count[0]['ref'])
        for sp in ["all", "canonical"]:
            answer = set_positive_negative(dict_count[0]['ref'], sp)
            fig=plt.figure(1)
            fig.set_size_inches((8, 6))
            count = 0
            for score_type in ["count", "IDR", "IDR-HMM-final"]:
                for sample in ['case']:
                    pos = self.get_pos(sample, score_type)
                    if score_type not in dict_count[0] or score_type not in dict_count[1]:
                        continue
                    pred = compute_ratio(dict_count[0][score_type], dict_count[1][score_type], ("IDR" in score_type))
                    if sample == 'cont':
                        pred = [-x for x in pred]
                    tpr, fpr, auc = calc_tp_fp(answer, pred, pos)
                    print("AUC_ratio", score_type, prefix, sample, auc)
                    if pos == 1: fig.add_subplot(111).plot(fpr, tpr, color=self.color[score_type], lw=1, label=score_type)
                    else: fig.add_subplot(111).plot(fpr, tpr, color=self.color[score_type], lw=1, label=score_type)
                count += 1
            fig.add_subplot(111).set_ylim([0, 1])
            fig.add_subplot(111).set_xlim([0, 1])
            fig.add_subplot(111).legend(loc=2)
            fig.savefig(prefix+"_"+sp+"_auc_ratio_idr.pdf")
            fig.clf()

    def plot_surface_auc(self, data, fname):
        X, Y, Z = data
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        # xlabel("case threshold")
        # ylabel("cont threshold")
        # surf.zlab("AUC")
        plt.savefig(fname)
        plt.clf()

    def plot_line_auc(self, data, fname):
        X, Y1, Y2 = data
        fig, ax1 = plt.subplots()
        ax1.plot(X, Y1, '-')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('AUC', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(X, Y2, 'r')
        ax2.set_ylabel('Positive dataset', color='r')
        ax2.tick_params('y', colors='r')
        # fig.legend(loc=2)
        fig.tight_layout()
        plt.savefig(fname)
        plt.clf()

    def check_print_debug(self, prefix, name, vec, score, answer):
        index = [i for i in range(len(score)) if score[i] == score[i]][0:10]
        print(prefix, name, [vec[x] for x in index], [answer[x] for x in index], file=sys.stderr, sep="\t")

    def calc_filtered_auc(self, dict_count, prefix, sp, sample, coverage, curve):
        index = {'case':0, 'cont':1}[sample]
        step = 10
        prange = range(0, 100, step)
        answer = set_positive_negative(dict_count[0]['ref'], sp)
        coverage_sorted = sorted(coverage)
        # print(len(coverage))
        # print([math.floor(percent/100.*len(coverage)) for percent in prange])
        thresholds = [coverage_sorted[math.floor(percent/100.*len(coverage))] for percent in prange]
        assym = False
        fig = plt.figure(1)
        fig.set_size_inches((8, 6))
        fprefix = prefix+"_"+sp+sample+("_assym" if assym else "")
        ymin = 1
        for filter_type in ['upper', 'lower']:
            for score_type in ["count", "score", "IDR", "IDR-HMM-final", "noHMMIDR", "BUMHMM", 'PROBer']:
                if score_type not in dict_count[index]:
                    print("No data!", score_type, file=sys.stderr)
                    continue
                pos = self.get_pos(sample, score_type)
                pred = [one_minus_nan_float(val) if "IDR" in score_type else nan_float(val) for val in dict_count[index][score_type]]
                for i, thres in enumerate(thresholds):
                    if filter_type == 'upper':
                        ind_list = [h for h in range(len(coverage)) if coverage[h] >= thres]
                    else:
                        ind_list = [h for h in range(len(coverage)) if coverage[h] <= thres]
                    # print(thres, len(ind_list), ind_list)
                    fanswer, fpred = [list(x) for x in zip(*[[answer[i], pred[i]] for i in ind_list])]
                    tpr, fpr, auc = calc_tp_fp(fanswer, fpred, pos, assym=assym)
                    print('fauc', filter_type, prefix, sp, sample, score_type, len(fpred), i*step, thres, auc)


    def plot_all_auc(self, tname, dict_count, prefix):
        if 'ref' in dict_count[0]:
            print(dict_count[0].keys())
            #if 'count' in dict_count[0]:    print(dict_count[0]['count'])
            #if 'count' in dict_count[1]:    print(dict_count[1]['count'])
            self.calc_all_auc(dict_count, prefix, curve='auc')
            self.calc_all_auc(dict_count, prefix, curve='prc')
            self.calc_all_auc(dict_count, prefix, curve='prc', top=True)
            self.plot_boxplot_class(dict_count, prefix)
            #if self.comp_sample:
            #    self.calc_ratio_auc(dict_count, prefix)

    def comp_cov_auc(self, tname, dict_count, prefix, cdata):
        print(tname, 'ref' in dict_count[0])
        if 'ref' not in dict_count[0]:
            return
        for sp in ["all", "canonical"]:
            for sample in ['case', 'cont']:
                coverage = np.sum([list(map(nan_float, x[3:])) for x in cdata if x[2] == sample], axis=0)
                self.calc_filtered_auc(dict_count, prefix, sp, sample, coverage, curve='auc')

    def comp_multi_auc(self, tname, prefix, dict_list, prefix_list):
        if 'ref' not in dict_list[0][0]:
            return
        sp, curve ='all', 'auc'
        index = {'case':0, 'cont':1}
        original_sample_list = ['cont' if 'pars' in p else 'case' for p in prefix_list]
        sample = 'case'
        criteria = '_2d'
        answer = set_positive_negative(dict_list[0][0]['ref'], sp)
        assym = True
        for auc_method, func in zip(['max', 'min', 'average', 'no'], [np.max, np.min, np.mean, lambda x: x]):
            if auc_method != 'no':
                fig = plt.figure(1)
                fig.set_size_inches((8, 6))
            else:
                break
            simple = True
            if criteria == '':  simple = False
            fprefix = prefix+"_"+sp+sample+("_assym" if assym else "")
            if simple:
                fprefix += "_sim"
            for score_type in ["count", "score", "IDR", "IDR-HMM-final", "noHMMIDR", "BUMHMM", 'PROBer']:
                if score_type not in dict_list[0][0]:
                    print("No data!", score_type, file=sys.stderr)
                    continue
                if score_type in ["count", "IDR", "IDR-HMM-final"]:
                    sample_list = original_sample_list
                else:
                    sample_list = ['case' for i in range(len(original_sample_list))]
                pred_list = [ dict_count[index[tsample]][score_type] for dict_count, tsample in zip(dict_list, sample_list) ]
                header = " ".join([auc_method, prefix, sp, sample, score_type])
                # print(score_type, len(pred_list), len(sample_list), print(prefix_list))
                if score_type in ['count', 'score']:
                    pred_list = [ scale_vector(data) for data in pred_list ]
                elif "IDR" in score_type:
                    pred_list = [[one_minus_nan_float(val) for val in data ] for data in pred_list]
                if auc_method == 'no':
                    fig2 = plt.figure(1)
                    ax = fig2.add_subplot(111)
                    bp = ax.boxplot([[x for x in pred if x == x] for pred in pred_list])
                    fig2.savefig("boxplot_"+fprefix+"_mauc_"+score_type+"_idr.pdf")
                    fig2.clf()
                    continue
                if auc_method == 'max':
                    self.multi_dim_classification_lda(answer, pred_list, header)
                    self.multi_dim_classification_svm(answer, pred_list, header)
                if '2d' in criteria and len(list(set(sample_list))) > 1:
                    fig = self.two_dim_evaluation(fig, sample_list, index, answer, pred_list, func, assym, header, score_type, simple)
                else:
                    for i, tsample in enumerate(sample_list):
                        if tsample == 'cont':
                            pred_list[i] = [one_minus_nan_float(val) for val in pred_list[i]]
                    fig = self.one_dim_evaluation(fig, sample_list, index, answer, pred_list, func, assym, header, score_type)
            fig.add_subplot(111).set_ylim([-0.01, 1.01])
            fig.add_subplot(111).set_xlim([-0.01, 1.01])
            # fig.add_subplot(111).legend(loc=4, numpoints=1)
            fig.savefig(fprefix+"_"+curve+"_mauc_"+auc_method+"_idr"+criteria+".pdf")
            fig.clf()

    def get_feature_vectors(self, pred_list, answer, i):
        if i < len(pred_list):
            X = np.array([ pred_list[i] ])
        else:
            X = np.array(pred_list)
        Y = answer[:]
        return X, Y

    def get_filtered_vectors(self, X, Y, pos, i):
        global REMOVE
        X_pred = []
        if REMOVE:
            X, Y = remove_nan(X, Y)
            if i < 3:
                X_pred, Y = pred_conversion_assym(X[0], True, False, pos, Y)
                X_pred = np.transpose(np.array([X_pred]))
        else:
            if i < 3:
                X_pred, Y = pred_conversion_assym(X.tolist()[0], True, False, pos, Y)
                X_pred = np.transpose(np.array([X_pred]))
            X = X*100.
            X[np.isnan(X)] = np.nanmin(X)-1.
            # X[np.isnan(X)] = 0.
        return X, X_pred, Y

    def multi_dim_classification_lda(self, answer, pred_list, header):
        for i in range(len(pred_list)+1):
            pos = 0
            if ('IDR' in header or 'count' in header) and i == 0:
                pos = 1
            X, Y = self.get_feature_vectors(pred_list, answer, i)
            X, X_pred, Y = self.get_filtered_vectors(X, Y, pos, i)
            X = np.transpose(X)
            clf = LinearDiscriminantAnalysis()
            print(i, len(pred_list))
            if i < len(pred_list):
                clf.fit(X_pred, Y)
                coef = clf.coef_
                print('score_lda', (self.arg.input+['all'])[i], header, clf.score(X_pred, Y))
                tpr, fpr, auc = calc_tp_fp(Y, X, True, True, False, True)
                print('auc_lda', (self.arg.input+['all'])[i], header, X.shape, auc)
            clf.fit(X, Y)
            coef = clf.coef_
            scores = cross_val_score(clf, X, Y, cv=10)
            print('lda', (self.arg.input+['all'])[i], header, scores.mean(), scores.std(), coef)


    def multi_dim_classification_svm(self, answer, pred_list, header):
        for i in range(len(pred_list)+1):
            pos = 0
            if ('IDR' in header or 'count' in header) and i == 0:
                pos = 1
            X, Y = self.get_feature_vectors(pred_list, answer, i)
            X, X_pred, Y = self.get_filtered_vectors(X, Y, pos, i)
            X = np.transpose(X)
            clf = svm.SVC()
            if i < len(pred_list):
                clf.fit(X_pred, Y)
                print('score_svm', (self.arg.input+['all'])[i], header, clf.score(X_pred, Y))
                data = [[X[i, 0] for i in range(X.shape[0]) if Y[i] == 0], [X[i,0] for i in range(X.shape[0]) if Y[i] == 1]]
                bins=np.linspace(np.min(X), np.max(X), 100)
                plt.hist(data[0], bins, alpha=0.5, label='loop')
                plt.hist(data[1], bins, alpha=0.5, label='stem')
                plt.legend(loc='upper right')
                plt.savefig('hist_'+header+'_'+str(i)+'.pdf')
                plt.close()
                plt.clf()
            scores = cross_val_score(clf, X, Y, cv=5)
            print('svm', (self.arg.input+['all'])[i], header, scores.mean(), scores.std())
            fig = plt.subplots()



    def two_dim_evaluation(self, fig, sample_list, index, answer, pred_list, func, assym, header, score_type, simple=True):
        def func_nan(p, func):
            vec = [x for x in p if x == x]
            if len(vec) == 0:
                return float('nan')
            else:
                return func(vec)
        pred = [[], []]
        for tsample in list(index.keys()):
            pred[index[tsample]] = [pred_list[i] for i in range(len(pred_list)) if sample_list[i] == tsample]
            if len(pred[index[tsample]]) == 1:
                pred[index[tsample]] = pred[index[tsample]][0]
            else:
                pred[index[tsample]] = [func_nan(p, func) for p in zip(*[x for x in pred[index[tsample]]])]
        pos = 0
        SIMPLE = False
        if SIMPLE:
            print('simple 2d')
            pred_int = [x-y if x == x and y == y else x if x == x else -y for x, y in zip(pred[0], pred[1])]
            tpr, fpr, auc = calc_tp_fp(answer, pred_int, pos, assym=assym)
        else:
            print('2d')
            tpr, fpr, auc = calc_tp_fp_2d(answer, pred, pos, assym=assym)
        print('auc', header, len(pred), auc)

        print(tpr, fpr)
        fig.add_subplot(111).plot(fpr, tpr, 'o', markersize=3, color=self.color[score_type], label=score_type, markeredgewidth=0)
        return fig


    def one_dim_evaluation(self, fig, sample_list, index, answer, pred_list, func, assym, header, score_type):
        pred_list = pred_list[0:3]
        pred = []
        for p in zip(*[x for i, x in enumerate(pred_list)]):
            vec = [x for x in p if x == x]
            if len(vec) == 0:
                pred.append(float('nan'))
            else:
                pred.append(func(vec))
        pos = 0
        tpr, fpr, auc = calc_tp_fp(answer, pred, pos, assym=assym)
        print('auc', header, len(pred), auc)
        xlab, ylab = "1-Specificity", "Sensitivity"
        fig.add_subplot(111).plot(fpr, tpr, 'o', markersize=3, color=self.color[score_type], label=score_type, markeredgewidth=0)
        # fig, tymin = self.compute_and_plot_curves(pos, answer, pred, score_type, fprefix, 'case', curve, False, fig, assym)
        return fig


    def get_sample_line(self, dict_count, seta):
        keys = list(dict_count[0].keys())+list(dict_count[1].keys())
        tlines = []
        row_labels = []
        for i, key in enumerate(seta):
            if key not in keys:
                continue
            for j in [1, 0]:
                if key not in dict_count[j]:
                    continue
                tlines.extend([[['case', 'cont'][j], '', key]+dict_count[j][key]])
            row_labels.append(key)
        if len(row_labels) == 1 and row_labels[0] == 'score':
            return [], []
        return tlines, row_labels

    def plot_double_heatmap(self, tname, dict_count, prefix):
        fprefix = prefix+"_double"
        tlines, row_labels = self.get_sample_line(dict_count, ['ref', 'IDR', 'IDR-HMM-final'])
        if len(tlines) == 0:
            return
        data = self.set_rgb_data(tlines, row_labels)
        plt.imshow(data, interpolation='nearest', aspect='auto')
        plt.yticks(range(len(row_labels)), row_labels)
        colors = ["#000000", "#7F007F", "#FF00FF", "#0000FF", "#FF0000"]
        labels = ['Unmapped', 'Non-canonical(ref)', 'Ambiguous', 'Stem', 'Acc']
        patches = [ mpatches.Patch(color=col, label=lab) for col, lab in zip(colors, labels) ]
        lgnd = plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        plt.tight_layout()
        plt.savefig(fprefix+"_dfull_length.pdf", bbox_extra_artists=(lgnd,), bbox_inches='tight')
        plt.clf()

    def plot_double_scatter(self, tname, dict_count, prefix):
        if 'ref' not in dict_count[0].keys() or ('all' not in tname and tname != "RNA18S5+"):
            return
        clist = ['lightblue', 'red', 'pink']
        slist = ['loop', 'stem', 'pseudoknot']
        score_list =  ["count", "noHMMIDR", "IDR-HMM-final"]
        colors = ["red" if x == 0.0 else "lightblue" if x == 1.0 else "pink" for x in dict_count[0]["ref"]]
        for method in ["raw", "rank"]:
            all_loop = []
            for score_type in score_list:
                fprefix = prefix+"_scatter_"+method+"_"+score_type+"_90"
                xlist, ylist = dict_count[0][score_type], dict_count[1][score_type]
                if "IDR" in prefix:
                    xlist = [one_minus_nan_float(x) for x in xlist]
                    ylist = [one_minus_nan_float(x) for x in ylist]
                if method == 'rank':
                    xlist = rankdata(xlist, method="average")
                    ylist = rankdata(ylist, method="average")
                artist = []
                plt.rcParams['figure.figsize'] = [8.0, 6.0]
                for tcol in [clist[1], clist[0], clist[2]]:
                    tx, ty = zip(*[(x, y) for x, y, c in zip(xlist, ylist, colors) if c == tcol])
                    artist.append(plt.scatter(tx, ty, c=tcol))
                plt.xlabel("Case sample enrichment")
                plt.ylabel("Cont sample enrichment")
                plt.legend(artist, slist, loc="best")
                plt.savefig(fprefix+'.png')
                plt.clf()
                if method == 'rank' or score_type == 'noHMMIDR':
                    mx, my = np.percentile(xlist, 90), np.percentile(ylist, 90)
                    if method == 'raw' and score_type == "noHMMIDR":
                        mx, my = 0.5, 0.5
                    group = [1 if x > mx and y > my else 0 if x > mx else 2 if y > my else 3 for x, y in zip(xlist, ylist)]
                    size = [[], [], []]
                    for g in range(0, 4):
                        index = [i for i in range(len(xlist)) if group[i] == g]
                        struct = Counter([colors[i] for i in index])
                        for i, c in enumerate(clist):
                            print(g, i, c, struct[c])
                            size[i].append(struct[c])
                    ind = np.arange(4)
                    ps = []
                    print(size)
                    width=0.55
                    total = [ sum([temp[x] for temp in size]) for x in range(4)]
                    for x in range(len(size)):
                        for y in range(len(size[x])):
                            size[x][y] = float(size[x][y]/max(1, total[y]))
                    all_loop.append([size[0][g] for g in range(0, 4)])
                    bottom = [0, 0, 0, 0]
                    for s, c in zip(size, clist):
                        ps.append(plt.bar(ind+width, s, bottom=bottom, color=c, width=width))
                        bottom = [bottom[x]+s[x] for x in range(len(bottom))]
                    plt.xticks(0.30+ind+width, ('Case enrich', 'Both enrich', 'Cont enrich', 'No'))
                    # plt.legend(ps, slist)
                    plt.legend(ps, slist, loc='lower right')
                    plt.title("Ratio "+", ".join(list(map(str, total))))
                    plt.xlim(0, max(ind)+width+1)
                    plt.savefig(fprefix+'_bar.pdf')
                    plt.clf()
            if method == 'raw': continue
            print(method, 'end')
            print(all_loop)
            fig=plt.figure()
            ax=fig.add_subplot(111)
            width=0.27
            ind=np.arange(4)
            rects1 = ax.bar(ind, all_loop[0], width, color='1')
            rects2 = ax.bar(ind+width, all_loop[1], width, color='0.5')
            rects3 = ax.bar(ind+width*2, all_loop[2], width, color='0.0')
            ax.set_ylabel('Ratio of loop')
            ax.set_xticks(ind+width)
            ax.set_xticklabels(('Case enrich', 'Both enrich', 'Cont enrich', 'No'))
            ax.legend((rects1[0], rects2[0], rects3[0]), score_list, loc=1)
            fig.savefig(fprefix+'_loopbar.pdf')
            fig.clf()

    def append_count_score(self, case_file, cont_file = ''): # reading raw counts for replicates
        lines = []
        for i in range(2):
            sample, file = ["case", "cont"][i], [case_file, cont_file][i]
            if file == '':
                continue
            with open(os.path.join(self.arg.dir, file)) as f:
                for line in f.readlines()[1:]: #skip header
                    if len(line) > 0 and line[0] != "#":
                        transcript, score1, score2 = line.rstrip('\n').split('\t')
                        lines.append(["score1", transcript, sample]+score1.split(self.sep))
                        lines.append(["score2", transcript, sample]+score2.split(self.sep))
        return lines


def plot_index(index, params, fix):
    plt.plot([p[0,index] for p in params[0] if len(p) > 0] , 'r', lw=3)
    plt.plot([p[0,index] for p in params[1] if len(p) > 0], 'b', lw=1)
    if fix:
        all_max = max(1, 0.05+max([max([p[0,index] for p in x if len(p) > 0]) for x in params]))
        all_min = min(0, -0.05+min([min([p[0,index] for p in x if len(p) > 0]) for x in params]))
        plt.ylim([all_min, all_max])
    plt.title(['mu', 'sigma', 'rho', 'p'][index])

class ScatVis:
    """docstring for ScatVis"""
    def __init__(self, arg):
        self.arg = arg
        self.skip = 3
        self.sep = ';'

    def read_replicates(self, file):
        dict = {}
        with open(file) as f:
            for line in f.readiles():
                contents = line.rstrip('\n').split('\t')
                dict[contents[0]] = []
                for i in range(1, len(contents)):
                    dict[contents[0]].append(list(map(float, contents[i].split(';'))))
        return dict

    def append_count_score(self, case_file, cont_file = ''): # reading raw counts for replicates
        lines = []
        for i in range(2):
            sample, file = ["case", "cont"][i], [case_file, cont_file][i]
            if file == '':
                continue
            with open(os.path.join(self.arg.dir, file)) as f:
                for line in f.readlines()[1:]: #skip header
                    if len(line) > 0 and line[0] != "#":
                        transcript, score1, score2 = line.rstrip('\n').split('\t')
                        lines.append(["score1", transcript, sample]+score1.split(self.sep))
                        lines.append(["score2", transcript, sample]+score2.split(self.sep))
        return lines

    def visualize_parameter(self, rep=False):
        print(self.arg)
        rep = ('ctss' not in self.arg.case)
        lines, prefix, both = get_lines(self.arg.input[0], self.skip, self.sep)
        gene_set = ["RNA18S5+", "RNA5-8S5+", "RNA28S5+", "ENSG00000201321|ENST00000364451+"]
        trans_list, alines = append_integrated_transcript(lines, self.skip, seta=gene_set, boundary=rep)
        lines = lines+alines
        if self.arg.case != "":
            clines = self.append_count_score(self.arg.case, self.arg.cont)
            print('line append count')
            trans_list, alines = append_integrated_transcript_count(clines, self.skip, gene_set, rep)
            clines += alines
        else:
            return
        for i, sample in enumerate(['case', 'cont']):
            for trans in list(set(contents[1] for contents in lines)):
                if 'all_unknown' == trans or trans[-1] == "-":
                    continue
                fprefix = prefix+"_"+trans
                data = [contents for contents in lines if contents[1] == trans]
                cdata = [contents for contents in clines if contents[1] == trans]
                if len(data) == 0:  continue
                dict_count = get_sample_dict(data, self.skip)
                if len(dict_count[0].keys()) <= 1:  continue
                rep_count = get_sample_dict(cdata, self.skip)
                # print(rep_count)
                print(trans, rep_count[i].keys())
                for score_type in ["noHMMIDR", "IDR-HMM-final", "BUMHMM", 'PROBer', "count", "score"]:
                # for score_type in ["count"]:
                    if 'all' in trans:
                        if rep:
                            if sample == 'case':
                                self.plot_rep_irep_prof(dict_count[i][score_type], rep_count[i], fprefix+"_"+sample+"_"+score_type)
                        else:
                            self.plot_scatter(dict_count[i][score_type], rep_count[i], fprefix+"_"+sample+"_"+score_type)
                    # if rep and (trans == 'RNA18S5+' or trans == 'RNA28S5+' or 'all' in trans):
                    #     self.plot_reproducible_prof(dict_count[i][score_type], rep_count[i], fprefix+"_"+sample+"_"+score_type)

    def plot_scatter(self, dict_count, rep_count, prefix, double=True):
        fprefix = prefix+"_scatter"
        plt.xscale('log')
        plt.yscale('log')
        s1 = [x+1 for x in rep_count["score1"]]
        s2 = [x+1 for x in rep_count["score2"]]
        top, bottom = -1, -1
        prob = copy.deepcopy(dict_count)
        if "IDR" in prefix:
            prob = [one_minus_nan_float(x) for x in prob]
        selected_vec = [prob[i] for i in range(len(prob)) if prob[i] == prob[i]]
        if len(set(selected_vec)) <= 1:
            return
        if 'BUMHMM' in prefix:
            top, bottom = 0.1, 0.0
        else:
            top, bottom = np.percentile(selected_vec, 75), np.percentile(selected_vec, 25)
        print(len(s1), len(s2), len(dict_count))
        if double:
            xlist = rankdata(s1, method="average")
            ylist = rankdata(s2 , method="average")
            self.plot_double_histogram(fprefix, xlist, ylist, prob, top, bottom)
        else:
            for i in range(4):
                assert len(s1) == len(s2) and len(prob) == len(s1)
                if i == 0: # nan
                    index = [idx for idx in range(len(s1)) if prob[idx] != prob[idx]]
                elif i == 1: # normal
                    index = [idx for idx in range(len(s1)) if prob[idx] > bottom and prob[idx] < top]
                elif i == 2: # top
                    index = [idx for idx in range(len(s1)) if prob[idx] >= top]
                else:    #bottom 10
                    index = [idx for idx in range(len(s1)) if prob[idx] <= bottom]
                col = ['white', 'grey', 'blue', 'red'][i]
                label = ['none', 'neutral', 'rep-'+str(top), 'irre-'+str(bottom)][i]
                plt.scatter([s1[idx] for idx in index], [s2[idx] for idx in index], c=col, label=label)
            plt.xlabel("Replicate 1 read count")
            plt.ylabel("Replicate 2 read count")
            plt.legend(loc='upper left', numpoints=1)
            plt.savefig(fprefix+'.png')
            plt.clf()
            plt.close()

    # def add_prof(self, h, j, target, win, prof, break_flag, cont, address, norm):
    #     if break_flag or j < 0 or j >= len(target):
    #         prof[address].append(float('nan'))
    #     else:
    #         if target[j] == float('-inf'):
    #             break_flag = True
    #             prof[address].append(float('nan'))
    #         else:
    #             if "raw" in norm:
    #                 prof[address].append(target[j])
    #             else:
    #                 prof[address].append(target[j]/cont)
    #     return prof, break_flag

    def pop_prof(self, idx, target, win, norm, log=False):
        def value(pos):
            if log:
                return target[pos]/norm
            else:
                return np.log10(1.+target[pos])/np.log10(1.+norm)
        prof_a, prof_b = [float('nan')]*(win+1), [float('nan')]*(win)
        cont = target[idx]
        for h, j in enumerate(range(idx, idx+win+1)):
            if j < 0 or j >= len(target):
                break
            if target[j] == float('-inf'):
                break
            prof_a[h] = value(j)
        for h, j in enumerate(range(idx-1, idx-win-1, -1)):
            if j < 0 or j >= len(target):
                break
            if target[j] == float('-inf'):
                break
            prof_b[h] = value(j)
        return prof_b[::-1]+prof_a

    def plot_rep_irep_prof(self, dict_count, rep_count, prefix):
        fprefix = prefix+"_scatter"
        s1 = [x for x in rep_count["score1"]]
        s2 = [x for x in rep_count["score2"]]
        exp = dict_count
        prob = copy.deepcopy(dict_count)
        #print(dict_count)
        print(len(s1), len(s2), len(prob))
        norm = "_raw"
        if "IDR" in prefix:
            prob = [one_minus_nan_float(x) for x in prob]
        selected_vec = [prob[i] for i in range(len(prob)) if prob[i] == prob[i]]
        if len(set(selected_vec)) <= 1:
            return
        # tp, bp = 90, 10
        tp, bp = 90, 50
        top, bottom = np.percentile(selected_vec, tp), np.percentile(selected_vec, bp)
        fig = plt.figure(1)
        fig.set_size_inches((8, 3))
        exp_top, exp_bottom = 2, 0.5
        win = 50
        count = 1
        for i in range(4):
            assert len(s1) == len(s2) and len(prob) == len(s1)
            if i == 0: # nan
                index = [idx for idx in range(len(s1)) if prob[idx] != prob[idx]]
            elif i == 1: # normal
                index = [idx for idx in range(len(s1)) if bottom < prob[idx] and prob[idx] < top]
            elif i == 2: # top
                index = [idx for idx in range(len(s1)) if prob[idx] >= top]
            else:    #bottom 10
                index = [idx for idx in range(len(s1)) if prob[idx] <= bottom]
            if i != 1 and i != 2:  continue
            print(prefix, i, len(index))
            prof = []
            for h in range(win*2+1):
                prof.append([])
            for s, target in enumerate([s1, s2]):
                for idx in index:
                    if target[idx] != target[idx] or target[idx] < exp_bottom:
                        continue
                    break_flag = False
                    temp = self.pop_prof(idx, target, win, target[idx])
                    print(temp)
                    for i in range(len(temp)):
                        if temp[i] == temp[i]:
                            prof[i].append(temp[i])
                        # else:
                        #     prof[i].append(0.)
                    # for h, j in enumerate(range(idx, idx+win+1)):
                    #     prof, break_flag = self.add_prof(h, j, target, win, prof, break_flag, target[idx], h+win, norm)
                    # break_flag = False
                    # for h, j in enumerate(range(idx-1, idx-win-1, -1)):
                    #     prof, break_flag = self.add_prof(h, j, target, win, prof, break_flag, target[idx], win-h-1, norm)
                # print(len(prof), [len(prof[i]) for i in range(len(prof))], prof)
                # print("rep"+str(s), len(prof), prof[win][0:10], prof[win+1][0:10])
            prof_mean = np.nanmean(prof, axis=1)
            print(prof_mean)
                # prof_mean = moving_average(prof_mean, 5)
                # print(prof_mean)
                # print(len(prof), len(prof_mean), len(range(-win, win+1)))
            fig.add_subplot(111).plot(range(-win, win+1), prof_mean, c=['red', 'blue'][count-1], label=['Middle', 'Reproducible', ][count-1])
            count = count+1
        fig.add_subplot(111).legend()
        fig.savefig(fprefix+"_rir_prof"+norm+str(tp)+"_"+str(bp)+".pdf")
        fig.clf()


    def plot_double_histogram(self, fprefix, x, y, prob, top, bottom):
        dfs = []
        index_a = [idx for idx in range(len(prob)) if prob[idx] >= top]
        # index_a = index_a[0:5]
        index_b = [idx for idx in range(len(prob)) if prob[idx] <= bottom]
        # index_b = index_b[0:5]
        all_index = copy.deepcopy(index_a)
        all_index.extend(index_b)
        if len(all_index) == 0:
            return
        print(all_index)
        print(len(index_a), len(index_b))
        df1 = pd.DataFrame(np.column_stack(([x[i] for i in all_index], [y[i] for i in all_index])), index=range(len(all_index)), columns=['x1', 'x2'])
        df2 = pd.DataFrame(np.column_stack(([x[i] for i in index_a], [y[i] for i in index_a])), index=range(len(index_a)), columns=['x1', 'x2'])
        df3 = pd.DataFrame(np.column_stack(([x[i] for i in index_b], [y[i] for i in index_b])), index=range(len(index_b)), columns=['y1', 'y2'])
        df = pd.concat([df2, df3], axis=1, ignore_index=False)
        # print(df)
        EPS=0.00000001
        a=np.nanmin([x[i] for i in all_index])
        b=np.nanmax([x[i] for i in all_index])
        c=np.nanmin([y[i] for i in all_index])
        d=np.nanmax([y[i] for i in all_index])
        print('min-max', a, b)
        print('min-max', c, d)
        xrange = np.arange(a, b, (b-a)/20)
        yrange = np.arange(c, d, (d-c)/20)
        print(xrange)
        print(yrange)
        print([i for i in index_a+index_b])
        print(len(index_a), len(index_b))
        print(y)
        print([y[i] for i in all_index])
        p = JointGrid(
            x = df1['x1'],
            y = df1['x2']
            )

        p = p.plot_joint(
            plt.scatter,
            color = ['b' for i in range(len(index_a))]+['r' for i in range(len(index_b))],
            alpha=0.3
            )

        p.ax_marg_x.hist(
            df2['x1'],
            alpha = 0.5,
            bins=xrange,
            color='b'
            )

        p.ax_marg_y.hist(
            df2['x2'],
            orientation = 'horizontal',
            alpha = 0.5,
            bins=yrange,
            color='b'
            )

        p.ax_marg_x.hist(
            df3['y1'],
            alpha = 0.5,
            bins=xrange,
            color='r'
            )

        p.ax_marg_y.hist(
            df3['y2'],
            orientation = 'horizontal',
            alpha = 0.5,
            bins=yrange,
            color='r'
            )
        p.savefig(fprefix+'_hist.pdf')
        # p.clf()
        # p.close()
        # p.clf()


    def plot_reproducible_prof(self, dict_count, rep_count, prefix):
        fprefix = prefix+"_rprof"
        s1 = [x for x in rep_count["score1"]]
        s2 = [x for x in rep_count["score2"]]
        mean_s = [float(s1[x]+s2[x])/2 for x in range(len(s1))]
        if False:
            top, bottom = 10000, 1000
        else:
            top, bottom = 1000, 100
            fprefix = fprefix+"_lowc"
        prob = copy.deepcopy(dict_count)
        if "IDR" in prefix:
            prob = [one_minus_nan_float(x) for x in prob]
        print(len(s1), len(s2), len(dict_count))
        selected = [i for i, x in enumerate(mean_s) if x >= bottom and x <= top]
        if len(selected) == 0:
            return
        print(len(selected), [mean_s[x] for x in selected])
        print(prefix)
        selected_vec = [prob[i] for i in selected if prob[i] == prob[i]]
        print(set(selected_vec))
        if len(set(selected_vec)) <= 1:
            return
        top_thres, low_thres = np.percentile(selected_vec, 50), np.percentile(selected_vec, 50)
        rep, irep = [i for i in selected if prob[i] >= top_thres], [i for i in selected if prob[i] < low_thres]
        win = 50
        tlabels = ["rep", "irep"]
        tcolors = ["blue", "red"]
        print(top_thres, low_thres, len(rep), len(irep))
        for s, target in enumerate([rep, irep]):
            prof = []
            for i in range(win*2+1):
                prof.append([])
            for i in target:
                for h, j in enumerate(range(i-win, i+win+1)):
                    if h == win:
                        assert mean_s[j] >= bottom and mean_s[j] <= top
                    if j >= 0 and j < len(s1):
                        prof[h].append(mean_s[j])
                        # prof[h].append(mean_s[j]/float(mean_s[i]))
                    else:
                        prof[h].append(float('nan'))
            print(tlabels[s], len(prof), prof[win][0:10], prof[win+1][0:10])
            prof_mean = np.nanmean(prof, axis=1)
            prof_mean = moving_average(prof_mean, 5)
            prof_mean = [float('nan')]*4+list(prof_mean)
            plt.plot(range(-win, win+1), prof_mean, label=tlabels[s], color=tcolors[s])
        plt.xlabel("Position")
        plt.ylabel("Average")
        plt.legend(loc='upper left')
        plt.savefig(fprefix+'.pdf')
        plt.clf()
        plt.close()


def add_prefix(lines):
    index = [i for i, line in enumerate(lines) if 'Transcript' in line]
    if len(index) == 0 or index[0] == len(lines) - 1:
        return ''
    else:
        return '_' + lines[index[0] + 1].strip('[]\n').split(',')[1].strip(' \'') + '_'

class ParamVis:
    """docstring for ParamVis"""
    def __init__(self, arg):
        self.arg = arg

    def visualize_idr_parameter(self, plines, prefix=''):
        params = [[], []]
        for line in plines:
            print(line)
            param = line.split('\t')[1]
            if '], [' in line:
                # param = str.replace(param, '], [', ']; [')
                param = np.array(eval(param))
                print(param)
                params[0].append(np.ndarray((1, 4), dtype=float, buffer=param[0,]))
                params[1].append(np.ndarray((1, 4), dtype=float, buffer=param[1,]))
            else:
                params[0].append(np.ndarray((1, 4), dtype=float, buffer=np.array(eval(param))[0,]))
        fig=plt.figure(1)
        for i in range(4):
            plt.subplot(220+(i+1))
            plot_index(i, params, self.arg.fix)
        colors = ["r", "b"]
        labels = ["case", "control"]
        patches = [ mpatches.Patch(color=col, label=lab) for col, lab in zip(colors, labels) ]
        plt.legend(handles=patches, loc=4)
        plt.savefig(prefix+"_idr.pdf")
        plt.clf()

    def visualize_transition_parameter(self, plines, prefix=''):
        fig=plt.figure(1)
        subplot_index = [311, 312, 313]
        if len(plines) == 0:
            return
        index = sorted(list(set([0, math.floor(len(plines)/2)-1,  len(plines)-1])))
        for i, p in enumerate(index):
            plt.subplot(subplot_index[i])
            # param = str.replace(plines[p].split('\t')[2], '], [', ']; [')
            param = plines[p].split('\t')[2]
            param = np.array(eval(param))
            plt.imshow(param, aspect='auto', interpolation='nearest')
            plt.colorbar(orientation='horizontal')
            plt.title("Transition matrix ("+str(p)+")")
        plt.tight_layout()
        plt.savefig(prefix+"_trans.pdf")
        plt.clf()

    def visualize_lhd(self, plines, prefix=''):
        params = []
        for line in plines:
            params.append(float(line.split('\t')[1]))
        plt.plot(params, 'bo', params, 'k')
        plt.savefig(prefix+"_q_func.pdf")
        plt.clf()

    def visualize_param_block(self, lines, prefix):
        prefix = prefix + add_prefix(lines)
        self.visualize_idr_parameter([line for line in lines if re.match('^Set new_p', line) is not None], prefix)
        self.visualize_transition_parameter([line for line in lines if re.search('^Set new_transition', line) is not None], prefix)
        self.visualize_lhd([line for line in lines if re.search('new lhd', line) is not None], prefix)

    def visualize_parameter(self):
        lines = []
        with open(self.arg.input[0]) as f:
            lines = [line.rstrip('\n') for line in f.readlines() if len(line) > 0 and line[0] != "#"]
        prefix = os.path.basename(self.arg.input[0])
        pre = -1
        for i in range(len(lines)+1):
            if i == len(lines) or 'Dataset' in lines[i]:
                if pre >= 0:
                    self.visualize_param_block(lines[pre:i], prefix)
                pre = i

def main(argv):
    parser = get_parser()
    options = parser.parse_args(argv[1:])
    print(options)
    if len(options.input) == 0:
        os.exit(0)
    if options.parameter:
        param = ParamVis(options)
        param.visualize_parameter()
    elif options.scatter:
        param = ScatVis(options)
        param.visualize_parameter()
    else:
        acc = AccEval(options)
        acc.plot_heatmap()

if __name__ == '__main__':
    main(sys.argv)
