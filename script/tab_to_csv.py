import sys
import argparse
import os.path
import math
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='+', type=str, help="csv files", metavar="FILES")
    parser.add_argument("--fasta", dest="fasta", type=str, help="fasta file (not required, but necessary for RNA structure prediction)", required=False)
    parser.add_argument("--dir", type=str, help="Directory for RT stop count and coverage data.", metavar="DIR", required=False)
    parser.set_defaults(dir="./", merge=False, case_norm="", cont_norm="", alpha=0.25, reverse=False)
    return parser

def get_fasta_dict(fname):
    if fname == "": return None
    fasta_dict = {}
    seq, name = "", ""
    with open(fname) as f:
        for line in f.readlines():
            if line == "":  continue
            if line[0] == "#":
                if len(name) > 0:
                    fasta_dict[name] = seq
                seq, name = "", ""
            if line[0] == ">":
                if len(seq) > 0:
                    fasta_dict[name] = seq
                seq, name = "", line[1:].rstrip('\n')
            else:
                seq += line.rstrip('\n')
        if len(seq) > 0:
            fasta_dict[name] = seq
    return fasta_dict

def nan_float(s):
    if s == 'None' or s == 'nan' or s == "NA":
        return float('nan')
    else:
        return float(s)

def none_float(s):
    if s == 'None' or s == 'nan' or s == "NA":
        return float('nan')
    else:
        return float(s)

def thres_cut_float(s, prefix, thres=0):
    if s == 'None' or s == 'nan' or s == "NA":
        return -1.
    else:
        x = float(s) if "IDR" not in prefix else 1.-float(s)
        return -1. if x < thres else x

def comp_threshold(score, prefix, top):
    temp = sorted([x for x in [none_float(x) for x in score] if x == x])
    if "IDR" in prefix:
        temp = [1.-float(s) for s in temp]
    if top == 0:
        return np.nanmin(temp)-1.
    th = temp[max(0, math.floor(float(top)/(100.)*len(temp))-1)]
    return th

def output_csv(trans, prefix, score, seq=""):
    with open("shape_format_"+prefix+".csv", 'w') as f:
        if seq != "":
            assert len(score) == len(seq)
            for i in range(len(score)):
                f.write(" ".join([str(i+1), seq[i], ('%.6f' % nan_float(score[i]))])+"\n")
        else:
            for i in range(len(score)):
                f.write(" ".join([str(i+1), ('%.6f' % nan_float(score[i]))])+"\n")
    if 'case' not in prefix:
        return
    for th in range(0, 101, 10):
        thres = comp_threshold(score, prefix, th)
        with open("shape_format_"+prefix+"_"+str(th)+".dat", 'w') as f:
            for i in range(len(score)):
                f.write("\t".join([str(i+1), ('%.6f' % thres_cut_float(score[i], prefix, thres))])+"\n")

    with open("seq_"+trans+".fa", 'w') as f:
        if seq != "":
            assert len(score) == len(seq)
            f.write(">"+trans+"\n")
            f.write(seq+"\n")

def file_convert(fname, options):
    skip = 3
    sep = ";"
    base = os.path.basename(fname)
    fasta_dict = get_fasta_dict(options.fasta)
    with open(os.path.join(options.dir, fname)) as f:
        lines = [line.rstrip('\n').split('\t') for line in f.readlines()]
        for score_type, trans, sample, score in lines:
            prefix = base+"_"+score_type+"_"+sample+"_"+trans
            if score_type == 'ref' or score_type == 'type':
                continue
            if fasta_dict is not None:
                if trans in fasta_dict: output_csv(trans, prefix, score.split(sep), fasta_dict[trans])
            else:
                output_csv(trans, prefix, score)


if __name__ == '__main__':
    parser = get_parser()
    options = parser.parse_args()
    if len(options.files) > 0:
        for fname in options.files:
            file_convert(fname, options)
