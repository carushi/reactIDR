#!/bin/python

import argparse
import sys
import os

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("fastqFiles", nargs='+', type=str, help="fastq files", metavar="INPUT")
    parser.add_argument("-b", "--barcode", dest="barcode", help="Set barcode sequence (ex NNNNXXXXNNNNN)", metavar="BARCODE", default="NNNNXXXXNNNNN", required=False)
    parser.add_argument("-m", "--mismatch", dest="mismatch", help="The number of mismatch to allow for collapsing", metavar="MISMATCH", default=0, type=int, required=False)
    parser.add_argument("-s", "--split", dest="split", action="store_true", help="Split fastq files depending on the barcode", required=False)
    parser.add_argument("-r", "--remove", dest="remove", action="store_true", help="Remove barcode sequences", required=False)
    parser.set_defaults(split=True, remove=False)
    return parser


def read_collapse():
    count = 0
    temp = []
    dict = {}
    for line in sys.stdin:
        if line == "":
            if count%4 != 0:
                sys.stderr("Warning: probably truncated")
            return
        count += 1
        if count%4 == 0:
            dict = add_seq(temp[3], dict)
            temp = []
        temp.append(line)

class readCollapse():
    """docstring for readCollapse"""
    def __init__(self, arg):
        self.files = arg.fastqFiles
        self.barcode = arg.barcode
        self.len_barcode = len(self.barcode)
        self.mismatch = arg.mismatch
        self.split = arg.split
        self.remove = arg.remove

    def add_read(self, mark, seq, score_list, ind):
        score = sum([ord(i) for i in score_list])
        if seq in mark:
            if mark[seq][1] < score:    mark[seq] = (ind, score)
        else:
            mark[seq] = (ind, score)
        return mark, self.extract_barcode(seq)

    def extract_barcode(self, seq):
        return "".join([seq[i] for i in range(min(len(seq), self.len_barcode)) if self.barcode[i] == "X"])

    def get_marked_reads(self, file):
        mark = {}
        barcode = {}
        output = {}
        with open(file) as f:
            temp, count = [], 0
            for line in f.readlines():
                if line == "":
                    break
                temp.append(line)
                if len(temp) == 4:
                    mark, tbarcode = self.add_read(mark, temp[1], temp[3], count)
                    barcode[tbarcode] = 1
                    temp, count = [], count+1
        return mark, list(set(barcode))

    def write_subjected_reads(self, file, ofile, mark, tbarcode):
        with open(ofile, 'w') as output:
            with open(file) as input:
                temp, count = [], 0
                for line in input.readlines():
                    if line == "":
                        break
                    temp.append(line)
                    if len(temp) == 4:
                        if temp[1] in mark and mark[temp[1]][0] == count and self.extract_barcode(temp[1]) == tbarcode:
                            if self.remove:
                                temp[1] = temp[1][len(self.barcode):]
                                temp[3] = temp[3][len(self.barcode):]
                            for elem in temp:
                                output.write(elem)
                        temp, count = [], count+1

    def write_splitted_reads(self, file, mark, barcode):
        basename = file.rstrip('.fastq')
        if self.split:
            for tbarcode in barcode:
                if len(tbarcode) == len([i for i in self.barcode if i == "X"]):
                    ofile = basename+"_"+tbarcode+".fastq"
                    print("Write to "+ofile+"...")
                    self.write_subjected_reads(file, ofile, mark, tbarcode)
        else:
            ofile = basename+"_nosplit.fastq"
            self.write_subjected_reads(file, ofile, mark, "")

    def calc_mismatch(self, left, right):
        return len([i for i in range(0, min(len(left), len(right))) if left[i] != right[i]])


    def collapse_with_mismatch(self, mark):
        seq_list = list(mark.keys())
        for i in range(len(seq_list)-2, -1, -1):
            for j in range(len(seq_list)-1, i, -1):
                if self.calc_mismatch(seq_list[i], seq_list[j]) < self.mismatch:
                    if mark[seq_list[i]][1] < mark[seq_list[j]][1]:
                        mark.pop(seq_list[i])
                        seq_list.pop(i)
                        break
                    else:
                        mark.pop(seq_list[j])
                        seq_list.pop(j)
        return mark


    def apply_collapsing(self):
        for file in self.files:
            mark, barcode = self.get_marked_reads(file)
            if self.mismatch > 0:
                mark = self.collapse_with_mismatch(mark)
            self.write_splitted_reads(file, mark, barcode)

if __name__ == '__main__':
    parser = get_parser()
    # try:
    options = parser.parse_args()
    collapse = readCollapse(options)
    collapse.apply_collapsing()
    # except:
        # parser.print_help()

