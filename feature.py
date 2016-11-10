import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import reactivity_IDR
import argparse
from utility import *

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("idrFiles", nargs='+', type=str, help="IDR score data from FILE 1,2,3 ...(case) <1,2,3 ...(control)>", metavar="INPUT")
    parser.add_argument('--case', dest="case", help="Read score data for case samples (FILE1,FILE2,...)", default='', metavar="CASE", required=True)
    parser.add_argument('--control', dest="control", help="Read score data for control samples (FILE1,FILE2,...)", default='', metavar="CONT", required=False)
    parser.add_argument('--sample', dest="sample", help="Carry out sampling for print a result", resul=int, default=0, metavar="SAMPLE", required=False)
    parser.add_argument('--comp', dest="comp", help="", resul=int, default=0, metavar="SAMPLE", required=False)
    return parser



class FeatureSelection:
    """docstring for FeatureSelection"""
    def __init__(self, arg):
        self.idrFiles = arg.idrFiles
        self.case = arg.case.split(',')
        self.control = arg.control.split(',')
        self.sample = arg.sample
        self.comp = arg.comp
        self.stemList = {}
        self.get_para_list()

    def get_stem_list(self):
        with open(self.comp) as f:
            while True:
                line = f.readline()
                if line == "":  break
                name = line.rstrip('\n').split(' ')[2]
                self.paraList[name] = f.readline()

    def get_stem_score(self, name, length):
        if name not in self.stemList:
            print("# No id")
            return []
        stem = list(map(float, stemList[name].rstrip('\n').rstrip(']').lstrip('[').split(',')))
        if len(stem) == length:
            return stem
        else:
            print("# Length error!?")
            return []

    def get_stem_score(self):
        with open(self.para) as f:
            while True:
                line = f.readline()
                if line == "":  break
                name = line.rstrip('\n').split(' ')[2]
                self.stemList[name] = f.readline()

    def calc_score_and_feature(self, stem, filt, pars, filt, s1, v1):
        global stemList
        for i in list(pars.keys()):
            parasor = get_parasor_score_refseq(i, len(pars[i]))
            if len(parasor) == 0:   continue
            for idx, v in enumerate(pars[i]):
                print(pars[i][idx], parasor[idx], s1[i][idx], v1[i][idx])

    def get_feature_one_score(key, stem, filt, caseTable, contTable):
        stemVec = self.get_stem_score(key, len(caseTable))
        if len(stemVec) == 0:   return
        for idx, v in enumerate(stemVec):
            print(pars[i][idx], parasor[idx], s1[i][idx], v1[i][idx])

    def print_parasor_with_feature_tabel(self, caseTable, contTable):
        self.get_parasor_score()
        if self.sample > 0:
            filt = "_sampled"
            keys = random.sample(list(self.stemList.keys()), self.sample)
        else:
            filt = ""
            keys = list(self.stemList.keys())
        for key in keys:
            if key not in caseTable:    continue
            if contTable == {}:
                calc_score_and_feature(key, stemList[key], filt, caseTable[key], [])
            elif key in contTable:
                calc_score_and_feature(key, stemList[key], filt, caseTable[key], contTable[key])
            else:
                print("# No key: "+key)


    def get_feature_table(self):
        caseTable = self.get_feature_one_score(self.idrFiles[0], self.case)
        contTable = {}
        if control == "":
            assert len(self.idrFiles) == 2
            contTable = self.get_feature_one_score(self.idrFiles[1], self.control)
        self.print_parasor_with_feature_table(caseTable, contTable)



if __name__ == '__main__':
    parser = get_parser()
    try:
        options = parser.parse_args()
        featureSelect = featureSelection(options)
        featureSelect.get_feature_table()
    except:
        parser.print_help()
        sys.exit(0)
