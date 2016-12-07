import sys
import random
import numpy as np
import itertools
# import matplotlib.pyplot as plt
# import math
# import reactivity_IDR
import argparse
import utility

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("idrFiles", nargs='+', type=str, help="IDR score data from case(, control)", metavar="INPUT")
    parser.add_argument('--case', dest="case", help="Read score data for case samples", default='', metavar="CASE", required=True)
    parser.add_argument('--control', dest="control", help="Read score data for control samples", default='', metavar="CONT", required=False)
    parser.add_argument('--sample', dest="sample", help="Carry out sampling for print a result", type=int, default=0, metavar="SAMPLE", required=False)
    parser.add_argument('--cutoff', dest="cutoff", help="Threshold for the minimum coverage", type=int, default=5, metavar="THRESHOLD", required=False)
    parser.add_argument('--peak', dest="peak", help="Print peak width around the position with the minimum coverage [PEAK]", type=int, default=0, metavar="PEAK", required=False)
    parser.add_argument('--comp', dest="comp", help="Set File name for stem probability", default='', metavar="COMP", required=True)
    parser.add_argument("--parasor", dest="parasor", action="store_true", help="Read ParasoR-formatted comp file (default: false)", required=False)
    parser.add_argument('--dir', dest="dir", help="Select directory of source files", default='./', metavar="DIR", required=False)
    # parser.add_argument('--float', dest="pfloat", action="store_true", required=False)
    parser.set_defaults(parasor=False)
    return parser



class FeatureSelection:
    """docstring for FeatureSelection"""
    def __init__(self, arg):
        self.idrFiles = arg.idrFiles
        self.case = arg.case
        self.control = arg.control
        self.sample = arg.sample
        self.comp = arg.comp
        self.peak = arg.peak
        self.cutoff = arg.cutoff
        self.parasor = arg.parasor
        self.dir = arg.dir
        self.stemList = {}
        self.window = 20
        self.get_stem_list()


    def get_stem_list(self):
        with open(self.comp) as f:
            if parasor:
                while True:
                    line = f.readline()
                    if line == "":  break
                    name = line.rstrip('\n').split(' ')[2]
                    self.stemList[name] = f.readline()
            else:
                name, ref = "", ""
                while True:
                    line = f.readline().rstrip('\n')
                    if line == "" or line[0] == ">":
                        if len(name) > 0:
                            self.stemList[name] = ref
                        if line == "":  break
                        else:   name, ref = line[1:].split(' ')[0], ""
                    else:
                        ref += line

    def get_stem_score(self, name, length):
        if name not in self.stemList:
            print("# No id")
            return []
        if self.parasor:
            stem = list(map(float, self.stemList[name].rstrip('\n').rstrip(']').lstrip('[').split(',')))
        else:
            table = {'.':-1, '(':1, ')':1}
            stem = [table[i] if i in table 0 else for i in self.stemList[name]
        if len(stem) == length:
            return stem
        else:
            print("# Length error!?")
            return []

    def comp_peak_width(self, meanList, pos, window, mincov, mindist, five = True, norm = None):
        if type(norm) == type(None):
            norm = meanList[pos]
        if norm == 0 or norm != norm:
            return 0
        dist, prev = 0, 0
        for i in range(1, window+1):
            value = 0
            if five and pos-i >= 0:  value = meanList[pos-i]
            if not five and pos+i < len(meanList):  value = meanList[pos+i]
            if value < mincov:  continue
            if i-prev > mindist:    break
            prev, dist = i, i
        return dist

    def comp_peak_width_up_and_down(self, meanList, norm, mincov, mindist):
        plus = self.comp_peak_width([meanList[i] for i in range(1, len(meanList), 2)], -1, self.window, mincov, mindist, False, norm)
        minus = self.comp_peak_width([meanList[i] for i in range(len(meanList)-2, -1, -2)], self.window, self.window, mincov, mindist, True, norm)
        if norm == 0 or norm != norm:
            return plus, minus, 0
        else:
            return plus, minus, 1

    def get_trimmed_output(self, contents, left, right):
        return " ".join(contents[0:8])+" "+str(left)+" "+" ".join(contents[(self.window*2+8):(self.window*2+8+5)])+" "+str(right)

    def transform_profile_to_peak_width(self, line, mincov, mindist = 2):
        contents = line.rstrip('\n').split(' ')
        sp, vp = [], []
        for cov in [mincov*1.5, mincov, mincov*0.5]:
            spwidth, smwidth, spos = self.comp_peak_width_up_and_down([float(contents[i]) for i in range(8, self.window*2+8)], float(contents[5]), cov, mindist)
            vpwidth, vmwidth, vpos = self.comp_peak_width_up_and_down([float(contents[i]) for i in range(13+self.window*2, 13+self.window*4)], float(contents[10+self.window]), cov, mindist)
            sp.append(str(spwidth+smwidth+spos))
            vp.append(str(vpwidth+vmwidth+vpos))
        return self.get_trimmed_output(contents, " ".join(sp), " ".join(vp))

    # def append_peak_width(self, file, mincov = 50, mindist = 2):
    #     with open(file) as f:
    #         header = True
    #         for line in f.readlines():
    #             if line == "":  continue
    #             if line[0] == "#":  continue
    #             if header:
    #                 if line[0:4] == "Stem":
    #                     contents = line.rstrip('\n').split(' ')
    #                     print(self.get_trimmed_output(contents, "Swidth" , "Vwidth"))
    #                     header = False
    #             else:
    #                 print(self.transform_profile_to_peak_width(line, float(mincov)/100.0, mindist))


    def mean_around(self, meanList, pos, window):
        temp = []
        if meanList[pos] == 0:
            return ["NAN"]*window*2
        for i in range(1, window+1):
            if pos-i >= 0:
                temp.append(meanList[pos-i]/meanList[pos])
            else:
                temp.append("NAN")
            if pos+i < len(meanList):
                temp.append(meanList[pos+i]/meanList[pos])
            else:
                temp.append("NAN")
        return temp

    def get_mean_pars(self, rep1Vec, rep2Vec):
        return [np.log10(np.mean(rep1Vec[max(0, i-2):min(len(rep1Vec), i+3)]+rep2Vec[max(0, i-2):min(len(rep1Vec), i+3)])+5.) for i in range(len(rep1Vec))]

    def extract_features(self, IDRVec, rep1Vec, rep2Vec, sumList = []):
        data = []
        meanList = [float(rep1Vec[i]+rep2Vec[i])/2. for i in range(len(IDRVec))]
        varList = [np.var([rep1Vec[i], rep2Vec[i]]) for i in range(len(IDRVec))]
        parsList = self.get_mean_pars(rep1Vec, rep2Vec)
        assert len(IDRVec) == len(rep1Vec)
        assert len(rep1Vec) == len(rep2Vec)
        for i in range(len(IDRVec)):
            values = [IDRVec[i], parsList[i], meanList[i], varList[i], float(sum(rep1Vec)+sum(rep2Vec))/(2.0*len(rep1Vec))]
            vec = self.mean_around(meanList, i, self.window)
            values.extend(vec)
            data.append(np.asarray(values))
        return data

    def get_feature_one_score(self, idrfile, scorefile, keys):
        IDR = utility.parse_idr(idrfile)
        rep1, rep2 = utility.parse_score(scorefile)
        table = {}
        for key in keys:
            if key in rep1 and key in rep2:
                caseTable = self.extract_features(IDR[key], rep1[key], rep2[key])
        return table

    def print_score_and_feature(self, key, sumList, caseTable, contTable):
        stem = self.get_stem_score(key, len(sumList))
        threshold = self.cutoff
        if len(stem) == 0:   return
        for i in range(len(sumList)):
            if abs(sumList[i]) > threshold:
                out = np.append(stem[i], sumList[i])
                out = np.append(out, float(caseTable[i][1])-float(contTable[i][1]))
                out = np.append(out, caseTable[i])
                out = np.append(out, contTable[i])
                if self.peak > 0:
                    line = " ".join(["%.4f" % float(n) for n in out])
                    print(self.transform_profile_to_peak_width(line, float(self.peak)/100.0))
                else:
                    print(" ".join(["%.4f" % float(n) for n in out]))

    def extract_dictionaries(self, keys, sidrfile, sscorefile, vidrfile, vscorefile):
        sIDR = utility.extract_dict(utility.parse_idr(self.dir+sidrfile), keys)
        vIDR = utility.extract_dict(utility.parse_idr(self.dir+vidrfile), keys)
        srep1, srep2 = utility.parse_score(self.dir+sscorefile)
        srep1 = utility.extract_dict(srep1, keys)
        srep2 = utility.extract_dict(srep2, keys)
        vrep1, vrep2 = utility.parse_score(self.dir+vscorefile)
        vrep1 = utility.extract_dict(vrep1, keys)
        vrep2 = utility.extract_dict(vrep2, keys)
        return sIDR, vIDR, srep1, srep2, vrep1, vrep2

    def get_feature_both_scores(self, keys, sidrfile, sscorefile, vidrfile, vscorefile):
        sIDR, vIDR, srep1, srep2, vrep1, vrep2 = self.extract_dictionaries(keys, sidrfile, sscorefile, vidrfile, vscorefile)
        header = " ".join(["Stem", "Sum_coverage", "PARS", "IDR_s", "PARS_s", "Mean_s", "Var_s", "Tran_Mean_s"]
            +[str(i)+str(t)+str(n) for n in range(1, self.window+1) for i in ["s"] for t in ["m", "p"]]
            +["IDR_v", "PARS_v", "Mean_v", "Var_v", "Tran_Mean_v"]
            +[str(i)+str(t)+str(n) for n in range(1, self.window+1) for i in ["v"] for t in ["m", "p"]])
        if self.peak > 0:
            print(self.get_trimmed_output(header.split(' '), " ".join(["Swidth", "Swidth2", "Swidth3"]), " ".join(["Vwidth", "Vwidth2", "Vwidth3"])))
        else:
            print(header)
        for key in list(keys):
            print("#", key)
            if key in srep1 and key in srep2 and key in vrep1 and key in vrep2:
                sumList = [srep1[key][i]+srep2[key][i]+vrep1[key][i]+vrep2[key][i] for i in range(len(sIDR[key]))]
                contTable = self.extract_features(sIDR[key], srep1[key], srep2[key])
                caseTable = self.extract_features(vIDR[key], vrep1[key], vrep2[key])
                self.print_score_and_feature(key, sumList, caseTable, contTable)
            else:
                utility.log("# No key: "+key)
                utility.log(key in srep1)
                utility.log(key in srep2)
                utility.log(key in vrep1)
                utility.log(key in vrep2)

    def extract_dictionaries_iterator(self, keys, sidrfile, sscorefile, vidrfile, vscorefile):
        sIDR = utility.parse_idr_iterator(self.dir+sidrfile)
        vIDR = utility.parse_idr_iterator(self.dir+vidrfile)
        srep = utility.parse_score_iterator(self.dir+sscorefile)
        vrep = utility.parse_score_iterator(self.dir+vscorefile)
        return sIDR, vIDR, srep, vrep

    def get_feature_both_scores_iterator(self, keys, sidrfile, sscorefile, vidrfile, vscorefile):
        sIDR_list, vIDR_list, srep_list, vrep_list = self.extract_dictionaries_iterator(keys, sidrfile, sscorefile, vidrfile, vscorefile)
        header = " ".join(["Stem", "Sum_coverage", "PARS", "IDR_s", "PARS_s", "Mean_s", "Var_s", "Tran_Mean_s"]
            +[str(i)+str(t)+str(n) for n in range(1, self.window+1) for i in ["s"] for t in ["m", "p"]]
            +["IDR_v", "PARS_v", "Mean_v", "Var_v", "Tran_Mean_v"]
            +[str(i)+str(t)+str(n) for n in range(1, self.window+1) for i in ["v"] for t in ["m", "p"]])
        if self.peak > 0:
            print(self.get_trimmed_output(header.split(' '), " ".join(["Swidth", "Swidth2", "Swidth3"]), " ".join(["Vwidth", "Vwidth2", "Vwidth3"])))
        else:
            print(header)
        for sIDR, vIDR, srep, vrep in itertools.zip_longest(sIDR_list, vIDR_list, srep_list, vrep_list):
            assert(sIDR[0] == vIDR[0] and vIDR[0] == srep[0] and srep[0] == vrep[0])
            print("#", sIDR[0])
            if sIDR[0][0] != "N":   continue
            sumList = [srep[1][i]+srep[2][i]+vrep[1][i]+vrep[2][i] for i in range(len(sIDR[1]))]
            contTable = self.extract_features(sIDR[1], srep[1], srep[2])
            caseTable = self.extract_features(vIDR[1], vrep[1], vrep[2])
            self.print_score_and_feature(sIDR[0], sumList, caseTable, contTable)

    def get_feature_table(self):
        if self.sample > 0:
            filt = "_sampled"
            keys = random.sample(list(self.stemList.keys()), self.sample)
        else:
            filt = ""
            keys = list(self.stemList.keys())
        if self.control == "":
            self.get_feature_one_score(keys, self.idrFiles[0], self.case), {}
        else:
            assert len(self.idrFiles) == 2
            # self.get_feature_both_scores(keys, self.idrFiles[1], self.control, self.idrFiles[0], self.case)
            self.get_feature_both_scores_iterator(keys, self.idrFiles[1], self.control, self.idrFiles[0], self.case)


if __name__ == '__main__':
    parser = get_parser()
    try:
        options = parser.parse_args()
        featureSelect = featureSelection(options)
        featureSelect.get_feature_table()
    except:
        parser.print_help()
        sys.exit(0)
