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
    parser.add_argument('--case', nargs='+', dest="case", help="Read score data for case samples (score file, coverage file, ...)", default='', metavar="CASE", required=True)
    parser.add_argument('--control', nargs='+', dest="control", help="Read score data for control samples *score file, coverage file, ...)", default='', metavar="CONT", required=False)
    parser.add_argument('--sample', dest="sample", help="Carry out sampling for print a result", type=int, default=0, metavar="SAMPLE", required=False)
    parser.add_argument('--cutoff', dest="cutoff", help="Threshold for the minimum coverage", type=int, default=5, metavar="THRESHOLD", required=False)
    parser.add_argument('--peak', dest="peak", help="Print peak width around the position with the minimum coverage [PEAK]", type=int, default=0, metavar="PEAK", required=False)
    parser.add_argument('--comp', dest="comp", help="Set File name for stem probability", default='', metavar="COMP", required=True)
    parser.add_argument('--header', dest="header", help="Print header for optional score printing", default='', metavar="HEADER", required=False)
    parser.add_argument("--optional", dest="optional", action="store_true", help="Optional score printing (default: false)", required=False)
    parser.add_argument("--parasor", dest="parasor", action="store_true", help="Read ParasoR-formatted comp file (default: false)", required=False)
    parser.add_argument("--parsnorm", nargs='+', type=str, dest="parsnorm", help="Calc norm factor based on sequencing depth for raw read coverage (case, control. default: 1, 1)", required=False)
    parser.add_argument('--dir', dest="dir", help="Select directory of source files", default='./', metavar="DIR", required=False)
    # parser.add_argument('--float', dest="pfloat", action="store_true", required=False)
    parser.set_defaults(parasor=False, parsnorm=[])
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
        self.optional = arg.optional
        self.header = arg.header
        self.parsnorm = arg.parsnorm
        self.stemList = {}
        self.window = 20
        self.trim = 32
        self.get_stem_list()


    def get_stem_list(self):
        with open(self.comp) as f:
            if self.parasor:
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

    def no_id_in_stem_list(self, name):
        if name not in self.stemList:
            print("# No id")
            return True
        else:
            return False

    def get_stem_score(self, name, length):
        if self.no_id_in_stem_list(name):
            return []
        if self.parasor:
            stem = list(map(float, self.stemList[name].rstrip('\n').rstrip(']').lstrip('[').split(',')))
        else:
            table = {'.':-1, '(':1, ')':1, '[': 0.5, ']':0.5}
            stem = [table[i] if i in table else 0 for i in self.stemList[name]]
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
        return " ".join(contents[0:9])+" "+str(left)+" "+" ".join(contents[(self.window*2+9):(self.window*2+9+5)])+" "+str(right)
        # 0-3: index and global features, 5: control data, self.window*2: coverage profile, 5: case data, self.window*2: coverage profile.

    def transform_profile_to_peak_width(self, line, mincov, mindist = 2):
        contents = line.rstrip('\n').split(' ')
        sp, vp = [], []
        for cov in [mincov*1.5, mincov, mincov*0.5]:
            spwidth, smwidth, spos = self.comp_peak_width_up_and_down([float(contents[i]) for i in range(9, self.window*2+9)], float(contents[6]), cov, mindist)
            vpwidth, vmwidth, vpos = self.comp_peak_width_up_and_down([float(contents[i]) for i in range(14+self.window*2, 14+self.window*4)], float(contents[11+self.window]), cov, mindist)
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

    def get_mean_pars(self, rep1Vec, rep2Vec, norm = 1):
        return [np.log10(norm)+np.log10(np.mean(rep1Vec[max(0, i-2):min(len(rep1Vec), i+3)]+rep2Vec[max(0, i-2):min(len(rep1Vec), i+3)])+5.) for i in range(len(rep1Vec))]

    def extract_features(self, IDRVec, rep1Vec, rep2Vec, sumList = [], norm = 1):
        data = []
        meanList = [float(rep1Vec[i]+rep2Vec[i])/2. for i in range(len(IDRVec))]
        varList = [np.var([rep1Vec[i], rep2Vec[i]]) for i in range(len(IDRVec))]
        parsList = self.get_mean_pars(rep1Vec, rep2Vec, norm)
        assert len(IDRVec) == len(rep1Vec)
        assert len(rep1Vec) == len(rep2Vec)
        for i in range(len(IDRVec)):
            values = [IDRVec[i], parsList[i], meanList[i], varList[i], float(sum(rep1Vec)+sum(rep2Vec))/(2.0*len(rep1Vec))]
            vec = self.mean_around(meanList, i, self.window)
            values.extend(vec)
            data.append(np.asarray(values))
        return data

    def extract_features_double_columns(self, key, rep1Vec, rep2Vec, iters = [], norm = 1.):
        max_norm = max(1, max([max(vec) for vec in [rep1Vec, rep2Vec]]))
        assert len(rep1Vec) == len(rep2Vec)
        data = [[x[i] for x in [rep1Vec, rep2Vec]] for i in range(len(rep1Vec))]
        result = [[np.mean(data[i])] for i in range(len(rep1Vec))]
        for i in range(len(rep1Vec)):
            result[i].append(np.mean([(vec[0]+vec[1])/2.0 for vec in data[max(0, i-2):(i+2)]]))
        for i in range(len(rep1Vec)):
            result[i].append(result[i][0]/norm)
        for i in range(len(rep1Vec)):
            result[i].append(result[i][0]/max_norm)
        return np.array(result)

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
                if self.parasor:
                    out = np.append(min(i, len(sumList)-i), stem[i])
                else:
                    out = np.append(i, stem[i])
                out = np.append(out, sumList[i])
                out = np.append(out, max(-7, min(7, float(caseTable[i][1])-float(contTable[i][1]))))
                out = np.append(out, contTable[i])
                out = np.append(out, caseTable[i])
                if self.peak > 0:
                    line = " ".join(["%.4f" % float(n) for n in out])
                    print(self.transform_profile_to_peak_width(line, float(self.peak)/100.0))
                else:
                    print(" ".join(["%.4f" % float(n) for n in out]))

    def print_score_and_feature_double_columns(self, key, sumList, caseTable, contTable, icSHAPE = []):
        stem = self.get_stem_score(key, len(sumList))
        threshold = self.cutoff
        if len(stem) == 0:   return
        for i in range(len(sumList)):
            if abs(sumList[i]) > threshold:
                out = np.append(contTable[i], caseTable[i])
                if len(icSHAPE):
                    out = np.append(out, icSHAPE[i])
                print(" ".join(["%.4f" % float(n) for n in out]))


    # def extract_dictionaries(self, keys, sidrfile, sscorefile, vidrfile, vscorefile):
    #     sIDR = utility.extract_dict(utility.parse_idr(self.dir+sidrfile), keys)
    #     vIDR = utility.extract_dict(utility.parse_idr(self.dir+vidrfile), keys)
    #     srep1, srep2 = utility.parse_score(self.dir+sscorefile)
    #     srep1 = utility.extract_dict(srep1, keys)
    #     srep2 = utility.extract_dict(srep2, keys)
    #     vrep1, vrep2 = utility.parse_score(self.dir+vscorefile)
    #     vrep1 = utility.extract_dict(vrep1, keys)
    #     vrep2 = utility.extract_dict(vrep2, keys)
    #     return sIDR, vIDR, srep1, srep2, vrep1, vrep2

    # def get_feature_both_scores(self, keys, sidrfile, sscorefile, vidrfile, vscorefile):
    #     sIDR, vIDR, srep1, srep2, vrep1, vrep2 = self.extract_dictionaries(keys, sidrfile, sscorefile, vidrfile, vscorefile)
    #     header = " ".join(["Index", "Stem", "Sum_coverage", "PARS", "IDR_s", "PARS_s", "Mean_s", "Var_s", "Tran_Mean_s"]
    #         +[str(i)+str(t)+str(n) for n in range(1, self.window+1) for i in ["s"] for t in ["m", "p"]]
    #         +["IDR_v", "PARS_v", "Mean_v", "Var_v", "Tran_Mean_v"]
    #         +[str(i)+str(t)+str(n) for n in range(1, self.window+1) for i in ["v"] for t in ["m", "p"]])
    #     if self.peak > 0:
    #         print(self.get_trimmed_output(header.split(' '), " ".join(["Swidth", "Swidth2", "Swidth3"]), " ".join(["Vwidth", "Vwidth2", "Vwidth3"])))
    #     else:
    #         print(header)
    #     for key in list(keys):
    #         print("#", key)
    #         if key in srep1 and key in srep2 and key in vrep1 and key in vrep2:
    #             sumList = [srep1[key][i]+srep2[key][i]+vrep1[key][i]+vrep2[key][i] for i in range(len(sIDR[key]))]
    #             contTable = self.extract_features(sIDR[key], srep1[key], srep2[key])
    #             caseTable = self.extract_features(vIDR[key], vrep1[key], vrep2[key])
    #             self.print_score_and_feature(key, sumList, caseTable, contTable)
    #         else:
    #             utility.log("# No key: "+key)
    #             utility.log(key in srep1)
    #             utility.log(key in srep2)
    #             utility.log(key in vrep1)
    #             utility.log(key in vrep2)

    def extract_dictionaries_idr_iterator(self, sidrfile, vidrfile):
        sIDR = utility.parse_idr_iterator(self.dir+sidrfile)
        vIDR = utility.parse_idr_iterator(self.dir+vidrfile)
        return sIDR, vIDR

    def extract_dictionaries_iterator_double_columns(self, sscorefile, vscorefile):
        sscore = utility.parse_score_iterator(self.dir+sscorefile)
        vscore = utility.parse_score_iterator(self.dir+vscorefile)
        return sscore, vscore

    def calc_pars_norm_factor(self, srep, vrep):
        ks = max(sum([sum(t[1])+sum(t[2]) for t in srep]), 1)
        kv = max(sum([sum(t[1])+sum(t[2]) for t in vrep]), 1)
        return (ks+kv)/(2.*float(ks)), (ks+kv)/(2.*float(kv))

    def extract_dictionaries_iterator(self, sidrfile, sscorefile, vidrfile, vscorefile):
        sIDR, vIDR = self.extract_dictionaries_idr_iterator(sidrfile, vidrfile)
        if len(self.parsnorm) == 2:
            # ks, kv = self.calc_pars_norm_factor(self.parsnorm[0], self.parsnorm[1])
            ks, kv = float(self.parsnorm[1]), float(self.parsnorm[0])
        else:
            ks, kv = 1., 1.
        srep, vrep = self.extract_dictionaries_iterator_double_columns(sscorefile, vscorefile)
        return sIDR, vIDR, srep, vrep, ks, kv

    def get_feature_both_scores_iterator(self, sidrfile, sfiles, vidrfile, vfiles):
        sscorefile, vscorefile = sfiles[0], vfiles[0]
        sIDR_list, vIDR_list, srep_list, vrep_list, ks, kv = self.extract_dictionaries_iterator(sidrfile, sscorefile, vidrfile, vscorefile) # key, value, value.
        header = " ".join(["Index", "Stem", "Sum_coverage", "PARS", "IDR_s", "PARS_s", "Mean_s", "Var_s", "Tran_Mean_s"]
            +[str(i)+str(t)+str(n) for n in range(1, self.window+1) for i in ["s"] for t in ["m", "p"]]
            +["IDR_v", "PARS_v", "Mean_v", "Var_v", "Tran_Mean_v"]
            +[str(i)+str(t)+str(n) for n in range(1, self.window+1) for i in ["v"] for t in ["m", "p"]])
        if self.peak > 0:
            print(self.get_trimmed_output(header.split(' '), " ".join(["Swidth", "Swidth2", "Swidth3"]), " ".join(["Vwidth", "Vwidth2", "Vwidth3"])))
        else:
            print(header)
        for sIDR, vIDR, srep, vrep in itertools.zip_longest(sIDR_list, vIDR_list, srep_list, vrep_list):
            key = sIDR[0]
            assert(sIDR[0] == vIDR[0] and vIDR[0] == srep[0] and srep[0] == vrep[0])
            if self.parasor and key[0] != "N":   continue
            print("#", key, len(srep[1]))
            if self.no_id_in_stem_list(key):
                continue
            sumList = [srep[1][i]+srep[2][i]+vrep[1][i]+vrep[2][i] for i in range(len(sIDR[1]))]
            if np.mean(sumList) < 10:    continue
            contTable = self.extract_features(sIDR[1], srep[1], srep[2], ks)
            caseTable = self.extract_features(vIDR[1], vrep[1], vrep[2], kv)
            self.print_score_and_feature(key, sumList, caseTable, contTable)

    def calc_icshape_norm_factor(self, scorefile):
        srep_list = utility.parse_score_iterator(self.dir+scorefile)
        all = []
        for vecs in srep_list:
            for vec in vecs[1:]:
                vec = vec[self.trim:max(self.trim, len(vec)-self.trim)]
            # vec = [x for x in vec if x > 0]
                all += vec
        all.sort()
        high = [all[i] for i in range(int(len(all)*90./100.), int(len(all)*95./100.))]
        if len(high) == 0:
            return 1.
        else:
            return max(1, np.mean(high))

    def calc_icshape_reactivity(self, contTable, caseTable, alpha=0.25):
        background = float(max(1, sum([ x for x in range(self.trim, max(self.trim, np.shape(contTable)[0]))])))
        icSHAPE = [(caseTable[i][2]-contTable[i][2]*alpha)/(max(1., contTable[i][0])/background) for i in range(np.shape(contTable)[0])]
        sortl = sorted(icSHAPE[self.trim:max(self.trim, len(icSHAPE)-self.trim)])
        if len(sortl) == 0:
            return [0.]*len(icSHAPE)
        tmin, tmax = sortl[max(0, int(5./100.*len(sortl))-1)], sortl[min(len(sortl)-1, int(95./100.*len(sortl))-1)]
        if tmin == tmax:
            return [0.]*len(icSHAPE)
        return [max(0., min(1., (x-tmin)/(tmax-tmin))) for i, x in enumerate(icSHAPE)]

    def calc_icshape_norm(self, sscorefile, vscorefile):
        if True:
            return self.calc_icshape_norm_factor(sscorefile), self.calc_icshape_norm_factor(vscorefile)
        else:
            return 1., 1.

    def get_feature_both_scores_iterator_double_columns(self, sfiles, vfiles):
        assert len(sfiles) >= 1 and len(vfiles) >= 1
        sscorefile, vscorefile = sfiles[0], vfiles[0]
        snorm, vnorm = self.calc_icshape_norm(sscorefile, vscorefile)
        sys.stderr.write("icshape_norm "+str(snorm)+" "+str(vnorm))
        srep_list, vrep_list = self.extract_dictionaries_iterator_double_columns(sscorefile, vscorefile)
        siters, viters = [], []
        for i in range(1, len(sfiles)):
            sscore, vscore = self.extract_dictionaries_iterator_double_columns(sfiles[i], vfiles[i])
            siters.append(sscore)
            viters.append(vscore)
        print(self.header)
        for srep, vrep in itertools.zip_longest(srep_list, vrep_list):
            key = srep[0]
            assert(srep[0] == vrep[0])
            if self.parasor and key[0] != "N":   continue
            print("#", key, len(srep[1]))
            if self.no_id_in_stem_list(key):
                continue
            sumList = [srep[1][i]+srep[2][i]+vrep[1][i]+vrep[2][i] for i in range(len(srep[1]))]
            if np.mean(sumList) < 10:    continue
            contTable = self.extract_features_double_columns(key, srep[1], srep[2], siters, snorm)
            caseTable = self.extract_features_double_columns(key, vrep[1], vrep[2], viters, vnorm)
            icSHAPE = self.calc_icshape_reactivity(contTable, caseTable)
            self.print_score_and_feature_double_columns(key, sumList, caseTable, contTable, icSHAPE)

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
            # self.get_feature_both_scores(self.idrFiles[1], self.control, self.idrFiles[0], self.case)
            if self.optional:
                self.get_feature_both_scores_iterator_double_columns(self.control, self.case)
            else:
                assert len(self.idrFiles) == 2
                self.get_feature_both_scores_iterator(self.idrFiles[1], self.control, self.idrFiles[0], self.case)


if __name__ == '__main__':
    parser = get_parser()
    try:
        options = parser.parse_args()
        featureSelect = featureSelection(options)
        featureSelect.get_feature_table()
    except:
        parser.print_help()
        sys.exit(0)
