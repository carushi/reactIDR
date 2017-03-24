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
    parser.add_argument('--control', nargs='+', dest="control", help="Read score data for control samples (score file, coverage file, ...)", default='', metavar="CONT", required=False)
    parser.add_argument('--sample', dest="sample", help="Carry out sampling for print a result", default=1.0, metavar="SAMPLE", type=float, required=False)
    parser.add_argument('--cutoff', dest="cutoff", help="Threshold for the minimum rpkm", type=int, default=-1, metavar="THRESHOLD", required=False)
    parser.add_argument('--peak', dest="peak", help="Print peak width around the position with the minimum coverage [PEAK]", type=int, default=0, metavar="PEAK", required=False)
    parser.add_argument('--trim', dest="trim", help="Trim first and list N bases from normalization", type=int, metavar="TRIM", required=False, default=32)
    parser.add_argument('--comp', dest="comp", help="Set File name for stem probability", default='', metavar="COMP", required=True)
    parser.add_argument("--parasor", dest="parasor", action="store_true", help="Read ParasoR-formatted comp file (default: false)", required=False)
    parser.add_argument("--total", nargs='+', type=str, dest="total", help="Calc norm factor based on sequencing depth for raw read coverage (case, control. default: 1, 1)", required=False)
    parser.add_argument("--parsnormed", nargs='+', type=str, dest="parsnorm", help="Calc norm factor based on sequecing depth for raw read coverage (None -> PARS score is already normalized)", required=False)
    parser.add_argument('--dir', dest="dir", help="Select directory of source files", default='./', metavar="DIR", required=False)
    parser.set_defaults(parasor=False, total=[], parsnormed=[])
    return parser


def empty_iter():
    raise StopIteration
    yield

def detect_peak_width(svec, mincov, mindist = 2):
    prev, count = -1, mindist
    peak = [0]*len(svec)
    index = 0
    while index <= len(svec):
        if index == len(svec) or svec[index] < mincov:
            if prev >= 0:
                within_range = [j for j in range(index+1, min(index+1+count, len(svec))) if svec[j] >= mincov]
                if  len(within_range) > 0 :
                    index, count = min(within_range), count-(min(within_range)-index)
                    continue
                else:
                    peak[prev:index] = [index-prev]*(index-prev)
                    prev, count = -1, mincov
            else:
                index += 1
                continue
        else:
            if prev < 0:
                prev = index
            index += 1
    return peak

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
        self.parsnorm = arg.parsnorm
        self.stemList = {}
        self.window = 20
        self.trim = arg.trim
        self.get_stem_list()
        self.old = arg.old
        self.set_header()

    def set_header(self):
        features = ['Transcript', 'Index', 'Sum', 'PARS', 'icSHAPE', 'LDR']
        features += [x+"_"+cond for x in ['Sum', 'Average_RT', 'Average_Cov', 'Peak'] for cond in ['c', 'b']]
        features += [x+"_"+cond for x in ['LDR'] for cond in ['c', 'b']]
        self.header = "\t".join(features)
        features.insert(2, 'Stem')
        self.pheader = "\t".join(features)

    def get_stem_list(self):
        if self.comp == "": return
        with open(self.comp) as f:
            if self.parasor: #ParasoR output
                while True:
                    line = f.readline()
                    if line == "":  break
                    name = line.rstrip('\n').split(' ')[2]
                    self.stemList[name] = f.readline()
            else: #fasta file
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
        if len(self.stemList.keys()) > 0 and name not in self.stemList:
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

    def get_mean_pars(self, rep1Vec, rep2Vec, norm = 1):
        return [np.log10(norm)+np.log10(np.mean(rep1Vec[max(0, i-2):min(len(rep1Vec), i+3)]+rep2Vec[max(0, i-2):min(len(rep1Vec), i+3)])+5.) for i in range(len(rep1Vec))]

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

    def top_90_95(self, vec):
        vec.sort()
        high = [vec[i] for i in range(math.floor(len(vec)*90./100.), math.ceil(len(vec)*95./100.))]
        if len(high) == 0:
            return 1.
        else:
            return np.mean(high)

    def calc_icshape_reactivity(self, back, target, back_cov, alpha = 0.25, subtract = False):
        assert len(back) == len(back_cov) and len(back) == len(target)
        scores = [None]*len(back)
        for i in range(len(back)-1):
            b, bc, t = sum(back[i]), sum(back[i+1]), sum(back[i])
            if subtract:
                bc = bc-b
            if bc > 0:
                scores[i] = (t-alpha*b)/bc
        sortl = [x for x in scores if x is not None].sorted()
        if len(sortl) == 0:
            return [0.]*len(back)
        tmin, tmax = sortl[max(0, math.floor(5./100.*len(sortl))-1)], sortl[min(len(sortl)-1, math.ceil(95./100.*len(sortl))-1)]
        if tmin == tmax:
            return [0.]*len(back)
        return [max(0., min(1., (x-tmin)/(tmax-tmin))) if x is not None else None for i, x in enumerate(scores)]

    def normalized_vector(self, vec, start = 32, end = 32):
        trimmed_vec = [x/lnorm for i, x in enumerate(zip(t, v)) for t, v in vec if i >= start and i < len(t)-end]
        norm = self.top_90_95(trimmed_vec, lnorm)
        return [x/norm for x in vec]

    def calc_icshape_score(self, srep, scov, vrep, stotal, vtotal):
        srep = [self.normalized_vector(svec, self.trim, self.trim, stotal) for svec in srep]
        if len(scov) == 0:
            scov = srep
        else:
            scov = [self.normalized_vector(bvec, self.trim, self.trim) for bvec in scov]
        vrep = [self.normalized_vector(vvec, self.trim, self.trim, vtotal) for vvec in vrep]
        return self.calc_icshape_reactivity(srep, scov, vrep)


    def calc_total_read_counts(self, ssfile, vsfile):
        if len(self.parsnorm) == 2:
            stotal, vtotal = float(self.parsnorm[1]), float(self.parsnorm[0])
        else:
            srep, vrep = self.extract_dictionaries_iterator_double_columns(ssfile, vsfile)
            stotal = max(sum([sum(t[1])+sum(t[2]) for t in srep]), 1)
            vtotal = max(sum([sum(t[1])+sum(t[2]) for t in vrep]), 1)
        return stotal, vtotal

    def calc_pars_norm_factor(self, stotal, vtotal):
        return (stotal+vtotal)/(2.*stotal), (stotal+vtotal)/(2.*vtotal)

    def extract_dictionaries_idr_iterator(self, sidrfile, vidrfile):
        sIDR = utility.parse_idr_iterator(self.dir+sidrfile)
        vIDR = utility.parse_idr_iterator(self.dir+vidrfile)
        return sIDR, vIDR

    def extract_dictionaries_score_iterator(self, sscorefile, vscorefile):
        sscore = utility.parse_score_iterator(self.dir+sscorefile)
        vscore = utility.parse_score_iterator(self.dir+vscorefile)
        return sscore, vscore

    def extract_dictionaries_iterator(self, sifile, ssfile, vifile, vsfile, scfile = []):
        sIDR, vIDR = self.extract_dictionaries_idr_iterator(sifile, vifile)
        srep, vrep = self.extract_dictionaries_score_iterator(ssfile, vsfile)
        if len(scfile) > 0:
            scov = self.extract_dictionaries_iterator_coverage(scfile[0])
        else:
            scov = empty_iter()
        if len(scfile) > 1:
            vcov = self.extract_dictionaries_iterator_coverage(scfile[1])
        else:
            vcov = empty_iter()
        return sIDR, vIDR, srep, vrep, scov, vcov

    def estimate_rpkm(self, srep, vrep, stotal, vtotal):
        srpkm = sum([sum(t) for t in srep])
        vrpkm = sum([sum(t) for t in vrep])
        return srpkm/len(srep[1])*10000000000/stotal

    def extract_default_features(self, svec, vvec, scov, vcov, rpkm):
        assert len(svec) == len(vvec)
        result_c = [[None]*4]*len(svec)
        result_t = [[None]*4]*len(svec)
        for i in range(len(svec)):
            result_c[i][0] = sum(svec[i])
            result_t[i][0] = sum(vvec[i])
        width = 5
        for i in range(len(svec)):
            start, end = max(0, i-(width-1)/2), min(len(svec), i+(width-1)/2+1)
            result_c[i][1] = np.mean(svec[start:end])
            result_t[i][1] = np.mean(vvec[start:end])
        if len(scov) > 0 and len(vcov) > 0:
            scov, vcov = [sum(x) for x in zip(*scov)], [sum(x) for x in zip(*scov)]
            for i in range(len(svec)):
                start, end = max(0, i-(width-1)/2), min(len(svec), i+(width-1)/2+1)
                result_c[i][2] = np.mean(scov[start:end])
                result_t[i][2] = np.mean(vcov[start:end])
        for x in self.detect_peak_range(svec, np.median(svec)):
            result_c[i][3] = x
        for x in self.detect_peak_range(vvec, np.median(vvec)):
            result_t[i][3] = x
        return np.array(result)

    def calc_ldr(self, s1, v1, control = False):
        if control:
            result_c = ''
            s1.sort()
            for i in range(len(s1)):
                for j in range(i+1, len(s1)):
                    result_c += str(np.log10(s1[j]/s1[i]))+","
            return result_c[0:-1]
        else:
            result_t = ''
            for s in s1:
                for v in v1:
                    result_t += str(np.log10(s/v))+","
            return result_t[0:-1]

    def calc_ldr_features(self, s1, v1):
        return [sum(s)/sum(v) for s, v in zip(s1, v1)]

    def calc_ldr_distribution(self, svec, vvec, scov, vcov):
        if len(scov) == 0 or len(vcov) == 0:
            return ['']*len(svec), ['']*len(svec)
        s1 = [[svec[i][j]/max(1., scov[i][j]) for j in range(len(svec[i]))] for i in range(len(svec))]
        v1 = [[vvec[i][j]/max(1., vcov[i][j]) for j in range(len(vvec[i]))] for i in range(len(vvec))]
        ldr = self.calc_ldr_features(s1, v1)
        ldr_c = [self.calc_ldr(s1[i], s1[i]) for i in range(len(svec))]
        ldr_t = [self.calc_ldr(s1[i], v1[i], True) for i in range(len(svec))]
        return [ldr, ldr_c, ldr_t]

    def calc_pars_single(self, score, width = 5.):
        return [np.log10(sum([ x/5.+1 for x in range(max(0, i-width/2+1), min(len(score), i+width/2+1)) ])) for i in range(len(score)) ]

    def calc_pars_features(self, sscore, vscore, ks, kv):
        func = sum
        sscore, vscore = [ks*func(temp) in temp for temp in zip(*sscore)], [kv*func(temp) for temp in zip(*vscore)]
        width = 5.
        s1 = self.calc_pars_single(sscore)
        v1 = self.calc_pars_single(vscore)
        return [v-s for s, v in zip(s1, v1)]

    def combine_dataset(self, dlist):
        return dlist

    def print_features(self, id, sumList, features, icshape, pars, ldr):
        for i in range(len(sumList)):
            print(" ".join([str(id), i, sumList[i], pars[i], icshape[i], ldr[0][i]]), end=" ")
            print(" ".join(["%.4f" % float(n) for n in features[i] for i in len(features)]), end=" ")
            print(" ".join(["%.4f" % float(n) for n in [ldr[1][i], ldr[2][i]]]))


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

    def get_feature_reactivity_scores(self, sifile, sfiles, vifile, vfiles, keys, refseq = False):
        assert len(sfiles) > 1 and len(vfiles) > 0
        ssfile, vsfile = sfiles[0], vfiles[0]
        sIDR_list, vIDR_list, srep_list, vrep_list, scov_list, vcov_list = self.extract_dictionaries_iterator(sifile, ssfile, vidrfile, vsfile, sfiles[1:]) # key, value, value.
        stotal, vtotal = self.calc_total_read_counts(ssfile, vsfile)
        ks, kv = self.calc_pars_norm_factor(stotal, vtotal)
        sys.stderr.write("PARS normalization: "+str(stotal)+" "+str(vtotal))
        if len(self.stemList) > 0:
            print(self.pheader)
        else:
            print(self.header)
        id = 0
        for sIDR, vIDR, srep, vrep, scov, vcov in itertools.zip_longest(sIDR_list, vIDR_list, srep_list, vrep_list, scov_list, vcov_list, fillvalue = [[]]):
            key = srep[0]
            assert(srep[0] == vrep[0])
            print("#", key, len(srep[1]))
            if (refseq and key[0] != "N") or (key not in keys and len(keys) > 0): continue
            if self.no_id_in_stem_list(key): continue
            rpkm = self.estimate_rpkm(stotal)
            if self.cutoff > 0 and rpkm < self.cutoff:
                print("# low expression", rpkm)
                continue
            print("#", key, len(srep[1]), id, rpkm)
            id += 1
            svec, vvec = [x for x in zip(*srep[1:])], [x for x in zip(*vrep[1:])]
            sumList = [sum(srep[1:][i])+sum(vrep[1:][i]) for i in range(len(srep[1]))]
            features = self.extract_default_features(svec, vvec, scov, vcov)
            icshape = self.calc_icshape_score(srep[1:], vrep[1:], scov[1:], stotal, vtotal)
            pars = self.calc_pars_features(srep[1:], vrep[1:], ks, kv)
            ldr = self.calc_ldr_features(svec, vvec, scov[1:], vcov[1:])
            self.print_features(id, sumList, features, icshape, pars, ldr)

    def get_feature_table(self):
        if self.sample > 0:
            filt = "_sampled"
            keys = random.sample(list(self.stemList.keys()), self.sample)
        else:
            filt = ""
            keys = list(self.stemList.keys())
        assert len(self.idrFiles) == 2
        self.get_feature_reactivity_scores(self.idrFiles[1], self.control, self.idrFiles[0], self.case, keys)


if __name__ == '__main__':
    parser = get_parser()
    try:
        options = parser.parse_args()
        featureSelect = featureSelection(options)
        featureSelect.get_feature_table()
    except:
        parser.print_help()
        sys.exit(0)
