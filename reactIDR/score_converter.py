import sys
import random
import math
import numpy as np
import itertools
import argparse
import utility
import os.path

def empty_iter():
    raise StopIteration
    yield

def ordered_common_transcript(dicts):
    if len(dicts) > 2:
        return sorted(list(utility.common_transcript(dicts[0], dicts[1], dicts[2])))
    else:
        return sorted(list(utility.common_transcript(dicts[0], dicts[1])))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("rtFiles", nargs='+', type=str, help="RT stop count data. (e.g. FILE1=case or S1 (, FILE2 control or V1))", metavar="RTFILES")
    parser.add_argument("--bed", dest="bed", action="store_true", help="read data in bed format.", required=False)
    parser.add_argument("--merge", dest="merge", action="store_true", help="Combine argument files and keep common transcript. (control1,control2,control3 case1,case2,case3 -> output1,output2")
    parser.add_argument("--dir", type=str, help="Directory for RT stop count and coverage data.", metavar="DIR", required=False)
    parser.add_argument("--coverage", dest="coverage", nargs='+', help="RT stop coverage data.", metavar="COVERAGE", required=False)
    parser.add_argument("--skip_header", dest="header", action="store_true", help="header is included in input.", required=False)
    # parser.add_argument("--control", dest="control", type=str, help="Non-treated RT stop coverage data.", metavar="CONTROL", required=False)
    parser.add_argument("--score", dest="score", type=str, help="use reactivity scores computed from case and control (default: raw -> pars, icshape, other=LDR)", required=False)
    parser.add_argument("--print_all", dest="print", action="store_true", help="print all use reactivity scores computed from case and control (default: raw -> PARS, icSHAPE, LDR)", required=False)
    parser.add_argument("--pvalue", dest="ndist", type=str, help="null distribution for pvalue computation (coverage, rank, poisson, nb, zinb, nb-cisgenome, and ec (estimated coverage for downstream normalization))", default='', metavar="NDIST", required=False)
    parser.add_argument("--output", dest="output", type=str, help="output filename prefix (prefix_case_nb.tab, prefix_cont_poisson.tab, prefix_icshape.tab...)", required=False, default="converted_score")
    parser.add_argument("--case_norm", dest="case_norm", type=str, help="Comma-separated normalization constant for treated sample.", required=False)
    parser.add_argument("--cont_norm", dest="cont_norm", type=str, help="Comma-separated normalization constant for non-treated sample.", required=False)
    parser.add_argument("--reverese", dest="reverese", action="store_true", help="If this option is set, the program deals files without assuming that each transcript appears in each file on the same line.", required=False)
    parser.add_argument("--variable_len", dest="variable", action="store_true", help="If this option is enabled, the program ignores assertion about the different length of data arrays of each transcript.", required=False)
    parser.add_argument("-k", dest="k", type=int, help="Minimum rt count.", required=False, default=1)
    parser.add_argument("-t", dest="t", type=int, help="Minimum read coverage.", required=False, default=1)
    parser.add_argument("--integrated", dest="integrated", action="store_true", help="Compute single reactivity by summing the replicates.", required=False)
    parser.add_argument("--alpha", dest="alpha", type=float, help="alpha variable for icshape scoring.", required=False)
    parser.set_defaults(dir="./", merge=False, case_norm="", cont_norm="", alpha=0.25, reverse=False)
    return parser

def slide_coverage(vec, offset):
    if offset == 0: return vec
    aft = [None]*len(vec)
    for i in range(len(vec)):
        if i+offset >= 0 and i+offset < len(vec):
            aft[i] = vec[i+offset]
    return aft

def count_log_ratio(rt, cov, offset):
    cov = slide_coverage(cov, offset)
    for i in range(len(rt)):
        if rt[i] is not None and cov[i] is not None:
            if cov[i] < rt[i]:
                sys.stderr.write("Coverage error at "+str(i)+" th, "+str(rt[i])+"/"+str(cov[i])+"\n")
                w=2
                sys.stderr.write(",".join(map(str, rt[i-w:i+w+1]))+" "+",".join(map(str, cov[i-w:i+w+1]))+"\n")
                cov[i] = rt[i]
    return [np.log2(rt[j]/max(1., cov[j])) if abs(rt[j]) > 0 and cov[j] is not None else None for j in range(len(rt))]

def count_ratio_ldr(rt, cov, offset, t=1, k=1):
    return [rt[j]/max(1., cov[j-1]) if j > 0 and rt[j] > k and cov[j] > t else None for j in range(len(rt))]

def calc_ldr_features(tdata, ndata, tcdata, ncdata, t=1, k=1):
    return [np.log2(t/n) if abs(t) > 0 and abs(n) > 0 else None for t, n in zip(count_ratio(tdata, ncdata, t, k), count_ratio(ndata, ncdata, t, k))]

def calc_pars_norm_factor(vlist, slist):
    return [(stotal+vtotal)/(2.*vtotal) for vtotal, stotal in zip(vlist, slist)], [(stotal+vtotal)/(2.*stotal) for vtotal, stotal in zip(vlist, slist)]

def averaging_score(score, width=5.):
    length = int(width/2)
    return [np.log10(sum([ score[x]/5.+1 for x in range(max(0, i-length), min(len(score), i+length+1)) ])) for i in range(len(score)) ]

def calc_pars_features(sscore, vscore, ks, kv):
    sscore, vscore = ks*np.array(sscore), kv*np.array(vscore)
    s1 = averaging_score(sscore)
    v1 = averaging_score(vscore)
    return np.clip([v-s for s, v in zip(s1, v1)], -7, 7)

def common_transcript(dict1, dict2):
    return set(dict1.keys()) & set(dict2.keys())

def top_90_95(vec, start=32, end=32):
    temp = vec[start:len(vec)-end+1]
    temp.sort()
    high = [temp[i] for i in range(int(math.floor(len(temp)*90./100.)), int(math.ceil(len(temp)*95./100.)))]
    if len(high) == 0:
        return 1.
    else:
        return np.mean(high)

def append_none_vector(vec, max_len):
    if len(vec) < max_len:
        return list(vec)+[0]*(max_len-len(vec))
    else:
        return vec

def append_none(tdata, tcdata, ndata, ncdata):
    max_len = max([len(tdata), len(tcdata), len(ndata), len(ncdata)])
    tdata = append_none_vector(tdata, max_len)
    tcdata = append_none_vector(tcdata, max_len)
    ndata = append_none_vector(ndata, max_len)
    ncdata = append_none_vector(ncdata, max_len)
    return tdata, tcdata, ndata, ncdata

def calc_icshape_reactivity(tdata, tcdata, ndata, ncdata, alpha=0.25, threshold=200, offset=0, ignore=False):
    if ignore:
        tdata, tcdata, ndata, ncdata = append_none(tdata, tcdata, ndata, ncdata)
    assert len(tdata) == len(ndata) and len(tdata) == len(ncdata)
    scores = [None]*len(tdata)
    for i in range(len(ncdata)-offset):
        t, b, bc = tdata[i], ndata[i], ncdata[i+offset]
        if bc > threshold:
            scores[i] = (t-alpha*b)/bc
    sortl = sorted([x for x in scores if x is not None])
    if len(sortl) == 0:
        return [0.]*len(ncdata)
    tmin, tmax = sortl[max(0, int(math.floor(5./100.*len(sortl))-1))], sortl[min(len(sortl)-1, int(math.ceil(95./100.*len(sortl))-1))]
    if tmin == tmax:
        return [0.]*len(ncdata)
    return [np.clip((x-tmin)/(tmax-tmin), 0, 1) if x is not None else None for i, x in enumerate(scores)]

def normalized_vector(vec, start=5, end =30, nstart=32, nend=32):
    trimmed_vec = [x for i, x in enumerate(vec) if i >= start and i < len(vec)-end]
    norm = top_90_95(trimmed_vec)
    if norm == 0.:
        norm = 1.0
    return [x/norm for x in vec]


def key_data_iterator(tfile, nfile, dir="./"):
    tlist = utility.parse_score_tri_iterator(os.path.join(dir, tfile))
    nlist = utility.parse_score_tri_iterator(os.path.join(dir, nfile))
    for (tkey, tdata), (nkey, ndata) in zip(tlist, nlist):
        assert tkey == nkey
        yield tkey, tdata, ndata

def key_data_cov_iterator(tfile, nfile, tcfile, ncfile, dir="./", offset=0):
    tlist = utility.parse_score_tri_iterator(os.path.join(dir, tfile))
    tclist = utility.parse_score_tri_iterator(os.path.join(dir, tcfile))
    nlist = utility.parse_score_tri_iterator(os.path.join(dir, nfile))
    nclist = utility.parse_score_tri_iterator(os.path.join(dir, ncfile))
    for (tkey, tdata), (nkey, ndata), (tckey, tcdata), (nckey, ncdata) in zip(tlist, nlist, tclist, nclist):
        assert tkey == nkey and tckey == nckey and tkey == tckey
        yield tkey, tdata, ndata, tcdata, ncdata

def check_average_hit(key, tlist, nlist, ave_hit):
    tave = sum([np.mean(temp) for temp in tlist])
    nave = sum([np.mean(temp) for temp in nlist])
    flag = (tave < ave_hit or nave < ave_hit)
    if flag: print("low average hit", key, min(tave, nave), "<", ave_hit)
    return flag, min(tave, nave)

class Converter:
    """docstring for Converter"""
    def __init__(self, arg, debug=False):
        self.arg = arg
        self.score_list = ['pars', 'icshape', 'ldr']
        self.vtotal, self.stotal = [], []
        self.ttotal, self.ntotal = [], []
        self.t, self.k = self.arg.t, self.arg.k
        self.cutoff = 200
        self.trim = 32
        self.debug = debug
        self.verbose = True

    def get_output_name(self, i=-1):
        suffix = ""
        if self.arg.integrated:
            suffix = "_integ"
        main = self.arg.ndist
        if len(self.arg.ndist) == 0:
            main = self.arg.score
        if i < 0:
            return self.arg.output+"_"+main+suffix+".tab"
        else:
            if i <= 1:
                base = ["case", "cont"][i]
            else:
                base = ".".join(self.arg.rtFiles[i].split('.')[:-1])
            return self.arg.output+"_"+main+suffix+"_"+base+".tab"

    def calc_total_read_counts(self):
        if len(self.arg.case_norm) > 0 and len(self.arg.cont_norm) > 0:
            ttotal, ntotal = list(map(float, self.arg.case_norm.split(','))), list(map(float, self.arg.cont_norm.split(',')))
        else:
            ttotal, ntotal = [], []
            for _, tcount, ncount in key_data_iterator(self.arg.rtFiles[0], self.arg.rtFiles[1], self.arg.dir):
                if len(ttotal) == 0:
                    ttotal = [sum(t) for t in tcount]
                    ntotal = [sum(n) for n in ncount]
                else:
                    ttotal = np.sum([ttotal, [sum(t) for t in tcount]], axis=0)
                    ntotal = np.sum([ntotal, [sum(n) for n in ncount]], axis=0)
        if self.arg.integrated:
            ttotal, ntotal = [sum(ttotal)], [sum(ntotal)]
        return ttotal, ntotal

    def compute_pars_score(self, vlist, slist):
        if self.arg.integrated:
            return [calc_pars_features(np.sum(vlist, axis=0), np.sum(slist, axis=0), self.vtotal, self.stotal)]
        else:
            return [calc_pars_features(vdata, sdata, self.vtotal[i], self.stotal[i]) for i, (vdata, sdata) in enumerate(zip(vlist, slist))]

    def calc_icshape_features(self, tdata, ndata, tcdata, ncdata, index):
        ttotal = self.ttotal[index]
        ntotal = self.ntotal[index]
        tdata = normalized_vector(tdata, self.trim, self.trim)
        ndata = normalized_vector(ndata, self.trim, self.trim)
        return calc_icshape_reactivity(tdata, tcdata, ndata, ncdata, threshold=200/len(self.ttotal), ignore=self.arg.variable)

    def compute_icshape_score(self, key, tlist, nlist, tclist, nclist, ave_hit, print_all=True):
        flag = check_average_hit(key, tlist, nlist, ave_hit)
        if self.arg.integrated:
            if flag and not print_all:
                return [[None for i in range(len(tlist[0]))]]
            return [self.calc_icshape_features(np.sum(tlist, axis=0), np.sum(nlist, axis=0), np.sum(tclist, axis=0), np.sum(nclist, axis=0), 0)]
        else:
            if flag and not print_all:
                return [[None for i in range(len(temp))] for temp in tlist]
            return [self.calc_icshape_features(tdata, ndata, tcdata, ncdata, i) for i, (tdata, ndata, tcdata, ncdata) in enumerate(zip(tlist, nlist, tclist, nclist))]

    def compute_ldr_score(self, tlist, nlist, tclist, nclist):
        if self.arg.integrated:
            return [calc_ldr_features(np.sum(tlist, axis=0), np.sum(nlist, axis=0), slide_coverage(np.sum(tclist, axis=0), 1), slide_coverage(np.sum(nclist, axis=0),1), self.t, self.k)]
        else:
            return [calc_ldr_features(tdata, ndata, tcdata, ncdata, self.t, self.k) for tdata, ndata, tcdata, ncdata in zip(tlist, nlist, tclist, nclist)]

    def compute_each_reactivity(self, rpkm_cutoff=1, ave_hit=2, print_all=True):
        if self.arg.score == "pars":
            assert len(self.arg.rtFiles) > 1
            for key, vdata, sdata in key_data_iterator(self.arg.rtFiles[0], self.arg.rtFiles[1], self.arg.dir):
                yield key, self.compute_pars_score(vdata, sdata)
        elif self.arg.score == "icshape":
            if self.arg.coverage is not None and len(self.arg.coverage) > 1:
                print("icshape with coverage info (background_base_density = read coverage)")
                for key, tdata, ndata, tcdata, ncdata in key_data_cov_iterator(self.arg.rtFiles[0], self.arg.rtFiles[1], self.arg.coverage[0], self.arg.coverage[1], self.arg.dir):
                    trpkm, nrpkm = utility.estimate_rpkm(tdata, sum(self.ttotal)), utility.estimate_rpkm(ndata, sum(self.ntotal))
                    if trpkm < rpkm_cutoff or nrpkm < rpkm_cutoff:
                        print("# low expression", key, trpkm, nrpkm)
                        if not print_all:
                            continue
                    yield key, self.compute_icshape_score(key, tdata, ndata, tcdata, ncdata, ave_hit, print_all)
            else:
                print("icshape without coverage info (background_base_density = RT stop coverage)")
                for key, tdata, ndata, in key_data_iterator(self.arg.rtFiles[0], self.arg.rtFiles[1], self.arg.dir):
                    trpkm, nrpkm = utility.estimate_rpkm(tdata, sum(self.ttotal)), utility.estimate_rpkm(ndata, sum(self.ntotal))
                    if trpkm < rpkm_cutoff or nrpkm < rpkm_cutoff:
                        print("# low expression", key, trpkm, nrpkm)
                        if not print_all:
                            continue
                    yield key, self.compute_icshape_score(key, tdata, ndata, tdata, ndata, ave_hit, print_all)
        elif self.arg.score == "ldr":
            assert len(self.arg.rtFiles) > 1 and len(self.arg.coverage) > 1
            for key, tdata, ndata, tcdata, ncdata in key_data_cov_iterator(self.arg.rtFiles[0], self.arg.rtFiles[1], self.arg.coverage[0], self.arg.coverage[1], self.arg.dir):
                yield key, self.compute_ldr_score(tdata, ndata, tcdata, ncdata)

    def convert_to_reactivity(self):
        print("Convert to reactivity.")
        with open(self.get_output_name(), "w") as f:
            for key, data in self.compute_each_reactivity():
                f.write(key)
                for vec in data:
                    f.write(utility.get_print_str_score(vec))
                f.write('\n')

    def compute_coverage_ratio(self, rtcount, coverage, dir="./", offset=1): #TODO 1 is correct?
        # print(rtcount, coverage)
        for key, data, cdata in key_data_iterator(rtcount, coverage, dir):
            # sys.stderr.write("Processing "+key+"\n")
            if self.arg.integrated:
                yield key, [count_log_ratio(np.sum(data, axis=0), np.sum(cdata, axis=0), offset)]
            else:
                yield key, [count_log_ratio(rt, cov, offset) for rt, cov in zip(data, cdata)]

    def convert_to_coverage_ratio(self):
        for i in range(len(self.arg.rtFiles)):
            print(self.arg.rtFiles[i], self.arg.coverage[i])
            with open(self.get_output_name(i), "w") as f:
                for key, data in self.compute_coverage_ratio(self.arg.rtFiles[i], self.arg.coverage[i], self.arg.dir):
                    f.write(key)
                    for vec in data:
                        f.write(utility.get_print_str_score(vec))
                    f.write('\n')

    def convert_to_pvalue(self):
        import pvalue
        sidx = pvalue.check_dist_model(self.arg.ndist)
        for i in range(len(self.arg.rtFiles)):
            with open(self.get_output_name(i), "w") as f:
                for key, data in utility.parse_score_tri_iterator(os.path.join(self.arg.dir, self.arg.rtFiles[i])):
                    f.write(key)
                    for tdata in data:
                        pvec = pvalue.score_to_pvalue(tdata, sidx, self.arg.reverse)
                        f.write(utility.get_print_str_score(pvec))
                    f.write('\n')

    def precompute_normalization(self):
        if self.arg.score == "pars":
            s, v = self.calc_total_read_counts()
            self.vtotal, self.stotal = calc_pars_norm_factor(v, s)
        elif self.arg.score == "icshape":
            self.ttotal, self.ntotal = self.calc_total_read_counts()
            pass
        elif self.arg.score == "ldr":
            self.ttotal, self.ntotal = self.calc_total_read_counts()
            pass

    def filter_no_expression(self, dicts):
        keys = list(dicts[0].keys())
        for key in keys:
            rm_flag = True
            for tdict in dicts:
                if max(tdict[key]) > 0.0:
                    rm_flag = False
            if rm_flag:
                for tdict in dicts:
                    del tdict[key]
        return dicts

    def merge_dataset(self):
        if self.verbose:
            print("Merge dataset.")
        case = self.arg.rtFiles[0]
        if len(self.arg.rtFiles) > 1:
            cont = self.arg.rtFiles[1]
        for name in ['case', 'cont'][0:len(self.arg.rtFiles)]:
            target = eval(name)
            if self.arg.bed:
                dicts = [utility.parse_file_bed(os.path.join(self.arg.dir, fname), float, self.arg.header) for fname in target.split(',')]
            else:
                dicts = [utility.parse_file_pars(os.path.join(self.arg.dir, fname), float, self.arg.header) for fname in target.split(',')]
            dicts = utility.get_dict_common_keys(dicts)
            # dicts = self.filter_no_expression(dicts)
            if len(dicts[0].keys()) > 0:
                utility.print_score_data(self.arg.output+"_"+name+".tab", dicts)
            else:
                print("No key remains among "+name+" files!", eval(name))

    def convert_count_to_score(self):
        if self.arg.merge or self.arg.bed:
            self.merge_dataset()
        elif len(self.arg.ndist) > 0:
            self.convert_to_pvalue()
        elif len(self.arg.score) > 0:
            if self.arg.score == "ratio":
                self.convert_to_coverage_ratio()
            elif self.arg.score in self.score_list:
                self.precompute_normalization()
                self.convert_to_reactivity()
        else:
            return

def main(argv):
    debug = False
    parser = get_parser()
    options = parser.parse_args(argv[1:])
    con = Converter(options, debug)
    con.convert_count_to_score()

if __name__ == '__main__':
    main(sys.argv)
