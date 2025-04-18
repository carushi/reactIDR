import argparse
import utility
import sys
import random
import subprocess
import os
import math
from statsmodels.tsa.stattools import acf
import pvalue
import pylab
from reactIDR.idr_wrapper import *
from reactIDR.plot_image import *
# import idr.optimization
# from idr.utility import calc_post_membership_prbs, compute_pseudo_values

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("ifiles", nargs='+', type=str, help="read score data from FILE 1,2,3 ...(case) <1,2,3 ...(control)>", metavar="INPUT")
    parser.add_argument("--prefix", dest="prefix", help="file prefix for score data", default='temp', metavar="OUTPUT", required=False)
    parser.add_argument("-I", "--input_format", dest="iformat", help="input file format (PARS, multi-column, etcetc)", default="PARS", metavar="FORMAT", required=False)
    parser.add_argument("-O", "--output_format", dest="oformat", help="output file format (txt, bed, wig, etcetc)", default="txt", metavar="FORMAT", required=False)
    parser.add_argument("-t", "--idr", dest="idr", type=float, help="idr cutoff threshold (default: 0.01)", default = 0.01, metavar="IDR", required=False)
    parser.add_argument("-n", "--sample", dest="sample_size", type=int, help="sample size for test (default: 1000)", default=-1, metavar="SAMPLESIZE", required=False)
    parser.add_argument("-d", "--dist", dest="dist_model", help="null distribution for pvalue computation (coverage, rank, poisson, nb, zinb, nb-cisgenome, and ec (estimated coverage for downstream normalization))", default='Coverage', metavar="DIST", required=False)
    parser.add_argument("-r", "--reverese", dest="reverse", action="store_true", help="calculate IDR for reveresed orders (default: false)", required=False)
    parser.add_argument("-p", "--percent", dest="percent", type=int, help="set a threshold for the minimum ratio of regions with more than 0 read (0 - 100\%)", default=0, required=False)
    parser.add_argument("-o", "--dir", dest="dir", help="output directory", default="./output/", required=False)
    parser.add_argument("--mu", dest="mu", help="mean of reproducible group", default="1", required=False)
    parser.add_argument("--sigma", dest="sigma", help="variance of reproducible group", default="0.2", required=False)
    parser.add_argument("--full", dest="full", action="store_true", help="store IDR as float", required=False)
    parser.add_argument("--grid", dest="grid", action="store_true", help="Force grid search for parameter estimation", required=False)
    parser.add_argument("--image", dest="plot", action="store_true", help="plot score-idr scatter images", required=False)
    parser.add_argument("--omit", dest="omit", action="store_true", help="print scores only for transcripts with the maxmimum score >0", required=False)
    parser.add_argument("--job_id", dest="job_id", type=int, help="parallel jobs for score transformation ( 0 ~ ( job_all - 1 ) )", default=0, required=False)
    parser.add_argument("--job_all", dest="job_all", type=int, help="parallel jobs for score transformation", default=0, required=False)
    parser.set_defaults(full=False, plot=False, reverse=False, omit=False)
    return parser


class reactIDR:
    """IDR computation and score optimization for reactivity measured by structure probing."""
    def __init__(self, options):
        self.options = options
        self.dicts = []
        self.cnames = ['case', 'control']
        self.print_options()
        self.max_print = 10
        self.zero_add = 0

    def print_options(self):
        for idx, files in enumerate(self.options.ifiles):
            print("# "+self.cnames[idx]+" sample:\t"+files)
        print("# input format:\t"+self.options.iformat)
        if self.options.reverse:
            print("# input rank:\t"+"reversed")
        print("# output prefix:\t"+self.options.prefix)
        print("# output format:\t"+self.options.oformat)
        print("# output directory:\t"+self.options.dir)
        print("# idr threshold:\t"+str(self.options.idr))
        print("# null distribution:\t"+self.options.dist_model)
        if self.options.sample_size > 0:
            print("# sampling:\t"+str(self.options.sample_size))
        else:
            print("# sampling:\tNo")
        if self.options.job_all > 1:
            print("# parallel:\t"+str(self.options.job_id)+" / "+str(self.options.job_all))
        print("# full:\t"+str(self.options.full))
        print("# force grid:\t"+str(self.options.full))


    def add_dict(self, file):
        self.dicts.append(utility.get_score_dict(file, self.options.iformat))

    def clear_dict(self):
        self.dicts = []

    def no_idr(self):
        IDR = {}
        for key in self.ordered_common_transcript():
            IDR[key] = [0]*len(self.dicts[0][key])
        return IDR

    def get_score_set(self, cidx, dir = ''):
        sidx = pvalue.check_dist_model(self.options.dist_model)
        score_file, idr_file = self.output_file_name(cidx, sidx)
        dicts = list(utility.parse_score(dir+score_file))
        if os.path.exists(dir+idr_file):
            IDR = utility.parse_idr(dir+idr_file, self.score_type())
        else:
            IDR = self.no_idr()
        return sidx, IDR, dicts

    def score_type(self):
        if self.options.full:
            return float
        else:
            return int

    def visualize_score_idr_scatter(self, cidx):
        sidx, IDR, self.dicts = self.get_score_set(cidx)
        s1, s2 = get_concatenated_scores(IDR.keys(), self.dicts[0], self.dicts[1])
        i12 = get_concatenated_score(IDR.keys(), IDR)
        self.visualize_density(i12, cidx, sidx)
        s1, s2, i12 = remove_no_data(s1, s2, i12)
        plot_score_idr_scatter(s1, s2, i12, self.output_file_head('scatter_score_idr_', cidx, sidx), sidx <= 0)

    def extract_idr_scatter(self, cidx, threshold = 1.):
        sidx, IDR, self.dicts = self.get_score_set(cidx)
        for key in IDR.keys():
            if max(self.dicts[0][key]) < 100: continue
            print("# ", key, max(self.dicts[0][key]), max(self.dicts[1][key]))
            for i in range(len(self.dicts[0][key])):
                if min(self.dicts[0][key][i], self.dicts[1][key][i]) < 100: continue
                ratio = float(self.dicts[0][key][i]+1)/float(self.dicts[1][key][i]+1)
                if abs(math.log10(ratio)) >= threshold:
                    print("# key", key)
                    print(self.dicts[0][key])
                    print(self.dicts[1][key])
                    break

    def visualize_density(self, IDR, cidx, sidx):
        ofile = self.output_file_head('density_score_idr_', cidx, sidx)+'.png'
        plot_density(IDR, ofile)

    def ordered_common_transcript(self):
        return sorted(list(utility.common_transcript(self.dicts[0], self.dicts[1])))

    def sampled_scores(self, sample=True):
        smax = max(0, self.options.sample_size)
        index = self.ordered_common_transcript()
        if not sample: return index, []
        sampled = random.sample(index, min(len(index), smax))
        return index, sampled

    def get_concatenated_rank_scores(self, seta, remove=False):
        s1, s2 = get_concatenated_scores(seta, self.dicts[0], self.dicts[1])
        if self.options.reverse:
            s1 = list(map(lambda x: -x, s1))
            s2 = list(map(lambda x: -x, s2))
        if remove:
            utility.log("Remove missing data: "+str(len(s1))+" and "+str(len(s2)))
            thres = 0
            before = len(s1)
            l = [(s1[i], s2[i]) for i in range(len(s1)) if abs(s1[i]) > thres and abs(s2[i]) > thres]
            if len(l) == 0: s1, s2 = [], []
            else:   s1, s2 = [list(t) for t in zip(*l)]
            after = len(s1)
            if self.options.reverse:    num = after*2
            else:   num = 0
            self.zero_add = min(1000, min(before*2, before-after))
            s1 = np.append([num]*(self.zero_add), s1)
            s2 = np.append([num]*(self.zero_add), s2)
            utility.log("-> "+str(len(s1))+" and "+str(len(s2)))
        return only_build_rank_vectors(s1, s2)

    def check_dir(self):
        if not os.path.exists(self.options.dir):
            os.makedirs(self.options.dir)
        if self.options.dir[-1] != "/":
            self.options.dir = self.options.dir+"/"

    def output_file_head(self, head, cidx, sidx):
        self.check_dir()
        if self.options.reverse: tail = '_rev'
        else:   tail = ''
        return self.options.dir+head+self.options.prefix+'_'+self.cnames[cidx]+'_'+str(sidx)+tail

    def output_file_name(self, cidx, sidx):
        self.check_dir()
        if self.options.reverse: tail = '_rev'
        else:   tail = ''
        score_file = self.options.prefix+'_'+self.cnames[cidx]+'_score_'+str(sidx)+'.'+self.options.oformat
        idr_file = self.options.prefix+'_'+self.cnames[cidx]+'_idr_'+str(sidx)+tail+'.'+self.options.oformat
        return self.options.dir+score_file, self.options.dir+idr_file

    def get_print_str_score(self, key, tdict):
        seq = '\t'
        if key in tdict: seq += ';'.join(list(map(str, tdict[key])))
        return seq

    def print_score_data(self, file, cidx, sidx):
        utility.log("Print score.")
        with open(file, 'w') as f:
            f.write('key\tscore1\tscore2\n')
            for key in (set(self.dicts[0].keys()) | set(self.dicts[1].keys())):
                f.write(key)
                for tdict in self.dicts:
                    f.write(self.get_print_str_score(key, tdict))
                f.write('\n')

    def score_transform(self, sidx):
        utility.log("Score transform.")
        if sidx <= 0:
            pass
        else:
            for key in self.ordered_common_transcript():
                for idx, tdict in enumerate(self.dicts):
                    if self.options.percent > 0:
                        if len([x for x in tdict[key] if x > 0])*100 < len(tdict[key])*self.options.percent:
                            flag = True
                            break
                    elif np.max(tdict[key]) < 1:
                        continue
                    tdict[key] = pvalue.score_to_pvalue(tdict[key], sidx, self.options.reverse)

    def delete_keys(self, keys):
        utility.log("Applied for "+str(len(keys))+" samples.")
        for i in range(len(self.dicts)):
            self.dicts[i] = {k: self.dicts[i][k] for k in keys}

    def print_transformed_score(self, file, cidx, sidx, job_id = 0, job_all = 1):
        utility.log("Score transform and print.")
        key_list = self.ordered_common_transcript()[job_id::job_all]
        if job_all > 1:
            self.delete_keys(key_list)
        with open(file, 'w') as f:
            if job_id == 0: f.write('key\tscore1\tscore2\n')
            for key in key_list:
                f.write(key)
                for i in [0, 1]:
                    if key in self.dicts[i]:
                        tdata = self.dicts[i][key]
                        if not self.options.omit or np.max(tdata) > 0:
                            self.dicts[i][key] = pvalue.score_to_pvalue(tdata, sidx, self.options.reverse)
                    f.write(self.get_print_str_score(key, self.dicts[i]))
                f.write('\n')

    def print_dataset_for_each_sample(self, file, index, IDR):
        count = 0
        IDR = np.ndarray.tolist(IDR)
        zero = []
        for i in range(0, self.zero_add):
            zero.append(-np.log10(IDR.pop(0))*10.)
        zero = np.mean(zero)
        if not self.options.full:   zero = int(np.round(zero))
        with open(file, 'w') as f:
            utility.log("# Write IDR scores to "+file+"\n")
            f.write("name\tpos1\tpos2\tIDR\n")
            count = 0
            for i in index:
                dind1, dind2 = list(self.dicts[0].keys()).index(i), list(self.dicts[1].keys()).index(i)
                start, end = count, count+len(self.dicts[0][i])
                f.write(i+"\t"+str(dind1)+"\t"+str(dind2)+"\t")
                tidr = []
                for x1, x2 in zip(self.dicts[0][i], self.dicts[1][i]):
                    if abs(x1) > 0 and abs(x2) > 0:
                        if not self.options.full:
                            tidr.append(int(min(1000, np.round(-np.log10(IDR.pop(0))*10.))))
                            # tidr.append(int(-np.log10(IDR.pop(0))*1000))
                        else:
                            tidr.append(-np.log10(IDR.pop(0)))
                    else:
                        tidr.append(zero)
                f.write(";".join([ str(x) for x in tidr ])+"\n")

    def get_idr_parameter(self, seta, cidx, sidx):
        r1, r2 = self.get_concatenated_rank_scores(seta, True)
        theta, loss =  only_fit_model_and_calc_idr(r1, r2, max_iter=100, mu=float(self.options.mu), sigma=float(self.options.sigma),
                        image=self.options.plot, header=self.output_file_head("param_em_", cidx, sidx), grid=self.options.grid)
        utility.log("End fitting")
        utility.log(theta)
        utility.log(loss)
        return theta

    def get_idr_value(self, index, theta):
        r1, r2 = self.get_concatenated_rank_scores(index, True)
        return get_idr_value(r1, r2, theta)
        # localIDRs, IDR = idr.idr.calc_IDR(np.array(theta), r1, r2)
        # return localIDRs, IDR

    def fit_and_calc_IDR(self, file, cidx, sidx, index=None, sampled=None):
        if type(index) == type(None):
            index, sampled = self.sampled_scores()
        if self.options.sample_size > 0:
            utility.log("Sampling and fitting")
            theta = self.get_idr_parameter(sampled, cidx, sidx)
        else:
            utility.log("Fitting without sampling")
            theta = self.get_idr_parameter(index, cidx, sidx)
        localIDRs, IDR = self.get_idr_value(index, theta)
        self.print_dataset_for_each_sample(file, index, IDR)

    def score_transform_parallel(self, cidx):
        sidx = pvalue.check_dist_model(self.options.dist_model)
        score_file, idr_file = self.output_file_name(cidx, sidx)
        if self.options.job_id < self.options.job_all:
            if not os.path.exists(score_file+"_"+str(self.options.job_id)):
                self.print_transformed_score(score_file+"_"+str(self.options.job_id), cidx, sidx, self.options.job_id, self.options.job_all)
        elif self.options.job_id == self.options.job_all:
            if not os.path.exists(score_file):
                for i in range(0, self.options.job_all):
                    if i == 0:  cmd = "cat "+score_file+"_"+str(i)+">"+score_file
                    else:   cmd = "cat "+score_file+"_"+str(i)+">>"+score_file
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    process.communicate()
            if not os.path.exists(idr_file) and sidx >= 0:
                self.dicts = list(utility.parse_score(score_file))
                self.fit_and_calc_IDR(idr_file, cidx, sidx)

    def write_pvalue_and_idr(self, cidx):
        sidx = pvalue.check_dist_model(self.options.dist_model)
        score_file, idr_file = self.output_file_name(cidx, sidx)
        index, sampled = self.sampled_scores()
        if not os.path.exists(score_file):
            self.print_transformed_score(score_file, cidx, sidx)
        if not os.path.exists(idr_file) and sidx >= 0:
            self.dicts = list(utility.parse_score(score_file))
            self.fit_and_calc_IDR(idr_file, cidx, sidx)

    def calculate_pvalue_and_idr(self):
        for cidx, files in enumerate(self.options.ifiles):
            dicts = []
            if self.options.job_id >= 0 and self.options.job_all >= 1: # parallel
                if self.options.job_id == self.options.job_all:
                    self.score_transform_parallel(cidx)
                    if self.options.plot:
                        self.visualize_score_idr_scatter(cidx)
                else:
                    for idx, file in enumerate(files.split(',')):
                        self.add_dict(file)
                    self.score_transform_parallel(cidx)
            else:
                for idx, file in enumerate(files.split(',')):
                    self.add_dict(file)
                self.write_pvalue_and_idr(cidx)
                if self.options.plot:
                    # self.extract_idr_scatter(cidx)
                    self.visualize_score_idr_scatter(cidx)
            self.clear_dict()
            # if cidx == 1: # found both of case and control samples.
                # self.calculate_react_score_and_idr()

    def sample_and_test(self):
        files = [samples.split(',') for samples in self.options.ifiles]
        files = [item for sublist in files for item in sublist]
        for idx, ifile in enumerate(files):
            dict = utility.get_score_dict(ifile, self.options.iformat)
            # keys = dict.keys()
            acc = []
            for i in random.sample(dict.keys(), min(len(list(dict.keys())), int(self.options.sample_size))):
                if sum(dict[i]) > 0:
                    acc.append(acf(dict[i], fft=True))
                    pylab.plot(acc[-1], 'b')
            m = []
            for i in range(min(len(acc[0]), 400)):
                t = np.mean(list(map(lambda x: x[i], acc)))
                m.append(t)
            pylab.plot(m, 'r')
            pylab.savefig(self.options.prefix+'_'+str(idx)+'_acc_mean_sample='+str(self.options.sample_size)+'.png')
            pylab.clf()



if __name__ == '__main__':
    parser = get_parser()
    try:
        options = parser.parse_args()
        react = reactIDR(options)
        #react.sample_and_test()
        react.calculate_pvalue_and_idr()
    except:
        parser.print_help()
        sys.exit(0)
