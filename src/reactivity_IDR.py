import argparse
import utility
import sys
import random
import subprocess
import os
from statsmodels.tsa.stattools import acf
import pvalue
import pylab
from idr_wrapper import *
from plot_image import *
# import idr.optimization
# from idr.utility import calc_post_membership_prbs, compute_pseudo_values

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("ifiles", nargs='+', type=str, help="read score data from FILE 1,2,3 ...(case) <1,2,3 ...(control)>", metavar="INPUT")
    parser.add_argument("-o", "--ofile", dest="ofile", help="write score data to FILE", default='temp', metavar="OUTPUT", required=False)
    parser.add_argument("-I", "--input_format", dest="iformat", help="input file format (PARS, multi-column, etcetc)", default="PARS", metavar="FORMAT", required=False)
    parser.add_argument("-O", "--output_format", dest="oformat", help="output file format (txt, bed, wig, etcetc)", default="txt", metavar="FORMAT", required=False)
    parser.add_argument("-t", "--idr", dest="idr", type=float, help="idr cutoff threshold (default: 0.01)", default = 0.01, metavar="IDR", required=False)
    parser.add_argument("-n", "--sample", dest="sample_size", type=int, help="sample size for test (default: 1000)", default=-1, metavar="SAMPLESIZE", required=False)
    parser.add_argument("-d", "--dist", dest="dist_model", help="null distribution for pvalue computation (coverage, rank, poisson, nb, zinb, nb-cisgenome)", default='Coverage', metavar="DIST", required=False)
    parser.add_argument("-r", "--reverese", dest="reverse", action="store_true", help="calculate IDR for reveresed orders (default: false)", required=False)
    parser.add_argument("--full", dest="full", action="store_true", help="store IDR as float", required=False)
    parser.add_argument("--image", dest="plot", action="store_true", help="plot score-idr scatter images", required=False)
    parser.add_argument("--job_id", dest="job_id", type=int, help="parallel jobs for score transformation ( 0 ~ ( job_all - 1 ) )", default=0, required=False)
    parser.add_argument("--job_all", dest="job_all", type=int, help="parallel jobs for score transformation", default=0, required=False)
    parser.set_defaults(full=False, plot=False, reverse=False)
    return parser


class reactIDR:
    """IDR computation and score optimization for reactivity measured by structure probing."""
    def __init__(self, options):
        self.options = options
        self.dicts = []
        self.cnames = ['case', 'control']
        self.print_options()
        self.max_print = 10

    def print_options(self):
        for idx, files in enumerate(self.options.ifiles):
            print("# "+self.cnames[idx]+" sample:\t"+files)
        print("# input format:\t"+self.options.iformat)
        if self.options.reverse:
            print("# input rank:\t"+"reversed")
        print("# output prefix:\t"+self.options.ofile)
        print("# output format:\t"+self.options.oformat)
        print("# idr threshold:\t"+str(self.options.idr))
        print("# null distribution:\t"+self.options.dist_model)
        if self.options.sample_size > 0:
            print("# sampling:\t"+str(self.options.sample_size))
        else:
            print("# sampling:\tNo")
        if self.options.job_all > 1:
            print("# parallel:\t"+str(self.options.job_id)+" / "+str(self.options.job_all))
        print("# full:\t"+str(self.options.full))

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
        plot_score_idr_scatter(s1, s2, i12, 'scatter_score_idr_'+self.output_file_head(cidx, sidx), sidx <= 0)

    def visualize_density(self, IDR, cidx, sidx):
        ofile = 'density_score_idr_'+self.output_file_head(cidx, sidx)+'.png'
        plot_density(IDR, ofile)

    def ordered_common_transcript(self):
        return sorted(list(common_transcript(self.dicts[0], self.dicts[1])))

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
            if self.options.reverse:
                after = len(s1)
                if self.options.reverse:    num = after*2
                else:   num = 0
                for i in range(0, min(before*2, before-after)):
                    s1 = np.append(s1, [num]*(after))
                    s2 = np.append(s2, [num]*(after))
                # np.save("bs1.txt", s1)
                # np.save("bs2.txt", s2)
                # print(s1)
                # print(s2)
                # r1, r2 = only_build_rank_vectors(s1, s2)
                # np.save("as1.txt", r1)
                # np.save("as2.txt", r2)
            utility.log("-> "+str(len(s1))+" and "+str(len(s2)))
        return only_build_rank_vectors(s1, s2)

    def output_file_head(self, cidx, sidx):
        if self.options.reverse: tail = '_rev'
        else:   tail = ''
        return self.options.ofile+'_'+self.cnames[cidx]+'_'+str(sidx)+tail

    def output_file_name(self, cidx, sidx):
        if self.options.reverse: tail = '_rev'
        else:   tail = ''
        score_file = self.options.ofile+'_'+self.cnames[cidx]+'_score_'+str(sidx)+'.'+self.options.oformat
        idr_file = self.options.ofile+'_'+self.cnames[cidx]+'_idr_'+str(sidx)+tail+'.'+self.options.oformat
        return score_file, idr_file

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
        if sidx == 0:
            pass
        else:
            for key in self.ordered_common_transcript():
                for idx, tdict in enumerate(self.dicts):
                    if np.max(tdict[key]) < 1:    continue
                    # if np.mean(tdict[key]) > 1000 and self.max_print > 0:
                    #     pvalue.score_to_pvalue(tdict[key], 2, True, "plot_fit_"+key+"_"+str(2))
                    #     pvalue.score_to_pvalue(tdict[key], 3, True, "plot_fit_"+key+"_"+str(3))
                    #     pvalue.score_to_pvalue(tdict[key], 4, True, "plot_fit_"+key+"_"+str(4))
                    #     pvalue.score_to_pvalue(tdict[key], 6, True, "plot_fit_"+key+"_"+str(6))
                    #     pvalue.score_to_pvalue(tdict[key], 7, True, "plot_fit_"+key+"_"+str(7))
                    #     self.max_print -= 1
                    tdict[key] = pvalue.score_to_pvalue(tdict[key], sidx, self.options.reverse)

    def print_transformed_score(self, file, cidx, sidx, job_id = 0, job_all = 1):
        utility.log("Score transform and print.")
        # if sidx == 0:   return
        with open(file, 'w') as f:
            if job_id == 0: f.write('key\tscore1\tscore2\n')
            for key in self.ordered_common_transcript()[job_id::job_all]:
                f.write(key)
                for i in [0, 1]:
                    if key in self.dicts[i]:
                        tdict = self.dicts[i][key]
                        if np.max(tdict) > 0:
                            self.dicts[i][key] = pvalue.score_to_pvalue(tdict, sidx, self.options.reverse)
                    f.write(self.get_print_str_score(key, self.dicts[i]))
                f.write('\n')

    def print_dataset_for_each_sample(self, file, index, IDR):
        count = 0
        IDR = np.ndarray.tolist(IDR)
        zero = []
        for i in range(0, int(len(IDR)/2)):
            zero.append(-np.log10(IDR.pop(0))*10.)
        zero = np.mean(zero)
        if not self.options.full:   zero = int(np.round(zero))
        with open(file, 'w') as f:
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

    def get_idr_parameter(self, seta):
        r1, r2 = self.get_concatenated_rank_scores(seta, True)
        theta, loss =  only_fit_model_and_calc_idr(r1, r2, max_iter=100)
        utility.log("End fitting")
        utility.log(theta)
        utility.log(loss)
        return theta

    def get_idr_value(self, index, theta):
        r1, r2 = self.get_concatenated_rank_scores(index, True)
        localIDRs, IDR = idr.idr.calc_IDR(np.array(theta), r1, r2)
        return localIDRs, IDR

    def fit_and_calc_IDR(self, file, cidx, index=None, sampled=None):
        if type(index) == type(None):
            index, sampled = self.sampled_scores()
        if self.options.sample_size > 0:
            utility.log("Sampling and fitting")
            theta = self.get_idr_parameter(sampled)
        else:
            utility.log("Fitting without sampling")
            theta = self.get_idr_parameter(index)
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
            if not os.path.exists(idr_file):
                self.dicts = list(utility.parse_score(score_file))
                self.fit_and_calc_IDR(idr_file, cidx)

    def write_pvalue_and_idr(self, cidx):
        sidx = pvalue.check_dist_model(self.options.dist_model)
        score_file, idr_file = self.output_file_name(cidx, sidx)
        index, sampled = self.sampled_scores()
        if not os.path.exists(score_file):
            self.print_transformed_score(score_file, cidx, sidx)
        if not os.path.exists(idr_file):
            self.dicts = list(utility.parse_score(score_file))
            self.fit_and_calc_IDR(idr_file, cidx)

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
                    self.visualize_score_idr_scatter(cidx)
            self.clear_dict()
            # if cidx == 1: # found both of case and control samples.
                # self.calculate_react_score_and_idr()

    def sample_and_test(self):
        files = [samples.split(',') for samples in self.options.ifiles]
        files = [item for sublist in files for item in sublist]
        for idx, ifile in enumerate(files):
            dict = utility.get_score_dict(ifile, self.options.iformat)
            keys = dict.keys()
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
            pylab.savefig(self.options.ofile+'_'+str(idx)+'_acc_mean_sample='+str(self.options.sample_size)+'.png')
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
