import os
import sys
import argparse
from param_fit_hmm import *
from utility import *

def get_parser():
    print("*******************************************************************************************\n"+\
          "* reactIDR: Statistical reproducibility evaluation of                                     *\n"+\
          "*           high-throughput structure analyses for robust RNA reactivity classification   *\n"+\
          "*                                                              @carushi 2017.11.07.v1.0   *\n"+\
          "*******************************************************************************************\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", dest="case", type=str, help="tab-separated input of replicated read counts or scores (loop/acc to be enriched)\n" + \
                        " (e.g., FILE1, FILE2, ...)", metavar="CASE", required=True)
    parser.add_argument("--control", dest="control", type=str, help="tab-separeted input of replicated read counts or scores (stem to be enriched)", metavar="CONTROL", required=False)
    parser.add_argument("--reverse", dest="reverse", action="store_true", help="recognize stem regions should be enriched in the case\n" + \
    "The case data in the output csv is always assumed to be enriched at accessible region even when reverse is valid.", required=False)
    parser.add_argument("--output", dest="output", type=str, help="output filename suffix (it will be concatenated with the mode name test, train, or global)", required=False, default="idr_output.csv")
    parser.add_argument("--ref", dest="ref", type=str, help="Fasta file of reference structures (dot-blacket style)\n" + \
    "(dot-blacket style e.g. x.().x)", required=False, default="")
    parser.add_argument("--DMS", dest="DMS", type=str, help="Use fasta file to exclude G and T (U) nucleotides from stem or accessible class due to no information.", required=False, default="")
    parser.add_argument("--test", dest="test", action="store_true", help="apply given parameteres (test_<output>)", required=False)
    parser.add_argument("--train", dest="train", action="store_true", help="train the parameters based on the reference information (train_<output>)", required=False)
    parser.add_argument("--time", dest="time", type=int, help="maximum iteration time during the training", required=False)
    parser.add_argument("--global", dest="noHMM", action="store_true", help="output IDR (noHMM_<output>)", required=False)
    parser.add_argument("--grid", dest="grid", action="store_true", help="grid search for initial IDR parameters (default: false)", required=False)
    parser.add_argument("--core", dest="core", type=int, help="Multi core processes for speed up.", required=False)

    parser.add_argument("--param", dest="param_file", type=str, help="parameter file name as input or output", required=False)
    parser.add_argument("--mu", dest="mu", type=float, help="mean of the reproducible group", default=1., required=False)
    parser.add_argument("--sigma", dest="sigma", type=float, help="variance of the reproducible group", default=0.2, required=False)
    parser.add_argument("--rho", dest="rho", type=float, help="correlation strength of the reproducible group among the replicates", default=0.8, required=False)
    parser.add_argument("--q", dest="p", type=float, help="ratio of reproducible group", default=0.3, required=False)
    parser.add_argument("--fix_mu", dest="fix_mu", action="store_true", help="fix mu, or no optimization.", required=False)
    parser.add_argument("--fix_sigma", dest="fix_sigma", action="store_true", help="fix sigma, or no optimization.", required=False)
    parser.add_argument("--fix_trans", dest="fix_trans", action="store_true", help="fix transition matrix, or no optimization.", required=False)
    parser.add_argument("-s", dest="start", type=int, help="ignore n bases from the very 5'-end", default=-1, required=False)
    parser.add_argument("-e", dest="end", type=int, help="ignore n bases from the very 3'-end", default=35, required=False)
    parser.add_argument("--idr", dest="idr", action="store_true", help="output 1-posterior probability during the test, train, and global mode", required=False)
    parser.add_argument("-n", "--sample", dest="sample_size", type=int, help="sample size for test (default: all)", default=-1, metavar="SAMPLESIZE", required=False)
    parser.add_argument("--independent", dest="inde", action="store_true", help="compute HMM for each transcript independently (avoid overflow).", required=False)
    parser.add_argument("--each-transcript", dest="each", action="store_true", help="compute HMM for each transcript at each time (independent read distribution).", required=False)
    parser.add_argument("--print_keys", dest="print_keys", action="store_true", help="print keys (e.g., gene name) to be computed.", required=False)
    parser.add_argument("--key_file", nargs='+', dest="key_files", help="read key files to be computed (one gene per one line).", required=False)
    parser.add_argument("--threshold", dest="threshold", type=float, help="remove transcripts whose average score is lower than a threshold.", default=float('nan'), required=False)
    parser.set_defaults(grid=False, test=False, fix_trans=False, param_file=None, control=[], ratio=False, debug=False, core=1, time=10, reverse=False, key_files=[])
    return parser

def test_each_set(para, *args):
    para.test_hmm_EMP(*args)

def read_keys(files, verbose=True):
    seta = []
    for fname in files:
        if len(fname) == 0: continue
        if not os.path.isfile(fname):
            if verbose: print("Cannot find", fname)
            continue
        with open(fname) as f:
            seta += [line.lstrip('>').rstrip('\n') for line in f.readlines() if line != '' and line[0] != '#']
    target = []
    for key in set([key.rstrip('+').rstrip('-') for key in seta]):
        target.append(key+"+")
        target.append(key+"-")
    if verbose:
        print("# Read key files", files, "->", len(target))
    return target

class IDRHmm:
    """docstring for IDRhmm"""
    def __init__(self, arg):
        self.input = [arg.case, arg.control]
        self.key_files = arg.key_files
        self.noHMM = arg.noHMM
        # self.coverage = arg.coverage
        self.sample_size = arg.sample_size
        self.params = [arg.mu, arg.sigma, arg.rho, arg.p]
        self.grid = arg.grid
        self.test = arg.test
        self.output = arg.output
        self.ref = arg.ref
        self.train = arg.train
        self.DMS = arg.DMS
        self.arg = arg


    def set_hclass(self):
        """ hclass 2: unmappable and accessible class.
            hclass 3: unmappable, stem, and accessible class.
        """
        if len(self.input[1]) == 0:
            hclass = 2
        else:
            hclass = 3
        return hclass

    def set_prefix_output(self):
        if self.train:
            self.output = "train_"+self.output
        elif self.test:
            self.output = "test_"+self.output
        elif self.arg.noHMM:
            self.output = "noHMM_"+self.output
        else:
            pass

    def get_print_keys(self):
        target = read_only_keys_of_high_expression_pars_multi(self.input, self.arg.threshold)
        print('\n'.join(target))

    def extract_each_trans(self, key, rep):
        dict = {chr(0): [0], key: rep[key]}
        if key[-1] == '+':
            minus = key[0:-1]+'-'
            if minus in rep:    dict[minus] = rep[minus]
        return dict

    def get_data(self):
        seta = None
        if len(self.key_files) > 0:
            seta = read_keys(self.key_files)
        if not self.arg.each:
            seta = [seta]
        if seta is None:
            if self.arg.threshold == self.arg.threshold:
                data = read_only_high_expression_pars_multi(self.input, self.arg.threshold, True, seta)
            else:
                data = read_data_pars_format_multi(self.input, True, seta)
            for key in data[0][0].keys():
                if key != chr(0) and key[-1] != '-':
                    yield [[self.extract_each_trans(key, rep) for rep in sample] for sample in data]
        else:
            for trans in seta:
                if self.arg.threshold == self.arg.threshold:
                    yield read_only_high_expression_pars_multi(self.input, self.arg.threshold, True, set(trans))
                else:
                    yield read_data_pars_format_multi(self.input, True, set(trans))

    def infer_reactive_sites(self):
        hclass = self.set_hclass()
        self.set_prefix_output()
        append = False
        for data in self.get_data():
            self.para = ParamFitHMM(hclass, data, self.sample_size, self.params, self.arg.debug, self.output, \
                                self.ref, self.arg.start, self.arg.end, -1, self.DMS, self.train, \
                                core=self.arg.core, reverse=self.arg.reverse, independent=self.arg.inde, idr=self.arg.idr, append=append)
            if self.train:
                self.para.train_hmm_EMP(self.grid, N=max(1, self.arg.time), param_file=self.arg.param_file, fix_mu=self.arg.fix_mu, \
                                    fix_sigma=self.arg.fix_sigma, fix_trans=self.arg.fix_trans)
            elif self.test:
                self.para.test_hmm_EMP(self.grid, N=-1, param_file=self.arg.param_file, fix_trans=self.arg.fix_trans, \
                                   fix_mu=self.arg.fix_mu, fix_sigma=self.arg.fix_sigma)
            elif self.noHMM:
                self.para.estimate_global_IDR(self.grid)
            else:
                self.para.estimate_hmm_based_IDR(self.grid, N=self.arg.time, fix_trans=self.arg.fix_trans, \
                                             fix_mu=self.arg.fix_mu, fix_sigma=self.arg.fix_sigma)
            append = True

def main(argv):
    parser = get_parser()
    options = parser.parse_args(argv[1:])
    hmm = IDRHmm(options)
    if options.print_keys:
        hmm.get_print_keys()
    else:
        hmm.infer_reactive_sites()

if __name__ == '__main__':
    main(sys.argv)
