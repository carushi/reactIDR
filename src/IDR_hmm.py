import sys
import argparse
from param_fit_hmm import *
from utility import *
import pvalue

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", dest="case", type=str, help="read replicate data from FILE1, FILE2, ...", metavar="CASE", required=True)
    parser.add_argument("--control", dest="control", type=str, help="read replicate data from FILE1, FILE2, ...", metavar="CONTROL", required=False)
    parser.add_argument("--global", dest="noHMM", action="store_true", help="Calculate global IDR, no HMM.", required=False)
    # parser.add_argument("--coverage", dest="coverage", nargs='+', type=str, help="use the ratio of read enrichment based on coverage", metavar="COVERAGE", required=False)
    parser.add_argument("-n", "--sample", dest="sample_size", type=int, help="sample size for test (default: all)", default=-1, metavar="SAMPLESIZE", required=False)
    parser.add_argument("--mu", dest="mu", type=float, help="mean of reproducible group", default=1., required=False)
    parser.add_argument("--sigma", dest="sigma", type=float, help="variance of reproducible group", default=0.2, required=False)
    parser.add_argument("--rho", dest="rho", type=float, help="dependency of reproducible group", default=0.8, required=False)
    parser.add_argument("--pi", dest="p", type=float, help="ratio of reproducible group", default=0.3, required=False)
    parser.add_argument("-s", dest="start", type=int, help="truncate start", default=-1, required=False)
    parser.add_argument("-e", dest="end", type=int, help="truncate end", default=35, required=False)
    parser.add_argument("--grid", dest="grid", action="store_true", help="grid search for initial IDR parameters (default: false)", required=False)
    parser.add_argument("--debug", dest="debug", action="store_true", help="", required=False)
    parser.add_argument("--test", dest="test", action="store_true", help="", required=False)
    parser.add_argument("--fix_mu", dest="fix_mu", action="store_true", help="Fix mu, no optimization.", required=False)
    parser.add_argument("--fix_sigma", dest="fix_sigma", action="store_true", help="Fix sigma, no optimization.", required=False)
    parser.add_argument("--fix_trans", dest="fix_trans", action="store_true", help="Fix transition matrix, no optimization.", required=False)
    parser.add_argument("--param", dest="param_file", type=str, help="Example: '[[1.0, 1.0, 0.9, 0.5], [1.0, 1.0, 0.9, 0.5]]'", required=False)
    parser.add_argument("--multi", dest="multi", action="store_true", help="For more than two replicates (but slow).", required=False)
    parser.add_argument("--core", dest="core", type=int, help="Multi core processes for speed up.", required=False)
    parser.add_argument("--independent", dest="inde", action="store_true", help="Compute HMM for each transcript independently (avoid overflow).", required=False)
    parser.add_argument("--time", dest="time", type=int, help="Iteration time for training.", required=False)
    parser.add_argument("--output", dest="output", type=str, help="output filename", required=False, default="idr_output.csv")
    parser.add_argument("--ref", dest="ref", type=str, help="Fasta file of reference structures (dot-blacket style).", required=False, default="")
    parser.add_argument("--train", dest="train", action="store_true", help="Train parameters based on the reference (dot-blacket style e.g. x.().x).", required=False)
    parser.add_argument("--reverse", dest="reverse", action="store_true", help="Stem regions should be enriched for case sample (for training).\n" + \
                                                                                "This option also enables replacement of case and cont in output csv so that case in output always shows an enrichment of accessible region.", required=False)
    parser.add_argument("--DMS", dest="DMS", type=str, help="Use fasta file to exclude G and T (U) nucleotides from stem or accessible class due to no information.", required=False, default="")
    parser.add_argument("--threshold", dest="threshold", type=float, help="Remove transcripts whose average score is lower than a threshold.", default=float('nan'), required=False)
    parser.set_defaults(grid=False, test=False, fix_trans=False, param_file=None, control=[], ratio=False, debug=False, core=1, time=10, reverse=False)
    return parser


class IDRHmm:
    """docstring for IDRhmm"""
    def __init__(self, arg):
        self.input = [arg.case, arg.control]
        self.noHMM = arg.noHMM
        # self.coverage = arg.coverage
        self.sample_size = arg.sample_size
        self.params = [arg.mu, arg.sigma, arg.rho, arg.p]
        self.grid = arg.grid
        self.test = arg.test
        self.output = arg.output
        self.ref = arg.ref
        self.multi = arg.multi
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

    def infer_reactive_sites(self):
        hclass = self.set_hclass()
        if self.arg.threshold == self.arg.threshold:
            data = read_only_high_expression_pars_multi(self.input, self.arg.threshold)
        else:
            data = read_data_pars_format_multi(self.input)
        self.set_prefix_output()
        self.para = ParamFitHMM(hclass, data, self.sample_size, self.params, self.arg.debug, self.output, \
                                self.ref, self.arg.start, self.arg.end, -1, self.DMS, self.train, \
                                core=self.arg.core, reverse=self.arg.reverse, independent=self.arg.inde)
        # if self.arg.debug:
        #     self.para.estimate_hmm_based_IDR_debug(self.grid, N=10)
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



if __name__ == '__main__':
    parser = get_parser()
    options = parser.parse_args()
    hmm = IDRHmm(options)
    hmm.infer_reactive_sites()
