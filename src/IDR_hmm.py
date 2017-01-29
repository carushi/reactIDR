import utility
import sys
import argparse
from param_fit_hmm import *
from utility import *

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", dest="case", nargs='+', type=str, help="read replicate data from FILE1, FILE2, ...", metavar="CASE", required=True)
    parser.add_argument("--control", dest="control", nargs='+', type=str, help="read replicate data from FILE1, FILE2, ...", metavar="CONTROL", required=False)
    parser.add_argument("-n", "--sample", dest="sample_size", type=int, help="sample size for test (default: 1000)", default=-1, metavar="SAMPLESIZE", required=False)
    parser.add_argument("--mu", dest="mu", type=float, help="mean of reproducible group", default=1., required=False)
    parser.add_argument("--sigma", dest="sigma", type=float, help="variance of reproducible group", default=0.2, required=False)
    parser.add_argument("--rho", dest="rho", type=float, help="dependency of reproducible group", default=0.8, required=False)
    parser.add_argument("--pi", dest="p", type=float, help="ratio of reproducible group", default=0.3, required=False)
    parser.add_argument("--grid", dest="grid", action="store_true", help="grid search for initial IDR parameters (default: false)", required=False)
    parser.add_argument("--test", dest="test", action="store_true", help="", required=False)
    parser.set_defaults(grid=False, test=False, control=[])
    return parser


class IDRHmm:
    """docstring for IDRhmm"""
    def __init__(self, arg):
        self.input = [arg.case, arg.control]
        self.sample_size = arg.sample_size
        self.params = [arg.mu, arg.sigma, arg.rho, arg.p]
        self.grid = arg.grid
        self.test = arg.test

    def read_data_pars_format(self):
        data = [[{chr(0):[0]}, {chr(0):[0]}]]
        for key, one, two in parse_score_iterator(self.input[0][0]):
            data[0][0][key] = one
            data[0][1][key] = two
        if len(self.input[1]) > 1:
            data.append([{chr(0):[0]}, {chr(0):[0]}])
            for key, one, two in parse_score_iterator(self.input[1][0]):
                data[1][0][key] = one
                data[1][1][key] = two
        return data

    def infer_reactive_sites(self):
        hclass = 2
        if self.input[1] != []:   hclass = 3
        data = self.read_data_pars_format()
        self.para = ParamFitHMM(hclass, data, self.sample_size, self.params)
        if self.test:
            self.para.hmm_EMP_with_pseudo_value_algorithm_test(self.grid)
            return
        self.para.hmm_EMP_with_pseudo_value_algorithm(self.grid)


if __name__ == '__main__':
    parser = get_parser()
    # try:
    options = parser.parse_args()
    hmm = IDRHmm(options)
    hmm.infer_reactive_sites()
    # except:
    #     options = parser.parse_args()
    #     print(options)
    #     parser.print_help()
    #     sys.exit(0)
