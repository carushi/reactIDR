import os
import sys
import argparse
import random
from reactIDR.param_fit_hmm import *
from reactIDR.utility import *
import traceback


def get_parser():
    print("*******************************************************************************************\n"+\
          "* reactIDR: Statistical reproducibility evaluation of                                     *\n"+\
          "*           high-throughput structure analyses for robust RNA reactivity classification   *\n"+\
          "*                                                              @carushi 2025.04.08.v2.0   *\n"+\
          "*******************************************************************************************\n")

    parser = argparse.ArgumentParser()
    parser.add_argument('argv_file', type=str, nargs='*')
    parser.add_argument("--case", dest="case", type=str, help="tab-separated input of replicated read counts or scores (loop/acc to be enriched)\n" + \
                        " (e.g., FILE1, FILE2, ...)", metavar="CASE", required=False)
    parser.add_argument("--control", dest="control", type=str, help="tab-separeted input of replicated read counts or scores (stem to be enriched)", metavar="CONTROL", required=False)
    parser.add_argument("--reverse", dest="reverse", action="store_true", help="recognize stem regions should be enriched in the case\n" + \
    "The case data in the output csv is always assumed to be enriched at accessible region even when reverse is valid.", required=False)
    parser.add_argument("--output", dest="output", type=str, help="output filename suffix (it will be concatenated with the mode name test, train, or global)", required=False, default="idr_output.csv")
    parser.add_argument("--ref", dest="ref", type=str, help="Fasta file of reference structures (dot-blacket style)\n" + \
    "(dot-blacket style e.g. x.().x)", required=False, default="")
    parser.add_argument("--DMS", dest="DMS", type=str, help="Use fasta file to exclude G and T (U) nucleotides from stem or accessible class due to no information.", required=False, default="")
    parser.add_argument("--test", dest="test", action="store_true", help="apply given parameteres (test_<output>)", required=False)
    parser.add_argument("--fit", dest="fit", action="store_true", help="apply given parameteres and fit (fit_<output>)", required=False)
    parser.add_argument("--train", dest="train", action="store_true", help="train the parameters based on the reference information (train_<output>)", required=False)
    parser.add_argument("--time", dest="time", type=int, help="maximum iteration time during the training", required=False)
    parser.add_argument("--global", dest="noHMM", action="store_true", help="output IDR (noHMM_<output>)", required=False)
    parser.add_argument("--grid", dest="grid", action="store_true", help="grid search for initial IDR parameters (default: false)", required=False)
    parser.add_argument("--core", dest="core", type=int, help="Multi core processes for speed up.", required=False)

    parser.add_argument("--param", dest="iparam", type=str, help="parameter file name as input or output", required=False)
    parser.add_argument("--output_param", dest="oparam", type=str, help="parameter file name as output", required=False)
    parser.add_argument("--mu", dest="mu", type=float, help="mean of the reproducible group", default=1., required=False)
    parser.add_argument("--sigma", dest="sigma", type=float, help="variance of the reproducible group", default=0.2, required=False)
    parser.add_argument("--rho", dest="rho", type=float, help="correlation strength of the reproducible group among the replicates", default=0.8, required=False)
    parser.add_argument("--q", dest="p", type=float, help="ratio of reproducible group", default=0.3, required=False)
    parser.add_argument("--csv", dest="csv", action="store_true", help="input is treated as csv files; only valid for when noHMM mode is turned on.", required=False)
    parser.add_argument("--fix_mu", dest="fix_mu", action="store_true", help="fix mu, or no optimization.", required=False)
    parser.add_argument("--fix_sigma", dest="fix_sigma", action="store_true", help="fix sigma, or no optimization.", required=False)
    parser.add_argument("--fix_trans", dest="fix_trans", action="store_true", help="fix transition matrix, or no optimization.", required=False)
    parser.add_argument("--random", dest="random", action="store_true", help="set initial parameters randomly", required=False)
    parser.add_argument("-s", dest="start", type=int, help="ignore n bases from the very 5'-end", default=-1, required=False)
    parser.add_argument("-e", dest="end", type=int, help="ignore n bases from the very 3'-end", default=35, required=False)
    parser.add_argument("--idr", dest="idr", action="store_true", help="output 1-posterior probability during the test, train, and global mode", required=False)
    parser.add_argument("-n", "--sample", dest="sample_size", type=int, help="sample size for test (default: all)", default=-1, metavar="SAMPLESIZE", required=False)
    parser.add_argument("--independent", dest="inde", action="store_true", help="compute HMM for each transcript independently (avoid overflow).", required=False)
    parser.add_argument("--each-transcript", dest="each", action="store_true", help="compute HMM for each transcript at each time (independent read distribution).", required=False)
    parser.add_argument("--print_keys", dest="print_keys", action="store_true", help="print keys (e.g., gene name) to be computed.", required=False)
    parser.add_argument("--key_file", nargs='+', dest="key_files", help="read key files to be computed (one gene per one line).", required=False)
    parser.add_argument("--threshold", dest="threshold", type=float, help="remove transcripts whose average score is lower than a threshold.", default=float('nan'), required=False)
    parser.set_defaults(grid=False, test=False, fix_trans=False, iparam=None, oparam=None, case=[], control=[], ratio=False, debug=False, core=1, time=10, reverse=False, key_files=[])
    return parser

def set_each_option(key, value, options):
    fargv_dict = {'mu':'mu', 'sigma':'sigma', 'rho':'rho', 'q':'p'}
    arg_dict = {'independent':'inde', 'each-transcript':'each', 'grid':'grid'}
    argv_dict = {'param':'iparam', 'output_param':'oparam'}
    if key == 'mode':
        exec("options."+value+" = True")
    elif key in ['core', 'start', 'end', 'time']:
        exec("options."+key+" = int(value)")
    elif key in fargv_dict:
        exec("options."+fargv_dict[key]+" = float(value)")
    elif key in argv_dict:
        exec("options."+argv_dict[key]+" = \""+value+"\"")
    elif key == 'set':
        if value in arg_dict:
            exec("options."+arg_dict[value]+" = True")
        else:
            exec("options."+value+" = True")
    else:
        exec("options."+key+" = \""+value+"\"")
    return options

def read_options(options):
    for fname in options.argv_file:
        with open(fname) as f:
            for line in f.readlines():
                contents = line.rstrip('\n').split('\t')
                if len(contents) >= 2:
                    options = set_each_option(contents[0], contents[1], options)
    return options

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

def set_random(options):
    options.mu = random.choice(np.linspace(0.1, 1.0, 10))
    options.sigma = random.choice(np.linspace(0.1, 1.0, 10))
    options.rho = random.choice(np.linspace(0.0, 0.9, 10))
    options.p = random.choice(0.1, 0.9, 10)
    return options


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
        self.fit = arg.fit
        self.output = arg.output
        self.ref = arg.ref
        self.train = arg.train
        self.DMS = arg.DMS
        self.arg = arg
        self.csv_info = []



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
        elif self.fit:
            self.output = "fit_"+self.output
        elif self.noHMM:
            self.output = "noHMM_"+self.output
        else:
            pass

    def get_print_keys(self):
        target = read_only_keys_of_high_expression_pars_multi(self.input, self.arg.threshold)
        print('\n'.join(target))

    def extract_each_trans(self, key, rep):
        dict = {chr(0): [0], key: rep[key]}
        if key[-1] == '+' and False:
            minus = key[0:-1]+'-'
            if minus in rep:    dict[minus] = rep[minus]
        return dict

    def get_data_csv(self, seta):
        data, gene_list, columns = read_csv_multi(self.input, self.arg.threshold, True, seta, delim=',')
        self.csv_info = [list(gene_list), columns]
        print('# CSV data --')
        print('# row(genes) :', self.csv_info[0])
        print('# column(samples):', self.csv_info[1])
        for key in data[0][0].keys():
            print(key)
            if key != chr(0) and key[-1] != '-':
                yield [[self.extract_each_trans(key, rep) for rep in sample] for sample in data]
        
    def get_data(self):
        seta = None
        if len(self.key_files) > 0:
            seta = read_keys(self.key_files)
        if not self.arg.each:
            seta = [seta]
        if self.arg.csv:
            for data in self.get_data_csv(seta):
                yield data
        else:
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
                        yield read_only_high_expression_pars_multi(self.input, self.arg.threshold, True, trans)
                    else:
                        yield read_data_pars_format_multi(self.input, True, trans)

    def infer_reactive_sites(self):
        hclass = self.set_hclass()
        self.set_prefix_output()
        append = False
        for data in self.get_data():
            print(data)
            if len(data) == 0 or len(data[0]) == 0:
                continue
            try:
                self.para = ParamFitHMM(hclass, data, self.sample_size, self.params, self.arg.debug, self.output, \
                                    self.ref, self.arg.start, self.arg.end, -1, self.DMS, self.train, \
                                    core=self.arg.core, reverse=self.arg.reverse, independent=self.arg.inde, idr=self.arg.idr, append=append, \
                                    iparam=self.arg.iparam, oparam=self.arg.oparam)
                if self.train:
                    self.para.train_hmm_EMP(self.grid, N=max(1, self.arg.time), fix_mu=self.arg.fix_mu, \
                                        fix_sigma=self.arg.fix_sigma, fix_trans=self.arg.fix_trans)
                elif self.test:
                    self.para.test_hmm_EMP(self.grid, N=-1, fix_trans=self.arg.fix_trans, \
                                       fix_mu=self.arg.fix_mu, fix_sigma=self.arg.fix_sigma)
                elif self.fit:
                    self.para.fit_hmm_EMP(self.grid, N=max(1, self.arg.time), fix_trans=self.arg.fix_trans, \
                                       fix_mu=self.arg.fix_mu, fix_sigma=self.arg.fix_sigma)
                elif self.noHMM:
                    self.para.estimate_global_IDR(self.grid)
                    if self.arg.csv:
                        convert_output_to_csv(self.output, self.csv_info[0], self.csv_info[1])
                else:
                    self.para.estimate_hmm_based_IDR(self.grid, N=self.arg.time, fix_trans=self.arg.fix_trans, \
                                                 fix_mu=self.arg.fix_mu, fix_sigma=self.arg.fix_sigma)
                append = True
            except:
                print('Unexpected error', sys.exc_info())
                print(traceback.print_exc())

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        parser = get_parser()
        options = parser.parse_args(argv[1:])
    except:
        print('Parse error', sys.exc_info())
        return
    print(options)
    if options.argv_file is not None:
        option = read_options(options)
    if options.random:
        options = set_random(options)
    if len(options.case) == 0:
        print("No data")
        return
    hmm = IDRHmm(options)
    if options.print_keys:
        hmm.get_print_keys()
    else:
        hmm.infer_reactive_sites()

if __name__ == '__main__':
    main(sys.argv)
