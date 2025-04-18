import sys
from .IDR_hmm import *

def run_reactIDR(argv=None):
    if argv is None:    
        argv = sys.argv
    parser = get_parser()
    options = parser.parse_args(argv[1:])
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
    run_reactIDR(sys.argv)
