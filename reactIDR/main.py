#!/usr/bin/env python

import sys
import IDR_hmm
import score_converter
import evaluate_IDR_csv

import sys
def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == 'convert':
            score_converter.main(['score_converter.py']+sys.argv[2:])
            return
        elif sys.argv[1] == 'visualize':
            evaluate_IDR_csv.main(['evaluate_IDR_csv.py']+sys.argv[2:])
            return
        else:
            IDR_hmm.main(['IDR_hmm.py']+sys.argv[2:])
            return
    IDR_hmm.main(['IDR_hmm.py'])


if __name__ == '__main__':
    main()
