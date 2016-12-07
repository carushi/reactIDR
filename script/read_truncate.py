#!/usr/bin/env python

""" Please apply this script to paired-end reads when you happen to see the bowtie error shown as below.
Error, fewer reads in file specified with -2 than in file specified with -1
terminate called after throwing an instance of 'int'
(ERR): bowtie2-align died with signal 6 (ABRT) (core dumped)
python read_truncate.py (1st fastq) (2nd fastq)
"""

import subprocess
import sys
import os

def trimmed(qual, length):
	return qual[0:length]

assert len(sys.argv) > 2
cmd = "paste <(cat "+sys.argv[1]+" | paste - - - - ) <( cat "+sys.argv[2]+" | paste - - - - ) > "+sys.argv[1]+".temp"
process = subprocess.Popen(['/bin/bash', '-c', cmd])
process.communicate()

out1 = open(sys.argv[1]+"_paired.fastq", 'w')
out2 = open(sys.argv[2]+"_paired.fastq", 'w')
count = 0
with open(sys.argv[1]+".temp") as f:
	rstack, lstack = [], []
	while True:
		line = f.readline().rstrip('\n')
		if line != "":
			contents = line.split('\t')
			lstack.append(contents[0:4])
			rstack.append(contents[4:len(contents)])
		if len(rstack) == 0 or len(lstack) == 0:	break
		left = lstack.pop(0)
		right = rstack.pop(0)
		lname, rname = int(left[0].split(' ')[0].split('.')[1]), int(right[0].split(' ')[0].split('.')[1])
		if lname == rname:
			left[3] = trimmed(left[3], len(left[1]))
			right[3] = trimmed(right[3], len(right[1]))
			out1.write('\n'.join(left)+"\n")
			out2.write('\n'.join(right)+"\n")
			count += 1
		elif lname < rname:
			rstack.insert(0, right)
		else:
			lstack.insert(0, left)

os.remove(sys.argv[1]+".temp")
sys.stderr.write("Written reads: "+str(count)+"\n")