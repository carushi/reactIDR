#!/bin/python

import argparse
import sys
import os
import subprocess
import bz2

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("fastqFiles", nargs='+', type=str, help="fastq files (or bz2 compressed fastq files)", metavar="INPUT")
    parser.add_argument("-b", "--barcode", dest="barcode", help="Set barcode sequence (ex NNNNXXXXNNNNN)", metavar="BARCODE", default="NNNNXXXXNNNNN", required=False)
    parser.add_argument("-m", "--mismatch", dest="mismatch", help="The number of mismatch to allow for collapsing (only valid without shell option)", metavar="MISMATCH", default=0, type=int, required=False)
    parser.add_argument("-s", "--split", dest="split", action="store_true", help="Split fastq files depending on the barcode", required=False)
    parser.add_argument("-r", "--remove", dest="remove", action="store_true", help="Remove barcode sequences", required=False)
    parser.add_argument("--job_all", dest="job_all", type=int, help="Number of parallel jobs", metavar="JOB_ALL", default=1, required=False)
    parser.add_argument("--job_id", dest="job_id", type=int, help="Job id", metavar="JOB_ID", default=0, required=False)
    parser.add_argument("--build", dest="build", action="store_true", help="Build sorted read files and barcode files before parallel job", required=False)
    parser.add_argument("--clean", dest="clean", action="store_true", help="Remove tempfiles (use after writing!)", required=False)
    parser.add_argument("--stdout", dest="stdout", action="store_true", help="Print to stdout", required=False)
    parser.add_argument("--shell", dest="shell", action="store_true", help="Use shell sort command for large files.", required=False)
    parser.set_defaults(split=False, remove=False, build=False)
    return parser


def read_collapse():
    count = 0
    temp = []
    dict = {}
    for line in sys.stdin:
        if line == "":
            if count%4 != 0:
                sys.stderr.write("Warning: probably truncated")
            return
        count += 1
        if count%4 == 0:
            dict = add_seq(temp[3], dict)
            temp = []
        temp.append(line)

class readCollapse():
    """docstring for readCollapse"""
    def __init__(self, arg):
        self.files = arg.fastqFiles
        self.barcode = arg.barcode
        self.len_barcode = len(self.barcode)
        self.mismatch = arg.mismatch
        self.split = arg.split
        self.remove = arg.remove
        self.stdout = arg.stdout
        self.job_id = arg.job_id
        self.job_all = arg.job_all
        self.build = arg.build
        self.clean = arg.clean
        if self.job_id == self.job_all: self.clean = True

    def add_read(self, mark, seq, score_list, ind):
        score = sum([ord(i) for i in score_list])
        if seq in mark:
            if mark[seq][1] < score:    mark[seq] = (ind, score)
        else:
            mark[seq] = (ind, score)
        return mark, self.extract_barcode(seq)

    def extract_barcode(self, seq):
        return "".join([seq[i] for i in range(min(len(seq), self.len_barcode)) if self.barcode[i] == "X"])

    def get_marked_reads(self, file):
        mark = {}
        barcode = {}
        output = {}
        with open(file) as f:
            temp, count = [], 0
            while True:
                line = f.readline()
                if line == "":
                    break
                temp.append(line.rstrip('\n'))
                if len(temp) == 4:
                    mark, tbarcode = self.add_read(mark, temp[1], temp[3], count)
                    barcode[tbarcode] = 1
                    temp, count = [], count+1
        return mark, list(set(barcode))

    def write_subjected_reads(self, file, ofile, mark, tbarcode):
        output = open(ofile, 'w') if not self.stdout else sys.stdout
        with open(file) as input:
            temp, count = [], 0
            while True:
                line = input.readline()
                if line == "":
                    break
                temp.append(line.rstrip('\n'))
                if len(temp) == 4:
                    if temp[1] in mark and mark[temp[1]][0] == count and self.extract_barcode(temp[1]) == tbarcode:
                        if self.remove:
                            temp[1] = temp[1][len(self.barcode):]
                            temp[3] = temp[3][len(self.barcode):]
                        for elem in temp:
                            output.write(elem+"\n")
                    temp, count = [], count+1
        if not self.stdout:
            out.close()

    def write_splitted_reads(self, file, mark, barcode):
        basename = self.get_base_name(file)
        if self.split:
            for tbarcode in barcode:
                if len(tbarcode) == len([i for i in self.barcode if i == "X"]):
                    ofile = basename+"_"+tbarcode+".fastq"
                    sys.stderr.write("Write to "+ofile+"...")
                    self.write_subjected_reads(file, ofile, mark, tbarcode)
        else:
            ofile = basename+"_nosplit.fastq"
            self.write_subjected_reads(file, ofile, mark, "")

    def calc_mismatch(self, left, right):
        return len([i for i in range(0, min(len(left), len(right))) if left[i] != right[i]])

    def collapse_with_mismatch(self, mark):
        seq_list = list(mark.keys())
        for i in range(len(seq_list)-2, -1, -1):
            for j in range(len(seq_list)-1, i, -1):
                if self.calc_mismatch(seq_list[i], seq_list[j]) < self.mismatch:
                    if mark[seq_list[i]][1] < mark[seq_list[j]][1]:
                        mark.pop(seq_list[i])
                        seq_list.pop(i)
                        break
                    else:
                        mark.pop(seq_list[j])
                        seq_list.pop(j)
        return mark

    def apply_collapsing(self):
        for file in self.files:
            mark, barcode = self.get_marked_reads(file)
            if self.mismatch > 0:
                mark = self.collapse_with_mismatch(mark)
            self.write_splitted_reads(file, mark, barcode)

    # def write_seq_and_score(self, file, tmpfile, bzflag = False):
    #     barcode = {}
    #     input = open(file) if not bzflag else bz2.BZ2File(file)
    #     with open(tmpfile, 'w') as output:
    #         temp, count = [], 0
    #         while True:
    #             line = input.readline()
    #             if bzflag:  line = line.decode("utf-8")
    #             if line == "":
    #                 break
    #             temp.append(line.rstrip('\n'))
    #             if len(temp) == 4:
    #                 output.write(temp[1]+"\t"+str(sum([ord(i) for i in temp[3]]))+"\t"+str(count)+"\n")
    #                 barcode[self.extract_barcode(temp[1])] = 1
    #                 temp, count = [], count+1
    #     input.close()
    #     return list(barcode.keys())

    # def run_sort(self, tmpfile, sortfile):
    #     cmd = "sort -k1,1 -k2,2n -r "+tmpfile+" > "+sortfile
    #     self.run_shell_cmd(cmd)
    #     self.remove_files(tmpfile)

    # def run_barcode(self, barfile, barcode):
    #     with open(barfile, 'w') as f:
    #         for tbarcode in barcode:
    #             f.write(tbarcode+"\n")

    # def run_mark(self, markfile, mark):
    #     with open(markfile, 'w') as f:
    #         for tmark in mark:
    #             f.write(str(tmark)+"\n")

    def remove_files(self, file):
        os.remove(file)

    # def extract_reads_from_file(self, file, sortfile):
    #     mark, prev = [], ""
    #     with open(sortfile) as input:
    #         while True:
    #             line = input.readline()
    #             if line == "":  break
    #             contents = line.rstrip('\n').split('\t')
    #             if prev == contents[0]:
    #                 mark.append(int(contents[2]))
    #             prev = contents[0]
    #     sys.stderr.write("# To be removed: "+str(len(mark)))
    #     mark.sort()
    #     return mark

    # def output_lines(self, temp, tbarcode, output):
    #     if tbarcode == "" or self.extract_barcode(temp[1]) == tbarcode:
    #         if self.remove:
    #             temp[1] = temp[1][len(self.barcode):]
    #             temp[3] = temp[3][len(self.barcode):]
    #         for elem in temp:
    #             output.write(elem)

    # def write_subjected_reads_from_file(self, file, sortfile, ofile, tbarcode, mark, bzflag = False):
    #     input = open(file) if not bzflag else bz2.BZ2File(file)
    #     outfile = ofile
    #     if self.job_all > 1:
    #         outfile = ofile+"_"+str(self.job_id)
    #     output = open(outfile, 'w') if not self.stdout else sys.stdout
    #     count = 0
    #     for tmark in mark:
    #         while count <= tmark:
    #             temp = [input.readline() for i in range(4)]
    #             if count%self.job_all != self.job_id: continue
    #             count = count+1
    #             self.output_lines(temp, tbarcode, output)
    #     while True:
    #         temp = [input.readline() for i in range(4)]
    #         if "" in temp:  break
    #         if count%self.job_all != self.job_id: continue
    #         count = count+1
    #         self.output_lines(temp, tbarcode, output)
    #     if not self.stdout:
    #         output.close()
    #     input.close()


    # def write_all_reads_from_file(self, file, sortfile, ofile, mark, bzflag = False):
    #     if bzflag:
    #         cat = "bzcat"
    #     else:
    #         cat = "cat"
    #     pre, last, append = 0, 0, False
    #     outfile = ofile
    #     if self.job_all > 1:
    #         outfile = ofile+"_"+str(self.job_id)
    #     for (i, tmark) in enumerate(mark):
    #         if i%self.job_all != 0:
    #             pre = tmark+1
    #             continue
    #         if tmark > pre:
    #             cmd = cat+" "+file+" 2>/dev/null | tail -n+"+str(pre*4+1)+" 2>/dev/null | head -n "+str(tmark*4-pre*4)+" >"
    #             # cmd = cat+" "+file+" | sed -n \""+str(pre*4+1)+","+str(tmark*4)+"p;"+str(tmark*4+1)+"q\" >"
    #             if append: cmd += ">"
    #             cmd += " "+outfile
    #             self.run_shell_cmd(cmd)
    #             append = True
    #         last = tmark
    #         pre = tmark+1
    #     if pre == last+1:
    #         cmd = cat+" "+file+" 2>/dev/null| tail -n+"+str(pre*4+1)+" >> "+outfile
    #     self.run_shell_cmd(cmd)

    # def read_barcode(self, barfile):
    #     barcode = []
    #     with open(barfile) as f:
    #         barcode = [line.rstrip('\n') for line in f.readlines()]
    #     return barcode

    # def read_mark(self, markfile):
    #     with open(markfile) as f:
    #         while True:
    #             line = f.readline().rstrip('\n')
    #             if line == "":  break
    #             yield int(line)

    def get_base_name(self, file):
        basename = file.split('.')
        while basename[-1] in ['fq', 'fastq', 'bz2']:  basename.pop(-1)
        return ".".join(basename)

    # def write_splitted_reads_from_file(self, file, sortfile, barfile, markfile, bzflag):
    #     basename = self.get_base_name(file)
    #     mark = self.read_mark(markfile)
    #     if self.split:
    #         barcode = self.read_barcode(barfile)
    #         for tbarcode in barcode:
    #             if len(tbarcode) == len([i for i in self.barcode if i == "X"]):
    #                 ofile = basename+"_"+tbarcode+".fastq"
    #                 sys.stderr.write("Write to "+ofile+"...\n")
    #                 self.write_subjected_reads_from_file(file, sortfile, ofile, tbarcode, mark, bzflag)
    #     else:
    #         ofile = basename+"_nosplit.fastq"
    #         # self.write_subjected_reads_from_file(file, sortfile, ofile, "", mark, bzflag)
    #         self.write_all_reads_from_file(file, sortfile, ofile, mark, bzflag)

    # def get_file_list(self, file):
    #     bzflag = (file[len(file)-3:len(file)] == "bz2")
    #     tmpfile = file+".tmp"
    #     barfile = file+".barcode"
    #     sortfile = file+".sorted"
    #     markfile = file+".mark"
    #     return bzflag, tmpfile, barfile, sortfile, markfile

    # def apply_collapsing_to_file(self):
    #     for file in self.files:
    #         bzflag, tmpfile, barfile, sortfile, markfile = self.get_file_list(file)
    #         barcode = self.write_seq_and_score(file, tmpfile, bzflag)
    #         self.run_sort(tmpfile, sortfile)
    #         self.run_barcode(barfile, barcode)
    #         mark = self.extract_reads_from_file(file, sortfile)
    #         self.run_mark(markfile, mark)

    # def write_collapsed_reads(self):
    #     for file in self.files:
    #         bzflag, tmpfile, barfile, sortfile, markfile = self.get_file_list(file)
    #         self.write_splitted_reads_from_file(file, sortfile, barfile, markfile, bzflag)

    def run_shell_cmd(self, cmd):
        sys.stderr.write("# running command: "+cmd+"\n")
        process = subprocess.Popen(cmd, shell=True, stdout=False)
        process.communicate()

    # def combine_files(self, ofile):
    #     if self.job_all > 1:
    #         for i in range(self.job_all):
    #             cmd = "cat "+ofile+"_"+str(i)+" >"
    #             if i > 0:  cmd += ">"
    #             cmd = " "+ofile
    #             self.run_shell_cmd(cmd)

    # def remove_combined_files(self, ofile):
    #     if self.job_all > 1:
    #         for i in range(self.job_all):
    #             self.remove_files(ofile+"_"+str(i))

    # def clean_temp_files(self):
    #     for file in self.files:
    #         basename = self.get_base_name(file)
    #         bzflag, tmpfile, barfile, sortfile, markfile = self.get_file_list(file)
    #         if self.split:
    #             barcode = self.read_barcode(barfile)
    #             for tbarcode in barcode:
    #                 if len(tbarcode) == len([i for i in self.barcode if i == "X"]):
    #                     ofile = basename+"_"+tbarcode+".fastq"
    #                     self.combine_files(ofile)
    #                     # self.remove_combined_files(ofile)
    #         else:
    #             ofile = basename+"_nosplit.fastq"
    #             self.combine_files(ofile)
    #           # self.remove_combined_files(ofile)
    #        # self.remove_files(sortfile)
    #        # self.remove_files(barfile)
    #        # self.remove_files(markfile)

    def get_score_file(self, file):
        bzflag = (file[len(file)-3:len(file)] == "bz2")
        return bzflag, file+"_score.txt", file+"_tmp"

    def read_and_write_score(self, file, scorefile, bzflag = False):
        input = open(file) if not bzflag else bz2.BZ2File(file)
        barcode = {}
        with open(scorefile, 'w') as output:
            temp, count = [], 0
            while True:
                temp = [input.readline().rstrip('\n') for i in range(4)]
                if "" in temp: break
                tbarcode = self.extract_barcode(temp[1])
                if tbarcode not in barcode:    barcode[tbarcode] = str(len(barcode.keys()))
                output.write(str(sum([ord(i) for i in temp[3]]))+"\t"+str(barcode[tbarcode])+" \n")
                count = count+1
        input.close()
        return barcode

    def score_sort_and_extract_uniq(self, file, ofile, scorefile, cut, bzflag):
        if bzflag:
            cat = "bzcat"
        else:
            cat = "cat"
        if self.split:
            cmd = cat+" "+file+" | paste "+scorefile+" - - - - | sort -k4,4 -k1,1n -t$'\t' -r > "+ofile
        else:
            cmd = cat+" "+file+" | paste "+scorefile+" - - - - | sort -k4,4 -k1,1n -t$'\t' -r | awk -F'\t' 'BEGIN{seq=\"\"}{if(seq != $4){print $3; print substr($4, cut); print $5; print $6; seq=$4;}}' > "+ofile
        self.run_shell_cmd(cmd)

    def score_sort_and_extract_barcode(self, tempfile, ofile, scorefile, id, cut, bzflag):
        if bzflag:
            cat = "bzcat"
        else:
            cat = "cat"
        cmd = cat+" "+tempfile+" | awk -F'\t' 'BEGIN{seq=\"\"}{if($2 == id && seq != $4){print $3; print substr($4, cut); print $5; print $6; seq=$4;}}' > "+ofile
        self.run_shell_cmd(cmd)

    def write_score_file(self):
        if self.remove: cut = len(self.barcode)+1
        else:   cut = 0
        for file in self.files:
            basename = self.get_base_name(file)
            bzflag, scorefile, tempfile = self.get_score_file(file)
            barcode = self.read_and_write_score(file, scorefile, bzflag)
            if self.split:
                self.score_sort_and_extract_uniq(file, tempfile, scorefile, cut, bzflag)
                for tbarcode in barcode.keys():
                    if len(tbarcode) == len([i for i in self.barcode if i == "X"]):
                        ofile = basename+"_"+tbarcode+".fastq"
                        self.score_sort_and_extract_barcode(tempfile, ofile, scorefile, barcode[tbarcode], cut, bzflag)
                self.remove_files(tempfile)
            else:
                ofile = basename+"_nosplit.fastq"
                self.score_sort_and_extract_uniq(file, ofile, scorefile, cut, bzflag)
            self.remove_files(scorefile)

if __name__ == '__main__':
    parser = get_parser()
    options = parser.parse_args()
    collapse = readCollapse(options)
    if options.shell:
        collapse.write_score_file()
        # collapse.sort_and_extract_reads()
        # collapse.apply_collapsing_to_file()
        # if (collapse.job_id == 0 and collapse.job_all  == 1):
        #     # collapse.apply_collapsing_to_file()
        #     collapse.write_collapsed_reads()
        #     collapse.clean_temp_files()
        # else:
        #     if collapse.build:
        #         collapse.apply_collapsing_to_file()
        #     elif not collapse.clean:
        #         collapse.write_collapsed_reads()
        #     else:
        #         collapse.clean_temp_files()
    else:
        collapse.apply_collapsing()

