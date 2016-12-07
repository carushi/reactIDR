#!/usr/bin/env python
import argparse
import subprocess
import tempfile
import sys
import os

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("bedfiles", nargs='+', type=str, help="read bed files", metavar="INPUT")
    parser.add_argument("--gtf", dest="gtf", help="gtf file to extract transcripts", metavar="GTF", default="", required=False)
    parser.add_argument("--gff", dest="gff", help="gff file to extract transcripts", metavar="GFF", default="", required=False)
    parser.add_argument("--feature", dest="feature", help="feature to count reads", metavar="FEATURE", default="exon", required=False)
    parser.add_argument("-g", dest="id", help="feature to summarive reads", metavar="ID", default="transcript_id", required=False)
    parser.add_argument("--stdout", dest="stdout", action="store_true", help="print results to stdout", required=False)
    parser.add_argument("--name", dest="name", action="store_true", help="print names to which more than one read is assigned", required=False)
    parser.add_argument('--chr', dest="chr", help="chromosome conversion for refseq", metavar="CHROMOSOME", default="", required=False)
    parser.add_argument('--fasta', dest="fasta", help="required to calculate base ratio at mapped position", metavar="FASTA", default="", required=False)
    parser.add_argument('--offset', dest="offset", help="offset for the target position (PARS: 0, DMS&SHAPE: -1...)", default=0, type=int, required=False)
    parser.set_defaults(stdout=False, name=False)
    return parser

def run_command(cmd):
    print("# running command: "+cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=False)
    process.communicate()

def write_coverage_data(options, file, afile, ind):
    cmd = "grep \"\t"+options.feature+"\t\" "+options.gff+" | sort -k1,1 -k"+str(ind)+","+str(ind)+" -k4,4n > "+afile
    run_command(cmd)

def write_values(contents, value, options, out, prev):
    if len(contents) == 0:  return
    if contents[6] == "-":  value = value[::-1]
    output = prev+"\t"+";".join(value)+'\n'
    if options.stdout:
        print(output, end='')
    else:
        out.write(output)

def get_count_dict(file):
    counts = {}
    with open(file) as f:
        for line in f.readlines():
            contents = line.rstrip('\n').split('\t')
            if len(contents) < 6:   continue
            if contents[0] not in counts.keys():
                counts[contents[0]] = {}
                counts[contents[0]]["+"] = {}
                counts[contents[0]]["-"] = {}
            if contents[5] == "+":
                counts[contents[0]][contents[5]][int(contents[1])+options.offset] = contents[4]
            else:
                counts[contents[0]][contents[5]][int(contents[1])-options.offset] = contents[4]
    return counts

def write_pars_format_data(options, file, names, afile):
    out = open(os.path.basename(file)+".tab", 'w') if not options.stdout else sys.stdout
    read_counts = get_count_dict(file)
    contents, value, count, prev = [], [], 0, ""
    with open(afile) as f:
        for line in f.readlines():
            temp = get_id_name(line.rstrip('\n').split('\t'))
            if temp != prev:
                if len(prev) > 0:
                    write_values(contents, value, options, out, prev)
                    if options.name and len([v for v in value if v != "0"]) > 0:
                        names.append(prev)
                contents = line.rstrip('\n').split('\t')
                value = []
                prev = temp
            if contents[0] not in read_counts:
                value.extend(['0']*(int(contents[4])-int(contents[3])))
            else:
                for i in range(int(contents[3]), int(contents[4])):
                    if contents[0] not in read_counts:
                        value.append('0')
                    if i in read_counts[contents[0]][contents[6]]:
                        value.append(read_counts[contents[0]][contents[6]][i])
                    else:
                        value.append('0')
    if len(prev) > 0:
        write_values(contents, value, options, out, prev)
        if options.name and len([v for v in value if v != "0"]) > 0:
            names.append(get_id_name(options, contents))
    if not options.stdout:
        out.close()
    return names

def get_chr_dict(chr_file):
    dict = {}
    if os.path.exists(chr_file):
        with open(chr_file) as f:
            for line in f.readlines():
                contents = line.rstrip('\n').split('\t')
                assert len(contents) >= 2
                dict[contents[1]] = contents[0]
    return dict

def get_id_name(options, contents):
    pos = contents[8].find(options.id)
    char = '='
    if contents[8][pos+len(options.id)] == ':':
        char = ':'
    # print(contents[8], options.id, pos, char)
    if pos < 0:
        return ""
        # return contents[8]
    else:
        return contents[8][pos:].split(';')[0].split(char)[1]

def get_chr(chr, chr_dict, keys):
    if chr in keys:
        return chr
    elif chr in chr_dict:
        return chr_dict[chr]
    else:
        return 'chr'+chr

def write_values_chr(fa, strand, value, options, out):
    output = fa+strand+"\t"+";".join(value)+'\n'
    if options.stdout:
        print(output, end='')
    else:
        out.write(output)


def baseToIndex(c, strand):
    c = str(c.upper())
    if strand == "-":
        c = c.translate(str.maketrans('ACGUTN', 'TGCAAN'))
    if c == "A": return 0
    elif c == "C":  return 1
    elif c == "G":  return 2
    elif c == "U" or c == "T":  return 3
    else:   return 4

def get_sequence(fasta):
    with open(fasta) as f:
        name, seq = '', ''
        while True:
            line = f.readline().rstrip('\n')
            if line == "":
                yield name, seq
                break
            if line[0] == ">":
                if name != '':  yield name, seq
                name, seq = line[1:], ''
            else:
                seq += line.rstrip('\n')

def write_pars_format_data_chr(options, file, names):
    out = open(os.path.basename(file)+".tab", 'w') if not options.stdout else sys.stdout
    read_counts = get_count_dict(file)
    baseCount = [0]*5
    for fa, seq in get_sequence(options.fasta):
        if fa in read_counts:
            for strand in read_counts[fa]:
                value = []
                for i in range(len(seq)):
                    if i+1 not in read_counts[fa][strand]: # bed -> 1-based
                        value.append('0')
                    else:
                        value.append(read_counts[fa][strand][i+1])
                if strand == "-":  value = value[::-1]
                write_values_chr(fa, strand, value, options, out)
                if len([v for v in value if v != "0"]) > 0: names.append(fa+strand)
                for i, c in enumerate(seq):
                    baseCount[baseToIndex(c, strand)] += int(value[i])
        else:
            value = ['0']*len(seq)
            write_values_chr(fa, "+", value, options, out)
            write_values_chr(fa, "-", value, options, out)
    if not options.stdout:
        out.close()
    for i, c in zip(baseCount, 'ACGTN'):
        print("# Count "+c+" = "+str(i))
    return names

def write_pars_format_data_gff(options, file, names, afile):
    out = open(file+".tab", 'w') if not options.stdout else sys.stdout
    read_counts = get_count_dict(file)
    chr_dict = get_chr_dict(options.chr)
    contents, value, count, prev = [], [], 0, ""
    with open(afile) as f:
        for line in f.readlines():
            tcont = line.rstrip('\n').split('\t')
            if options.feature != tcont[2]:  continue
            temp = get_id_name(options, tcont)
            if temp != prev:
                if len(prev) > 0:
                    write_values(contents, value, options, out, prev)
                    if options.name and len([v for v in value if v != "0"]) > 0:
                        names.append(prev)
                contents = tcont
                value = []
                prev = temp
            if len(contents) == 0:  continue
            contents[0] = get_chr(contents[0], chr_dict, read_counts.keys())
            if contents[0] not in read_counts:
                value.extend(['0']*(int(contents[4])-int(contents[3])))
            else:
                for i in range(int(contents[3]), int(contents[4])):
                    if contents[0] not in read_counts:
                        value.append('0')
                    if i in read_counts[contents[0]][contents[6]]: # both gff and bed are 1-based
                        value.append(read_counts[contents[0]][contents[6]][i])
                    else:
                        value.append('0')
    if len(prev) > 0:
        write_values(contents, value, options, out, prev)
        if options.name and len([v for v in value if v != "0"]) > 0:
            names.append(get_id_name(options, contents))
    if not options.stdout:
        out.close()
    return names

def map_to_gff_file(options, file, names):
    write_coverage_data(options, file, file+".ann", 9)
    names = write_pars_format_data_gff(options, file, names, file+".ann")
    os.remove(file+".ann")
    return names

def map_to_gtf_file(options, file, names):
    write_coverage_data(options, file, file+".ann", 9)
    names = write_pars_format_data(options, file, names, file+".ann")
    os.remove(file+".ann")
    return names

def file_convert(options):
    # combine_files(options.bedfiles)
    names = []
    for file in options.bedfiles:
        if len(options.gff) > 0:
            names = map_to_gff_file(options, file, names)
        else:
            names = map_to_gtf_file(options, file, names)
    if options.name:
        if len(names) > 0:
            print("\n".join(list(set(names))))

def file_convert_chr(options):
    names = []
    for file in options.bedfiles:
        names = write_pars_format_data_chr(options, file, names)
    if options.name:
        if len(names) > 0:
            print("\n".join(list(set(names))))

if __name__ == '__main__':
    parser = get_parser()
    options = parser.parse_args()
    if len(options.bedfiles) > 0:
        if len(options.gtf) == 0 and len(options.gff) == 0:
            file_convert_chr(options)
        else:
            file_convert(options)
