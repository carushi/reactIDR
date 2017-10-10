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
    parser.add_argument("--bam", dest="bam", help="bam file to extract reference length", metavar="BAM", default="", required=False)
    parser.add_argument("--feature", dest="feature", help="feature to count reads", metavar="FEATURE", default="exon", required=False)
    parser.add_argument("-g", dest="id", help="feature to summarive reads", metavar="ID", default="transcript_id", required=False)
    parser.add_argument("--stdout", dest="stdout", action="store_true", help="print results to stdout", required=False)
    parser.add_argument("--name", dest="name", action="store_true", help="print names to which more than one read is assigned", required=False)
    parser.add_argument('--chr', dest="chr", help="chromosome conversion for refseq", metavar="CHROMOSOME", default="", required=False)
    parser.add_argument('--fasta', dest="fasta", help="required to calculate base ratio at mapped position", metavar="FASTA", default="", required=False)
    parser.add_argument('--offset', dest="offset", help="offset for the target position (PARS: 0, DMS&SHAPE: -1...)", default=0, type=int, required=False)
    parser.add_argument("--memory", dest="memory", action="store_true", help="Memory saving mode. Files should be sorted.", required=False)
    parser.set_defaults(stdout=False, name=False)
    return parser

def run_command(cmd):
    print("# running command: "+cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=False)
    process.communicate()

def write_coverage_data(options, file, afile, ind, ann):
    cmd = "grep \"\t"+options.feature+"\t\" "+ann+" | sort -T ./ -k1,1 -k"+str(ind)+","+str(ind)+" -k4,4n > "+afile
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
            if len(contents) < 6:
                sys.stderr.write('Warning: a line is truncated.'+" ".join(contents)+"\n")
                continue
            if contents[0] not in counts.keys():
                counts[contents[0]] = {}
                counts[contents[0]]["+"] = {}
                counts[contents[0]]["-"] = {}
            if contents[4]  == '':
                sys.stderr.write('Warning: a line is truncated.'+" ".join(contents)+"\n")
                # counts[contents[0]][contents[5]][int(contents[1])+options.offset] = '0'
            else:
                if contents[5] == "+":
                    off = options.offset
                else:
                    off = -options.offset
                for i in range(int(contents[1]), int(contents[2])):
                    counts[contents[0]][contents[5]][i+off] = contents[4]
    return counts

def read_count_dict_iter(file):
    with open(file) as f:
        while True:
            line = f.readline()
            if line == '':
                break
            yield line.rstrip('\n').split('\t')

def get_count_dict_key(lines):
    counts = {lines[0][0]:{lines[0][5]:{}}}
    for contents in lines:
        if len(contents) < 6 or contents[4] == '':
            sys.stderr.write('Warning: a line is truncated.'+" ".join(contents)+"\n")
            continue
        else:
            if contents[5] == "+":
                off = options.offset
            else:
                off = -options.offset
            for i in range(int(contents[1]), int(contents[2])):
                counts[contents[0]][contents[5]][i+off] = contents[4]
    return counts

# def get_count_dict_keys(file):
#     keys = {}
#     with open(file) as f:
#         temp = [ (line.split('\t')[0], i) if len(line.split('\t')) >= 6 else ('nan', i) for i, line in enumerate(f.readlines()[0:10000]) ]
#     temp = sorted(temp, key=lambda x: x[0])
#     ids, index = zip(*temp)
#     del temp
#     assert len(ids) == len(index)
#     temp, prev = [], ""
#     for j in range(len(index)):
#         if j%100000 == 0:
#             print("#key reading", j, ids[j], index[j])
#             sys.stdout.flush()
#         if prev != ids[j]:
#             if len(prev) > 0:
#                 if prev == 'nan':
#                     sys.stderr.write('Warning: "+",".join(map(str, temp))+"th lines might be truncated.'+"\n")
#                 else:
#                     keys[prev] = (min(temp), max(temp))
#             temp, prev = [], ids[j]
#         temp.append(index[j])
#     if len(prev) > 0 and prev != 'nan':
#         keys[prev] = [min(temp), max(temp)]
#     del ids
#     del index
#     return keys

def write_pars_format_data(options, file, names, afile):
    out = open(os.path.basename(file)+".tab", 'w') if not options.stdout else sys.stdout
    read_counts = get_count_dict(file)
    contents, value, count, prev = [], [], 0, ""
    with open(afile) as f:
        for line in f.readlines():
            temp = get_id_name(options, line.rstrip('\n').split('\t'))
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
    if len(contents) < 9:
        return ""
    pos = contents[8].find(options.id)
    char = '='
    if contents[8][pos+len(options.id)] == ':':
        char = ':'
    elif char not in contents[8][pos:]:
        char = ' '
    if pos < 0:
        return ""
        # return contents[8]
    else:
        id = contents[8][pos:].split(';')[0].split(char)[1]
        return id.strip('\"')

def get_chr(chr, chr_dict, keys):
    if chr in keys:
        return chr
    elif chr in chr_dict:
        return chr_dict[chr]
    else:
        return 'chr'+chr

def write_values_chr(fa, strand, value, options, out):
    output = fa.split('|')[0]+strand+"\t"+";".join(value)+'\n'
    if options.stdout:
        print(output, end='')
    else:
        out.write(output)
    sys.stdout.flush()


def baseToIndex(c, strand):
    c = str(c.upper())
    if strand == "-":
        c = c.translate(str.maketrans('ACGUTN', 'TGCAAN'))
    if c == "A": return 0
    elif c == "C":  return 1
    elif c == "G":  return 2
    elif c == "U" or c == "T":  return 3
    else:   return 4

def get_sequence_key(fasta):
    assert fasta != "" or bam != ""
    seq_list = []
    name_list = []
    with open(fasta) as f:
        name, seq = '', ''
        for line in f.readlines():
            if line == '':  continue
            if line[0] == ">":
                if name != '':
                    seq_list.append(seq)
                    name_list.append(name)
                name, seq = line[1:], ''
            else:
                if len(name) > 0:
                    seq += line.rstrip('\n')
    return name_list, seq_list

def get_sequence(fasta, bam):
    assert fasta != "" or bam != ""
    if len(fasta) > 0:
        with open(fasta) as f:
            name, seq = '', ''
            for line in f.readlines():
                line = line.rstrip('\n')
                if line == '':  continue
                if line[0] == ">":
                    if name != '':
                        yield name, seq
                    name, seq = line[1:], ''
                else:
                    seq += line.rstrip('\n')
        if len(name) > 0:
            yield name, seq
    else:
        cmd = "samtools view -H "+bam+" >"+bam+".ann"
        run_command(cmd)
        with open(bam+".ann") as f:
            for line in f.readlines():
                if line[0:3] == "@SQ":
                    name, seq = line.split('\t')[1].split(':')[-1], "N"*int(line.split(':')[-1])
                    yield name, seq
        os.remove(bam+".ann")

def write_pars_format_data_chr(options, file, names):
    out = open(os.path.basename(file)+".tab", 'w') if not options.stdout else sys.stdout
    read_counts = get_count_dict(file)
    baseCount = [0]*5
    for fa, seq in get_sequence(options.fasta, options.bam):
        print(fa, seq)
        if fa in read_counts:
            for strand in read_counts[fa]:
                value = []
                for i in range(len(seq)):
                    if i+1 in read_counts[fa][strand]: # bed -> 1-based
                        # print(fa, strand, i+1, read_counts[fa][strand][i+1])
                        value.append(read_counts[fa][strand][i+1])
                    else:
                        value.append('0')
                if strand == "-":
                    value = value[::-1]
                write_values_chr(fa, strand, value, options, out)
                if len([v for v in value if v != "0"]) > 0: names.append(fa+strand)
                for i, c in enumerate(seq):
                    baseCount[baseToIndex(c, strand)] += int(float(value[i]))
        else:
            value = ['0']*len(seq)
            write_values_chr(fa, "+", value, options, out)
            write_values_chr(fa, "-", value, options, out)
    if not options.stdout:
        out.close()
    print("# File "+file)
    for i, c in zip(baseCount, 'ACGTN'):
        print("# Count "+c+" = "+str(i))
    return names

def write_pars_format_data_chr_memory_saving(options, file, names):
    out = open(os.path.basename(file)+".tab", 'w') if not options.stdout else sys.stdout
    # print("Transcript type =", len(keys.keys()))
    baseCount = [0]*5
    data = []
    fa, seq = get_sequence_key(options.fasta)
    for line in read_count_dict_iter(file):
        if len(data) == 0 or data[0] == line[0]:
            data.append(line)
        else:
            if line[0] in fa:
                out, baseCount = write_data_each_line(options, data, out, seq[fa.index(line[0])], baseCount)
            data = []
    if len(data):
        out, baseCount = write_data_each_line(options, data, out, seq[fa.index(line[0])], baseCount)
    if not options.stdout:
        out.close()
    print("# File "+file)
    for i, c in zip(baseCount, 'ACGTN'):
        print("# Count "+c+" = "+str(i))
    return names

def write_data_each_line(options, lines, out, seq, baseCount):
    if len(seq) > 0:
        read_counts = get_count_dict_key(lines)
        for strand in read_counts[fa]:
            if len(strand) == 0:
                continue
            value = []
            for i in range(len(seq)):
                if i+1 in read_counts[fa][strand]: # bed -> 1-based
                    value.append(read_counts[fa][strand][i+1])
                else:
                    value.append('0')
            if strand == "-":
                value = value[::-1]
            write_values_chr(fa, strand, value, options, out)
            sys.stdout.flush()
            if len([v for v in value if v != "0"]) > 0: names.append(fa+strand)
            for i, c in enumerate(seq):
                baseCount[baseToIndex(c, strand)] += int(float(value[i]))
    else:
        pass
    return out, baseCount

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
    write_coverage_data(options, file, file+".ann", 9, options.gff)
    names = write_pars_format_data_gff(options, file, names, file+".ann")
    os.remove(file+".ann")
    return names

def map_to_gtf_file(options, file, names):
    write_coverage_data(options, file, file+".ann", 9, options.gtf)
    names = write_pars_format_data(options, file, names, file+".ann")
    os.remove(file+".ann")
    return names

def file_convert(options):
    # combine_files(options.bedfiles)
    names = []
    for file in options.bedfiles:
        print(options)
        if len(options.gff) > 0:
            names = map_to_gff_file(options, file, names)
        elif len(options.gtf) > 0:
            names = map_to_gtf_file(options, file, names)
        else:
            assert False, "No annotation file."
    if options.name:
        if len(names) > 0:
            print("\n".join(list(set(names))))

def file_convert_chr(options):
    names = []
    for file in options.bedfiles:
        if options.memory:
            names = write_pars_format_data_chr_memory_saving(options, file, names)
        else:
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
