import argparse
import subprocess
import tempfile
import sys
import os

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("bedfiles", nargs='+', type=str, help="read bed files", metavar="INPUT")
    parser.add_argument("-g", "--gtf", dest="gtf", help="gtf file to extract transcripts", metavar="GTF", default="", required=False)
    parser.add_argument("--gff", dest="gff", help="gff file to extract transcripts", metavar="GFF", default="", required=False)
    parser.add_argument("--feature", dest="feature", help="feature to count reads", metavar="FEATURE", default="exon", required=False)
    parser.add_argument("--stdout", dest="stdout", action="store_true", help="print results to stdout", required=False)
    parser.add_argument("--name", dest="name", action="store_true", help="print names to which more than one read is assigned", required=False)
    parser.set_defaults(stdout=False, name=False)
    return parser

def write_coverage_data(options, file):
    cmd = "grep "+options.feature+" "+options.gff+" | sort -k1,1 -k9,9 -k4,4n > "+file+".ann"
    print("# running command: "+cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=False)
    process.communicate()
    # cmd = "sort -k1,1 -k2,2n "+file+" > "+file+".sorted"
    # print("# running command: "+cmd)
    # process = subprocess.Popen(cmd, shell=True, stdout=False)
    # process.communicate()
    # cmd = "bedtools coverage -s -a "+file+".ann -b "+file+" -d | sort -k1,1n -k9,9 -k4,4n -k10,10n > "+file+".tmp"
    # # cmd = "bedtools map -a "+file+".ann -b "+file+".sorted -c 5 -o sum -s -null 0 | sort -k1,1n -k9,9 -k4,4n -k10,10n > "+file+".tmp"
    # print("# running command: "+cmd)
    # process = subprocess.Popen(cmd, shell=True)
    # process.communicate()


def write_values(contents, value, options, out):
    if len(contents) == 0:  return
    if contents[6] == "-":  value = value[::-1]
    output = contents[8].split('=')[1]+"\t"+";".join(value)+'\n'
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
            counts[contents[0]][contents[5]][int(contents[1])] = contents[4]
    return counts

def write_pars_format_data(options, file, names):
    out = open(file+".tab", 'w') if not options.stdout else sys.stdout
    read_counts = get_count_dict(file)
    contents, value, count, prev = [], [], 0, ""
    # with open(file+".tmp") as f:
    #     for line in f.readlines():
    #         if count == 0:
    #             temp = line.rstrip('\n').split('\t')[8].split('=')[1]
    #             if temp != prev and len(prev) > 0:
    #                 write_values(contents, value, options, out)
    #                 if options.name and len([v for v in value if v != "0"]) > 0:
    #                     names.append(prev)
    #                 value = []
    #             contents = line.rstrip('\n').split('\t')
    #             value.append(contents[9])
    #             count = int(contents[4])-int(contents[3])-1
    #             prev = temp
    #         else:
    #             value.append(line.rstrip('\n').split('\t')[9])
    #             count -= 1
    # if count == 0:
    #     write_values(contents, value, options, out)
    #     if options.name and len([v for v in value if v != "0"]) > 0: names.append(contents[8].split('=')[1])
    # print(read_counts)
    with open(file+".ann") as f:
        for line in f.readlines():
            temp = line.rstrip('\n').split('\t')[8].split('=')[1]
            if temp != prev:
                if len(prev) > 0:
                    write_values(contents, value, options, out)
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
        write_values(contents, value, options, out)
        if options.name and len([v for v in value if v != "0"]) > 0:    names.append(contents[8].split('=')[1])
    if not options.stdout:
        out.close()
    return names

def map_to_gff_file(options, file, names):
    write_coverage_data(options, file)
    names = write_pars_format_data(options, file, names)
    os.remove(file+".ann")
    # os.remove(file+".tmp")
    # os.remove(file+".sorted")
    return names

def file_convert(options):
    # combine_files(options.bedfiles)
    names = []
    for file in options.bedfiles:
        if len(options.gff) > 0:
            names = map_to_gff_file(options, file, names)
    if options.name:
        if len(names) > 0:
            print("\n".join(list(set(names))))

if __name__ == '__main__':
    parser = get_parser()
    # try:
    options = parser.parse_args()
    if len(options.gtf) == 0 and len(options.gff) == 0:
        raise Exception('No annotation file')
    if len(options.bedfiles) > 0:
        file_convert(options)
    # except Exception:
        # parser.print_help()
        # sys.exit(0)
