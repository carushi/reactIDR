import sys
import numpy as np

def log(s):
    print(s)
    sys.stdout.flush()

MAXLINE=-1

def common_transcript(dict1, dict2, dict3=None):
    if dict3 is None:
        return set(dict1.keys()) & set(dict2.keys())
    else:
        return set(dict1.keys()) & set(dict2.keys()) & set(dict3.keys())

def parse_line(line, func):
    contents = line.rstrip('\n').split('\t')
    # return contents[0], list(map(int, map(float, contents[1].split(';'))))
    return contents[0], list(map(func, contents[1].split(';')))

def parse_file_pars(file, func, header=True):
    global MAXLINE
    tdict = {}
    start, count = 0, 0
    if header:
        start = 1
    with open(file) as f:
        for line in f.readlines()[start:]:
            name, data = parse_line(line, func)
            tdict[name] = data
            count += 1
            if MAXLINE > 0 and count > MAXLINE: break
    return tdict

def parse_file_bed(file, func, header=True):
    tdict = {}
    start = 0
    if header:
        start = 1
    with open(file) as f:
        for line in f.readlines()[start:]:
            contents = line.rstrip('\n').split('\t')
            name, data = "_".join(contents[0:4]), [float(contents[4])]
            tdict[name] = data
        return tdict

def get_score_dict(file, format, func = int):
    if format == "PARS":
        return parse_file_pars(file, func)

def parse_score(file):
    dict1, dict2 = {}, {}
    with open(file) as f:
        f.readline()
        while True:
            line = f.readline()
            if line == "":  break
            key, score1, score2 = line.rstrip('\n').split('\t')
            if len(score1) > 0:
                dict1[key] = list(map(int, score1.split(';')))
            if len(score2) > 0:
                dict2[key] = list(map(int, score2.split(';')))
    return dict1, dict2

def parse_idr(file, func = int):
    tdict = {}
    with open(file) as f:
        f.readline()
        while True:
            line = f.readline()
            if line == "":  break
            key, idx1, idx2, IDR = line.rstrip('\n').split('\t')
            tdict[key] = list(map(func, IDR.split(';')))
    return tdict

def parse_score_iterator(file, header=True):
    with open(file) as f:
        while True:
            line = f.readline()
            if line == "":  break
            if header or ';' not in line:
                header = False
                continue
            key, score1, score2 = line.rstrip('\n').split('\t')
            if len(score1) > 0 and len(score2) > 0:
                yield key, list(map(int, score1.split(';'))), list(map(int, score2.split(';')))

def parse_bed_tri_iterator(file, header=True):
    with open(file) as f:
        while True:
            line = f.readline()
            if line == "":  break
            if header or ';' not in line:
                header = False
                continue
            contents = line.rstrip('\n').split('\t')
            yield "_".join(contents[0:4]), [int(contents[4])]

def parse_score_tri_iterator(file, header=True):
    with open(file) as f:
        while True:
            line = f.readline()
            if line == "":  break
            if header or ';' not in line:
                header = False
                continue
            contents = line.rstrip('\n').split('\t')
            key = contents[0].split('|')[0]
            if '|' in contents[0]:  key += contents[0].split('|')[-1]
            yield key, [list(map(float, score.split(';'))) for score in contents[1:]]

def get_print_str_score(data):
    seq = '\t'
    seq += ';'.join(list(map(str, data)))
    return seq

def set_of_nested_dicts(l):
    x = []
    for s in l:
        x.extend(s.keys())
    return sorted(list(set(x)))

def print_score_data(fname, dicts):
    log("Print score.")
    with open(fname, 'w') as f:
        f.write('key\t'+'\t'.join(['score'+str(i) for i in range(len(dicts))])+'\n')
        for key in set_of_nested_dicts(dicts):
            f.write(key)
            for tdict in dicts:
                f.write(get_print_str_score(tdict[key]))
            f.write('\n')

def read_only_high_expression_pars_multi(fnames, threshold, verbose=True):
    data = []
    target = []
    for index in range(len(fnames)):
        print("# Reading ", fnames[index])
        count, rcount = 0, 0
        if len(fnames[index]) == 0:
            continue
        data.append([])
        for key, tdata in parse_score_tri_iterator(fnames[index]):
            count += 1
            if count%1000 == 0 and verbose:
                print("# Reading lines (%d/%d lines)..." % (rcount, count))
                sys.stdout.flush()
            if len(data[index]) == 0:
                data[index] = [{chr(0):[0]} for t in tdata]
            if key not in target:
                # rm_flag = (len([True for t in tdata if np.nanmax(t) >= threshold]) == 0)
                rm_flag = all([len([True for x in t if x > 0.]) < threshold for t in tdata])
                if rm_flag: continue
            if index == 0:
                target.append(key)
            for i, temp in enumerate(tdata):
                data[index][i][key] = temp
            rcount += 1
    return data

def read_data_pars_format_multi(fnames, verbose=True):
    data = []
    count = 0
    for index in range(len(fnames)):
        if len(fnames[index]) == 0:
            continue
        data.append([])
        for key, tdata in parse_score_tri_iterator(fnames[index]):
            count += 1
            if count%1000 == 0 and verbose:
                print("# Reading lines (%s lines)..." % count)
            if len(data[index]) == 0:
                data[index] = [{chr(0):[0]} for t in tdata]
            for i, temp in enumerate(tdata):
                data[index][i][key] = temp
    return data

def get_dict_common_keys(dicts):
    keys = [tdict.keys() for tdict in dicts]
    seta = None
    for key in keys:
        if seta is None:
            seta = set(key)
        else:
            seta = seta.intersection(key)
    dicts = [dict((k, tdict[k]) for k in seta) for tdict in dicts]
    return dicts

def estimate_rpkm(rep, total):
    rpkm = sum([sum(t) for t in rep])
    return rpkm/len(rep[1])*10000000000/total


# def parse_score_rowwise(file, sep):
#     with open(file) as f:
#         header = f.readline().rstrip('\n')
#         header.split(sep)
#         while True:
#             line = f.readline()
#             if line == "":  break
#             key, score1, score2 = line.rstrip('\n').split('\t')
#             if len(score1) > 0 and len(score2) > 0:
#                 yield key, list(map(int, score1.split(';'))), list(map(int, score2.split(';')))
#
# def parse_idr_iterator(file, func = int):
#     tdict = {}
#     with open(file) as f:
#         f.readline()
#         while True:
#             line = f.readline()
#             if line == "":  break
#             key, idx1, idx2, IDR = line.rstrip('\n').split('\t')
#             yield key, list(map(func, IDR.split(';')))
#
#
#
