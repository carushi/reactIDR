import sys

def log(s):
    print(s)
    sys.stdout.flush()

MAXLINE=100000

def parse_line(line, func):
    contents = line.rstrip('\n').split('\t')
    return contents[0], list(map(func, contents[1].split(';')))

def parse_file_pars(file, func):
    global MAXLINE
    tdict = {}
    count = 0
    with open(file) as f:
        for line in f.readlines():
            name, data = parse_line(line, func)
            tdict[name] = data
            count += 1
            if MAXLINE > 0 and count > MAXLINE: break
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
