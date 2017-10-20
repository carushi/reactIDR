import sys

def print_pars_format_score(fname1, fname2):
    dict = {}
    with open(fname1) as f:
        f.readline()
        while True:
            line = f.readline().rstrip('\n')
            if line == '':
                break
            contents = line.split('\t')
            name = contents[0]
            expression = contents[1]
            if name in ['RNA18S5', 'RNA28S5', 'RNA5-8S5', 'ENSG00000201321|ENST00000364451']:
                with open('seq_'+name+'+.fa') as sf:
                    seq = sf.readlines()[1].rstrip('\n')
                    # print(len(seq))
                # print(name, len(contents[2:]))
                nan_len = len(seq)-len(contents[2:])
                dict[name] = ';'.join(contents[2:]+['None']*nan_len)
    print('\t'.join(['key', 'score0', 'score1']))
    with open(fname2) as f:
        f.readline()
        while True:
            line = f.readline().rstrip('\n')
            if line == '':
                break
            contents = line.split('\t')
            name = contents[0]
            expression = contents[1]
            if name in ['RNA18S5', 'RNA28S5', 'RNA5-8S5', 'ENSG00000201321|ENST00000364451']:
                with open('seq_'+name+'+.fa') as sf:
                    seq = sf.readlines()[1].rstrip('\n')
                    # print(len(seq))
                print('\t'.join([name+'+', dict[name], ';'.join(contents[2:]+['None']*nan_len)]))

def print_pars_format(fname):
    with open(fname) as f:
        f.readline()
        while True:
            line = f.readline().rstrip('\n')
            if line == '':
                break
            contents = line.split('\t')
            name = contents[0]
            expression = contents[1]
            if name in ['RNA18S5', 'RNA28S5', 'RNA5-8S5', 'ENSG00000201321|ENST00000364451']:
                with open('seq_'+name+'+.fa') as sf:
                    seq = sf.readlines()[1].rstrip('\n')
                    # print(len(seq))
                # print(name, len(contents[2:]))
                nan_len = len(seq)-len(contents[2:])
                print('\t'.join(['PROBer', name+'+', 'case'])+'\t'+';'.join(contents[2:]+['None']*nan_len))
                print('\t'.join(['PROBer', name+'+', 'cont'])+'\t'+';'.join([str(1.-float(x)) for x in contents[2:]]+['None']*nan_len))

if __name__ == '__main__':
    if len(sys.argv) > 2:
        print_pars_format_score(sys.argv[1], sys.argv[2])
    else:
        print_pars_format(sys.argv[1])
