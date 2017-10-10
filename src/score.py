import utility
import numpy as np


SCORE_LIST = ["icshape", "pars", "ldr"]

def check_reactivity(react_model):
    global SCORE_LIST
    if react_model in SCORE_LIST:
        return SCORE_LIST.index(react_model)
    else:
        return 0

def top_90_95(vec):
    vec.sort()
    high = [vec[i] for i in range(math.floor(len(vec)*90./100.), math.ceil(len(vec)*95./100.))]
    if len(high) == 0:
        return 1.
    else:
        return np.mean(high)

def normalized_vector(vec, start = 32, end = 32):
    trimmed_vec = [x/lnorm for i, x in enumerate(zip(t, v)) for t, v in vec if i >= start and i < len(t)-end]
    norm = top_90_95(trimmed_vec, lnorm)
    return [x/norm for x in vec]

def calc_icshape_reactivity(back, target, back_cov, alpha = 0.25, subtract = False):
    assert len(back) == len(back_cov) and len(back) == len(target)
    scores = [None]*len(back)
    for i in range(len(back)-1):
        b, bc, t = sum(back[i]), sum(back[i+1]), sum(back[i])
        if subtract:
            bc = bc-b
        if bc > 0:
            scores[i] = (t-alpha*b)/bc
    sortl = [x for x in scores if x is not None].sorted()
    if len(sortl) == 0:
        return [0.]*len(back)
    tmin, tmax = sortl[max(0, math.floor(5./100.*len(sortl))-1)], sortl[min(len(sortl)-1, math.ceil(95./100.*len(sortl))-1)]
    if tmin == tmax:
        return [0.]*len(back)
    return [max(0., min(1., (x-tmin)/(tmax-tmin))) if x is not None else None for i, x in enumerate(scores)]

def calc_pars_norm_factor(stotal, vtotal):
    return (stotal+vtotal)/(2.*stotal), (stotal+vtotal)/(2.*vtotal)

def get_mean_pars(self, rep1Vec, rep2Vec, norm = 1):
    return [np.log10(norm)+np.log10(np.mean(rep1Vec[max(0, i-2):min(len(rep1Vec), i+3)]+rep2Vec[max(0, i-2):min(len(rep1Vec), i+3)])+5.) for i in range(len(rep1Vec))]

# def calc_ldr(self, s1, v1, control = False):
#     if control:
#         result_c = ''
#         s1.sort()
#         for i in range(len(s1)):
#             for j in range(i+1, len(s1)):
#                 result_c += str(np.log10(s1[j]/s1[i]))+","
#         return result_c[0:-1]
#     else:
#         result_t = ''
#         for s in s1:
#             for v in v1:
#                 result_t += str(np.log10(s/v))+","
#         return result_t[0:-1]

# def calc_ldr_features(self, s1, v1):
#     return [sum(s)/sum(v) for s, v in zip(s1, v1)]

# def calc_ldr_distribution(self, svec, vvec, scov, vcov):
#     if len(scov) == 0 or len(vcov) == 0:
#         return ['']*len(svec), ['']*len(svec)
#     s1 = [[svec[i][j]/max(1., scov[i][j]) for j in range(len(svec[i]))] for i in range(len(svec))]
#     v1 = [[vvec[i][j]/max(1., vcov[i][j]) for j in range(len(vvec[i]))] for i in range(len(vvec))]
#     ldr = self.calc_ldr_features(s1, v1)
#     ldr_c = [self.calc_ldr(s1[i], s1[i]) for i in range(len(svec))]
#     ldr_t = [self.calc_ldr(s1[i], v1[i], True) for i in range(len(svec))]
#     return [ldr, ldr_c, ldr_t]

class Reactivity_fitting:
    """Convert read count to reactivity, and file I/O."""
    def __init__(self, data):
        self.data = data
        self.params = {}

    def set_params(self, name, para):
        self.params[name] = para

    def calc_icshape_score(self, srep, scov, vrep, stotal, vtotal):
        srep = [normalized_vector(svec, self.trim, self.trim, stotal) for svec in srep]
        if len(scov) == 0:
            scov = srep
        else:
            scov = [normalized_vector(bvec, self.trim, self.trim) for bvec in scov]
        vrep = [normalized_vector(vvec, self.trim, self.trim, vtotal) for vvec in vrep]
        return self.calc_icshape_reactivity(srep, scov, vrep)

    def calc_total_read_counts(self, ssfile, vsfile, parsnorm=[]):
        if len(parsnorm) == 2:
            stotal, vtotal = float(parsnorm[1]), float(parsnorm[0])
        else:
            srep, vrep = self.extract_dictionaries_iterator_double_columns(ssfile, vsfile)
            stotal = max(sum([sum(t[1])+sum(t[2]) for t in srep]), 1)
            vtotal = max(sum([sum(t[1])+sum(t[2]) for t in vrep]), 1)
        return stotal, vtotal


def score_to_reactivity(cont, case, score_ind, output=False, head='plot_fitting_'):
    if score_ind <= 0:
        return data
    reactivity = Reactivity_fitting(data)
    dictionary = dict(zip(pvalue.uniq_list, pvalue.num_to_pvalue_dict(score_ind)))
    assert len(list(dictionary.keys())) == len(pvalue.uniq_list)
    return dict_to_real_value(data, dictionary, score_ind)
