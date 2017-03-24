# from __init__ import *
import sys
import scipy.linalg
from scipy.stats import rankdata
from scipy.optimize import fminbound, minimize
import numpy as np
import copy
import math
from enum import Enum
from prob_optimize import *
from utility import *
from idr_wrapper import common_transcript, only_build_rank_vectors, get_concatenated_scores, get_idr_value
# import idr.idr
# import pdb; pdb.set_trace()

# import random
# import subprocess
# import os
# import math
# from statsmodels.tsa.stattools import acf
# import pvalue
# import pylab
# from idr_wrapper import *
# from plot_image import *

def print_mat_one_line(mat, end=''):
    print([[mat[i,j] for j in range(mat.shape[1])] for i in range(mat.shape[0])], end=end)

def dot_blacket_to_float(c):
    if c == ".":    return 0.0
    elif c == "(" or c == ")":  return 1.0
    else:   return 0.5

def get_struct_dict(file):
    struct_dict = {}
    with open(file) as f:
        name = "", ""
        for line in f.readlines():
            if line[0] == '>':
                name = line[1:].rstrip('\n')
            else:
                struct_dict[name] = [dot_blacket_to_float(c) for c in line.rstrip('\n')]
    return struct_dict

class Parameter:
    def __init__(self, arg):
        self.mu = arg.mu
        self.sigma = arg.sigma
        self.rho = arg.rho
        self.pi = arg.pi

class ForwardBackward:
    """docstring for ForwardBackward"""
    def __init__(self, length, dim):
        self.INF = float('infinity')
        self.forward = np.matrix([[-self.INF]*dim]*(length+1), dtype=float)
        self.backward = np.matrix([[-self.INF]*dim]*(length+1), dtype=float)
        self.responsibility = np.matrix([[-self.INF]*(dim**2)]*(length+1), dtype=float)
        self.emission = np.matrix([[-self.INF]*dim]*(length+1), dtype=float)

    def fill_inf(self):
        self.forward.fill(-self.INF)
        self.backward.fill(-self.INF)

    def pf(self):
        return self.forward[self.forward.shape[0]-1,0]

class Hidden(Enum):
    unmappable = 0
    acc = 1
    stem = 2

class Sample(Enum):
    case = 0
    cont = 1

def set_hmm_transcripts(data, sample_size, max_len, skip_start, skip_end, debug = True):
    if len(data) == 1:
        rankdata = [[]]
        seta = data[0][0].keys()
    else:
        rankdata = [[], []]
        seta = common_transcript(data[0][0], data[1][0])
    if sample_size > 0:
        seta = random.sample(seta, min(len(seta), sample_size))
    if debug: # debug mode
        seta = set([chr(0), "RNA18S5+", "RNA28S5+", "RNA18S5-", "RNA28S5-"])
    seta = sorted(list(seta))
    for i in range(len(data)):
        for x in seta:
            rep1, rep2 = data[i][0][x], data[i][1][x]
            if max_len > 0:
                rep1, rep2 = rep1[0:max_len], rep2[0:max_len]
            else:
                if skip_start > 0:  rep1, rep2 = rep1[skip_start:], rep2[skip_start:]
                if skip_end > 0:    rep1, rep2 = rep1[0:len(rep1)-skip_end], rep2[0:len(rep2)-skip_end]
            data[i][0][x], data[i][1][x] = rep1, rep2
    for i, x in enumerate(data):
        s1, s2 = get_concatenated_scores(seta, x[0], x[1])
        rankdata[i] = only_build_rank_vectors(s1, s2)
    temp = [len(data[0][0][x]) for x in seta]
    print("length:", temp)
    stop_sites = [sum(temp[0:(i+1)])-1 for i in range(len(temp))]
    print("unmappable:", stop_sites)
    return rankdata, stop_sites, seta

class HMM:
    """docstring for ClassName"""
    def __init__(self, hclass, v, stop_sites):
        self.hclass = hclass
        self.v = v
        self.stop_sites = stop_sites
        self.fb = None
        self.pseudo_value = None
        self.transition_param = None
        self.params = []
        self.verbose = False
        self.resp_file = 'responsibility.txt'
        self.f_file = 'forward.txt'
        self.b_file = 'backward.txt'

    def set_IDR_params(self, index, params):
        if self.verbose:    print("HMM: change params.")
        if index < 0:
            self.params = params.copy()
        else:
            self.params[index] = params.copy()

    def get_IDR_params(self, index = -1):
        if index < 0:
            return self.params
        return self.params[index]

    def set_transition_param_log(self, raw_transition_param):
        if self.verbose:    print("HMM: change transition params.")
        self.transition_param = raw_transition_param.copy()
        for i in range(self.transition_param.shape[0]):
            for j in range(self.transition_param.shape[1]):
                if self.transition_param[i, j] == 0.0:
                    self.transition_param[i, j] = -float('infinity')
                else:
                    self.transition_param[i, j] = min(0.0, math.log(self.transition_param[i, j]))

    def set_pseudo_value(self, index = -1):
        if self.verbose:    print("HMM: set pseudo_value (%i)." % index)
        if index < 0:
            self.pseudo_value = [[] for i in range(len(self.v))]
        else:
            self.pseudo_value[index] = []
        for i in range(len(self.v)):
            if index < 0 or index == i:
                param = self.get_IDR_params(i)
                self.pseudo_value[i] = [hmm_compute_pseudo_values(self.v[i][0], param[0], param[1], param[3]),
                                        hmm_compute_pseudo_values(self.v[i][1], param[0], param[1], param[3])]
                if self.verbose:
                    for j in range(len(self.pseudo_value[i])):
                        print("HMM: set pseudo value: ", i, j, "->", self.pseudo_value[i][j][0:10], "...")
                    for j in range(len(self.v[i])):
                        print("HMM: (original value) ", i, j, "->", self.v[i][j][0:10], "...")

    def responsibility_state(self, index, h, debug = False):
        return sum([self.fb.responsibility[index, h*self.hclass+k] for k in range(self.hclass)])

    def responsibility_transition(self, index, h, k):
        return self.fb.responsibility[index, h*self.hclass+k]

    def set_responsibility(self):
        if self.verbose:    print("HMM: set responsibility")
        for i in range(self.length-1):
            for h in range(self.hclass):
                for k in range(self.hclass):
                    prob = self.fb.forward[i, h]+self.transition_param[h, k]+self.fb.backward[i+1, k]-self.fb.pf()
                    if abs(prob) == float('infinity'):
                        self.fb.responsibility[i, h*self.hclass+k] = 0.0
                    else:
                        self.fb.responsibility[i, h*self.hclass+k] = exp(prob)
        if self.verbose:
            print("HMM: write responsibility to file...")
            np.savetxt(self.resp_file, self.fb.responsibility)

    def emission_prob_log_value(self, rep, x1, x2, param):
        deno = logS(x1, x2, param)
        if rep:
            return fastpo.c_calc_gaussian_lhd(x1, x2, param)-deno
        else:
            return fastpo.c_calc_gaussian_lhd(x1, x2, [0., 1., 0., 0.])-deno

    def emission_prob_log(self, index, h, params):
        def rep(i, h):
            return ((i == Sample.case.value and h == Hidden.acc.value) or (i == Sample.cont.value and h == Hidden.stem.value))
        assert index > 0
        return sum([self.emission_prob_log_value(rep(i, h), self.pseudo_value[i][0][index], self.pseudo_value[i][1][index], params[i]) for i in range(len(self.pseudo_value)) ])

    def set_emission(self):
        if self.verbose:    print("HMM: set emission")
        for i in range(1, self.length-1):
            for h in range(self.hclass):
                self.fb.emission[i, h] = self.emission_prob_log(i, h, self.params)

    def debug_print_q_function(self, sindex, params):
        first = 0.0
        for h in range(self.hclass):
            for i in range(1, self.length-1):
                value = self.responsibility_state(i, h)*self.emission_prob_log(i, h, params)
                print(value, end=" ")
        print('')
        second = sum([self.responsibility_state(i, h)*self.emission_prob_log(i, h, params) for i in range(1, self.length-1) for h in range(self.hclass) if self.responsibility_state(i, h) > 0.0])
        third = sum([self.responsibility_transition(i, h, k)*self.transition_param[h, k] for k in range(self.hclass) for h in range(self.hclass) for i in range(0, self.length-1) if self.responsibility_transition(i, h, k) > 0.0])
        q = sum((first, second, third))
        print('q ', q)
        sys.exit(1)

    def q_function(self, sindex = -1, param = None):
        if sindex >= 0:
            self.set_IDR_params(sindex, param)
            self.set_pseudo_value(sindex)
        params = self.get_IDR_params()
        first = 0.
        if sindex >= 0:
            second = sum([sum([ self.responsibility_state(i, h, True)*self.emission_prob_log(i, h, self.params) for i in range(1, self.length-1) if self.responsibility_state(i, h, True) > 0]) for h in range(self.hclass)])
        else:
            second = sum([sum([ self.responsibility_state(i, h, True)*self.fb.emission[i, h] for i in range(1, self.length-1) if self.responsibility_state(i, h, True) > 0]) for h in range(self.hclass)])
        third = 0.
        for i in range(0, self.length-1):
            for h in range(self.hclass):
                third += sum([ self.responsibility_transition(i, h, k)*self.transition_param[h, k] for k in range(self.hclass) if self.responsibility_transition(i, h, k) > 0.0])
        q = first+second+third
        if self.verbose:
            print('q_function', first+second+third, "=", first, second, third)
        return first+second+third

    def q_function_const(self, sindex = -1, param = None):
        if sindex >= 0:
            original_params = self.get_IDR_params().copy()
            self.set_IDR_params(sindex, param)
            self.set_pseudo_value(sindex)
        params = self.get_IDR_params().copy()
        if self.verbose:
            print("HMM: calc q_function (%s)." % ['new aparam', 'new sparam', 'no renewal'][sindex], end="")
            print(params)
        q = self.q_function()
        if sindex >= 0:
            self.set_IDR_params(sindex, original_params[sindex])
            self.set_pseudo_value(sindex)
        return q

    def calc_q_function_grad_at_once(self, s, rep_class):
        grad = [0.]*3
        for i, (x, y) in enumerate(zip(self.pseudo_value[s][0], self.pseudo_value[s][1])):
            if i == 0:  continue
            q = QfuncGrad(x, y, self.get_IDR_params(s))
            for h in range(self.hclass):
                if self.responsibility_state(i, h, True) == 0.0:  continue
                r = 0 if h == rep_class else 1
                for p in range(3):
                    grad[p] += self.responsibility_state(i, h)*q.grad_list_log[r][p]
        return grad

    def q_function_grad(self, grad_max = 3):
        grad = []
        for s in range(len(self.v)):
            if s == 0:
                rep_class = 1 # acc == reproducible
            elif s == 1:
                rep_class = 2 # stem == reproducible
            grad.append(self.calc_q_function_grad_at_once(s, rep_class))
        return grad

    def q_function_discrete(self, sindex, pindex, pthres, mthres):
        if pindex == 2 or pindex == 3:
            mthres = max(0.0, mthres)
            pthres = min(1.0, pthres)
        if pindex == 1:
            mthres = max(EPS, mthres)
        params = self.get_IDR_params(sindex).copy()
        params[pindex] = pthres
        qp = self.q_function_const(sindex, params)
        plus = params[pindex]
        params[pindex] = mthres
        qm = self.q_function_const(sindex, params)
        minus = params[pindex]
        if self.verbose:
            print('disc=', (qp-qm)/(plus-minus), '(', qp, '-', qm, ')/', plus-minus, plus, pthres, minus, mthres)
        return qp, qm, plus-minus

    def q_function_grad_num(self, grad_max = 3):
        grad = []
        alpha = 0.001
        for s in range(len(self.v)):
            rep_class = 1 if s == 0 else 2 # acc == reproducible or stem == reproducible
            grad.append([])
            for i, p in enumerate(self.get_IDR_params(s)):
                if i == grad_max:  break
                pthres, mthres = p+alpha, p-alpha
                qp, qm, diff = self.q_function_discrete(s, i, pthres, mthres)
                grad[s].append((qp-qm)/diff)
                if self.verbose:
                    print('HMM:', i, "gradient", qp-qm, qp, qm)
        return grad

    def q_function_emi_grad_num(self, x1, x2, params, rep, grad_max = 3, log = True):
        grad = []
        alpha = 0.001
        if len(params) != 4:
            params = params[0]
        for i, p in enumerate(params):
            if i == grad_max:  break
            pthres, mthres = p+alpha, p-alpha
            if i == 2 or i == 3:
                mthres = max(0.0, mthres)
                pthres = min(1.0, pthres)
            if i == 1:
                mthres = max(EPS, mthres)
            tparams = params.copy()
            tparams[i] = pthres
            qp = self.emission_prob_log_value(rep, x1, x2, tparams)
            plus = tparams[i]
            tparams[i] = mthres
            qm = self.emission_prob_log_value(rep, x1, x2, tparams)
            minus = tparams[i]
            if not log:
                qp, qm = exp(qp), exp(qm)
            grad.append((qp-qm)/(plus-minus))
            if self.verbose:
                print('HMM:', i, "emi_gradient", (qp-qm)/(plus-minus), qp, qm, plus, minus)
        return grad

    def q_function_comp_grad_num(self, x1, x2, params, rep, grad_max = 3, log = True):
        grad = []
        alpha = 0.001
        if len(params) != 4:
            params = params[0]
        for i, p in enumerate(params):
            if i == grad_max:  break
            pthres, mthres = p+alpha, p-alpha
            if i == 2 or i == 3:
                mthres = max(0.0, mthres)
                pthres = min(1.0, pthres)
            if i == 1:
                mthres = max(EPS, mthres)
            tparams = params.copy()
            tparams[i] = pthres
            if rep:
                qp = logR(x1, x2, tparams)-logS(x1, x2, tparams)
            else:
                qp = logI(x1, x2, tparams)-logS(x1, x2, tparams)
            plus = tparams[i]
            tparams[i] = mthres
            if rep:
                qm = logR(x1, x2, tparams)-logS(x1, x2, tparams)
            else:
                qm = logI(x1, x2, tparams)-logS(x1, x2, tparams)
            minus = tparams[i]
            if not log:
                qp, qm = exp(qp), exp(qm)
            grad.append((qp-qm)/(plus-minus))
            if self.verbose:
                print('HMM:', i, "comp_gradient", (qp-qm)/(plus-minus), qp, qm, plus, minus)
        return grad


    def fill_fb(self):
        self.set_emission()
        self.fb.forward[0, 0] = 0.
        self.fb.backward[self.length-1,0] = 0.
        if self.verbose:
            print('HMM: start forward and backward')
        for i in range(1, self.length-1):
            if i in self.stop_sites:
                self.fb.forward[i, 0] = logsumexp_inf([self.fb.forward[i-1, k]
                                        +self.transition_param[k, 0]
                                        +self.fb.emission[i, 0] for k in range(self.hclass)])
            else:
                self.fb.forward[i, :] = [logsumexp_inf([self.fb.forward[i-1, k]
                                                  +self.transition_param[k, h]
                                                  +self.fb.emission[i, h] for k in range(self.hclass)])
                                                                               for h in range(self.hclass)]
        self.fb.forward[self.length-1, 0] = logsumexp_inf([self.fb.forward[self.length-2, k]+self.transition_param[k, 0] for k in range(self.hclass)])
        for i in range(self.length-2, 0, -1): # transition_param is used as transposed.
            if i in self.stop_sites:
                self.fb.backward[i, 0] = logsumexp_inf([self.fb.backward[i+1, k]
                                             +self.transition_param[0, k]
                                             +self.fb.emission[i, 0] for k in range(self.hclass)])
            else:
                self.fb.backward[i, :] = [logsumexp_inf([self.fb.backward[i+1, k]
                                                   +self.transition_param[h, k]
                                                   +self.fb.emission[i, h] for k in range(self.hclass)])
                                                                                              for h in range(self.hclass)]
                # print([[(h, k, self.fb.backward[i+1, k], self.transition_param[h, k], self.emission_prob_log(i, h, self.params)) for k in range(self.hclass)] for h in range(self.hclass)])
        self.fb.backward[0, 0] = logsumexp_inf([self.fb.backward[1, k]+self.transition_param[0, k] for k in range(self.hclass)])
        if self.verbose:
            print('HMM: done forward and backward -> ', (self.fb.pf(), self.fb.backward[0, 0]))
            self.print_result()
        assert abs(self.fb.pf()-self.fb.backward[0, 0]) < EPS
        self.set_responsibility()

        """
        state and output location
                0           1          ...     length-2     length-1
        state   unmappable  any        ...     unmappable   unmappable
        output  u1          u2         ...     uN        none
        stop_sites = [0, n1, n2, ..., length-2] (end of each transcript should be unmappable)

        """

    def check_prob(self):
        for i in range(1, self.length-1):
            temp = [self.responsibility_state(i, h) for h in range(self.hclass)]
            assert abs(sum(temp)-1.0) < EPS
        for i in range(1, self.length-1):
            temp = [[self.responsibility_transition(i, h, k) for k in range(self.hclass)] for h in range(self.hclass)]
            # print(sum([sum(x) for x in temp]))
            assert abs(sum([sum(x) for x in temp])-1.0) < EPS

    def debug_print(self):
        self.check_prob()
        # print(self.rankdata)
        # print(self.fb)

    def print_result(self):
        print('HMM: write fb to file ...')
        np.set_printoptions(threshold=np.nan)
        np.savetxt(self.f_file, self.fb.forward)
        np.savetxt(self.b_file, self.fb.backward)

    def forward_backward(self, transition_param, params):
        if self.verbose:    print("HMM: forward backward.")
        self.set_IDR_params(-1, params)
        self.set_transition_param_log(transition_param)
        self.set_pseudo_value()
        if self.fb == None or self.fb.forward.shape[0] != len(self.pseudo_value[0][0]):
            self.fb = ForwardBackward(len(self.pseudo_value[0][0]), self.hclass)
            self.length = self.fb.forward.shape[0]
        else:
            self.fb.store_old()
            self.fb.fill_inf()
        self.fill_fb()
        self.debug_print()
        return self.q_function()


class ParamFitHMM:
    """docstring for ParamFitHMM"""
    def __init__(self, hclass, data, sample = -1, param=None, debug = False, idr_output = 'idr_output.csv', ref = '', start = -1, end = 35, max_len = -1):
        self.hclass = hclass
        # self.fb = None
        assert hclass == 2 or hclass == 3
        if self.hclass == 2:
            neutral = '0.7 0.3; 0.3 0.7'
            drastic = '0.95 0.05; 0.8 0.2'
            self.init_transition_param = np.matrix(neutral, dtype=float)
            self.transition_param = self.init_transition_param.copy()
        else:
            neutral = '0.5 0.25 0.25; 0.25 0.5 0.25; 0.25 0.25 0.5'
            drastic = '0.9 0.05 0.05; 0.7 0.2 0.1; 0.7 0.1 0.2'
            self.init_transition_param = np.matrix(neutral, dtype=float)
            self.transition_param = self.init_transition_param.copy()
        if type(param) == type(None):
            param = (1, 0.2, 0.8, 0.2) # mu, sigma, rho, pi
        self.params = []
        for i in range(hclass-1):
            self.params.append(param.copy())
        self.skip_start, self.skip_end, self.max_len = -1, 35, -1 # Truncate from start and end for each transcript.
        self.v, self.stop_sites, self.keys = set_hmm_transcripts(data, sample, self.max_len, self.skip_start, self.skip_end, debug)
        self.length = len(self.v[0][0])
        self.HMM = HMM(hclass, self.v, self.stop_sites)
        self.max_len_trans = 10000
        self.verbose = True
        self.idr_output = idr_output
        self.ref = ref

    def set_IDR_params(self, index, theta):
        self.params[index] = theta.copy()

    def set_IDR_param(self, index, pos, theta):
        self.params[index][pos] = theta

    def get_IDR_params(self, index = -1):
        if index < 0:
            return self.params
        return self.params[index].copy()

    def get_pseudo_value(self, index):
        assert index+1 < self.hclass
        return self.HMM.pseudo_value[index][0], self.HMM.pseudo_value[index][1]

    def normalized_prob(self, vec):
        vec = np.asarray(vec).reshape(-1)
        count = [np.real(i) if i > 0.0 else EPS for i in vec]
        return [c/float(sum(count)) for c in count]

    def set_new_transition(self):
        for h in range(self.hclass):
            count = [sum([self.HMM.responsibility_transition(i, h, k) for i in range(0, self.length-1)]) for k in range(self.hclass)]
            self.transition_param[h,:] = self.normalized_prob(count)
        if self.verbose:
            print('set new_transition -> ', end='')
            print_mat_one_line(self.transition_param, '\n')

    def set_new_p(self):
        value, vector = scipy.linalg.eig(self.transition_param)
        vector = np.matrix(vector)
        N = self.length-1
        if self.verbose:
            np.set_printoptions(linewidth=200)
            print('computed eigenvector', value, end="")
            print_mat_one_line(vector, '\n')
        index = [i for i in range(len(value)) if abs(value[i]-1.0) < EPS]
        if len(index) == 0:
            print(self.transition_param)
            sys.exit('Transition probability matrix error!')
        if N > self.max_len_trans:
            vector = self.normalized_prob(vector[:,index[0]])
            for i in range(len(self.params)):
                self.set_IDR_param(i, 3, vector[i+1])
        else:
            u, u_inv = vector, np.linalg.inv(vector)
            An = self.transition_param[0,]
            for n in range(N):
                Dn = np.diag(value**n)
                An = An+(u*Dn*u_inv)[0]
            An = self.normalized_prob(An)
            for i in range(self.hclass-1):
                self.set_IDR_param(i, 3, An[i+1])
        if self.verbose:
            print('set new_p -> ', self.params)
        self.HMM.set_pseudo_value(-1)

    def EM_CA_step(self, sindex, theta, index, min_val, max_val):
        def f(alpha):
            inner_theta = theta.copy()
            inner_theta[index] = theta[index] + alpha
            q = -self.HMM.q_function_const(sindex, inner_theta)
            # print("alpha ->", index, alpha, q, inner_theta, theta)
            return q
        min_step_size = min_val - theta[index]
        max_step_size = max_val - theta[index]
        assert theta[index] >= min_val and theta[index] <= max_val
        alpha = fminbound(f, min_step_size, max_step_size)
        prev_lhd, new_lhd = -f(0), -f(alpha)
        print("CA step -> ", new_lhd, prev_lhd, alpha, theta)
        return alpha, new_lhd

    def EM_CA_iteration(self, sindex, prev_theta, prev_lhd):
        min_vals = [MIN_MU, MIN_SIGMA, MIN_RHO] #MIN_MIX_PARAM
        max_vals = [MAX_MU, MAX_SIGMA, MAX_RHO] #MAX_MIX_PARAM
        update_amount = [0., 0., 0., 0.]
        new_lhd = [0., 0., 0., 0.]
        for index, (min_val, max_val) in enumerate(zip(min_vals, max_vals)):
            theta = prev_theta.copy()
            alpha, lhd = self.EM_CA_step(sindex, theta, index, min_val, max_val)
            update_amount[index] = alpha
            new_lhd[index] = lhd
        return update_amount, new_lhd

    def EM_iteration(self, sindex, init_theta, init_lhd, fix_mu, fix_sigma, eps, grad = True, alpha = 1e-6):
        prev_theta, prev_lhd = init_theta.copy(), init_lhd
        prev_theta, _        = clip_model_params(prev_theta)
        min_vals = [MIN_MU, MIN_SIGMA, MIN_RHO] #MIN_MIX_PARAM
        max_vals = [MAX_MU, MAX_SIGMA, MAX_RHO] #MAX_MIX_PARAM
        new_lhd = [0., 0., 0., 0.]
        if grad:
            update_amount = self.HMM.q_function_grad()[sindex]
        else:
            update_amount, new_lhd = self.EM_CA_iteration(sindex, prev_theta, prev_lhd)
        for index, (min_val, max_val) in enumerate(zip(min_vals, max_vals)):
            if index == 0 and fix_mu: continue
            if index == 1 and fix_sigma: continue
            theta = prev_theta.copy()
            # print(theta, alpha, update_amount)
            theta[index] = theta[index]+alpha*(min(max(min_vals[index], update_amount[index]), max_vals[index]))
            if grad or new_lhd[index] + eps >= prev_lhd:
                if not grad and self.verbose:
                   print(['mu', 'sigma', 'rho', 'q'][index], '('+str(new_lhd[index])+'-'+str(prev_lhd)+')', prev_theta[index], '->', theta[index])
                theta, changed_params = clip_model_params(theta)
                prev_theta = theta
        sys.stdout.flush()
        return prev_theta, max(new_lhd)

    def print_dataset_for_each_sample(self, index, IDR, head):
        flag = 'a'
        with open(self.idr_output, flag) as f:
            for i in range(0, len(self.stop_sites)-1):
                start, end = self.stop_sites[i]+1, self.stop_sites[i+1]+1
                tIDR = IDR[start:end]
                if self.skip_start > 0:
                    tIDR = np.append([float('nan')]*self.skip_start, tIDR)
                if self.skip_end > 0:
                    tIDR = np.append(tIDR, [float('nan')]*self.skip_end)
                f.write(head+"\t"+self.keys[i+1]+"\t"+['cond', 'case'][index]+"\t"+":".join([("%.4e" % x) for x in tIDR])+"\n")

    def print_header_for_each_sample(self, index, data, head):
        if index == 0:
            flag = 'w'
        else:
            flag = 'a'
        with open(self.idr_output, flag) as f:
            for i, temp in enumerate(data):
                f.write(head+"\t"+self.keys[i+1]+"\t"+['cond', 'case'][index]+"\t"+":".join(list(map(str, temp)))+"\n")

    def print_reference_for_each_sample(self, index, ref, head, key):
        flag = 'a'
        with open(self.idr_output, flag) as f:
            f.write(head+"\t"+key+"\t"+['cond', 'case'][index]+"\t"+":".join([("%.4e" % x) for x in ref])+"\n")

    def write_header_to_file(self):
        for i in range(len(self.v)):
            param = self.get_IDR_params(i)
            offset = max(0, self.skip_start)+max(0, self.skip_end)
            position = [list(range(0, self.stop_sites[i+1]-self.stop_sites[i]+offset)) for i in range(0, len(self.stop_sites)-1)]
            self.print_header_for_each_sample(i, position, "type")

    def write_idr_value_to_file(self):
        for i in range(len(self.v)):
            param = self.get_IDR_params(i)
            z1, z2 = self.v[i]
            localIDRs, IDR = get_idr_value(z1, z2, param)
            self.print_dataset_for_each_sample(i, IDR, "IDR")

    def write_responsibility_to_file(self, head):
        for i in range(len(self.v)):
            param = self.get_IDR_params(i)
            IDR = [1.-self.HMM.responsibility_state(x, i+1) for x in range(self.length)]
            self.print_dataset_for_each_sample(i, IDR, head)

    def visualize_IDR_file(self):
        if len(self.ref) > 0:
            struct_dict = get_struct_dict(self.ref)
            for key in struct_dict.keys():
                self.print_reference_for_sample(0, struct_dict[key], "ref", key)


    def set_init_theta(self):
        for i in range(len(self.v)):
            gtheta = hmm_grid_search(self.v[i][0], self.v[i][1])
            lhd = log_lhd_loss(self.v[i][0], self.v[i][1], gtheta)
            tlhd = log_lhd_loss(self.v[i][0], self.v[i][1], self.get_IDR_params(i))
            print("# Grid search: ", gtheta, lhd, tlhd)
            if tlhd < lhd:
                self.set_IDR_params(i, list(gtheta))

    def apply_forward_backward(self, transition_param = None, params = None):
        if type(transition_param) == type(None):
            transition_param = self.transition_param
        if type(params) == type(None):
            params = self.get_IDR_params()
        return self.HMM.forward_backward(transition_param, params)

    def print_result(self):
        if self.verbose:
            print("HMM: print result.")
            self.HMM.print_result()

    def check_value_change(self, iindex, index, prev_theta, theta, new_lhd):
        prev_z1, prev_z2 = self.get_pseudo_value(index)
        self.set_IDR_params(index, theta)
        self.HMM.set_pseudo_value(index)
        z1, z2 = self.get_pseudo_value(index)
        mean_pseudo_val_change = np.mean([np.abs(p-z) for p, z in zip(prev_z1, z1)]) + np.mean([np.abs(p-z) for p, z in zip(prev_z2, z2)])
        sum_param_change = sum([np.abs(x-y) for x, y in zip(theta, prev_theta) ])#np.abs(theta - prev_theta).sum()
        print(("Iter %i" % iindex).ljust(12), ("Dataset %i" % index),
            "%.2e" % sum_param_change,
            "%.2e" % mean_pseudo_val_change,
            "%.4e" % new_lhd,
            theta, )
        return sum_param_change, mean_pseudo_val_change

    def hmm_EMP_with_pseudo_value_algorithm_test(self, grid = False, N = 100, EPS = 1e-4, fix_mu = False, fix_sigma = False):
        self.hmm_EMP_with_pseudo_value_algorithm(grid, N, EPS, fix_mu, fix_sigma, True)

    def hmm_EMP_with_pseudo_value_algorithm(self, grid = False, N = 100, EPS = 1e-4, fix_mu = False, fix_sigma = False, test = False):
        if grid:
            self.set_init_theta()
        lhd = self.apply_forward_backward()
        print('(initial) new lhd ->\t%f ' % (lhd), end='\t')
        print(0, -1, self.get_IDR_params(), end=' ', sep='\t')
        print_mat_one_line(self.transition_param, '\n')
        space = np.linspace(-1., np.log10(EPS), N)
        self.write_header_to_file()
        self.write_idr_value_to_file()
        for i in range(N):
            break_flag = True
            self.set_new_transition()
            self.set_new_p()
            lhd = self.apply_forward_backward()
            print('(%d time) new lhd ->\t%f' % (i, lhd), end='\t')
            print(i, -1, self.get_IDR_params(), end='\t', sep='\t')
            print_mat_one_line(self.transition_param, '\n')
            for j in range(len(self.v)):
                prev_theta = self.get_IDR_params(j)
                theta, new_lhd = self.EM_iteration(j, prev_theta, lhd, fix_mu=fix_mu, fix_sigma=fix_sigma, eps=EPS/10., grad=True, alpha=10**space[i])
                sum_param_change, mean_pseudo_val_change = self.check_value_change(i, j, prev_theta, theta, new_lhd)
                if not (i > 5 and (sum_param_change < EPS and mean_pseudo_val_change < EPS)):
                    break_flag = False
            if test:
                # print('(%d-th dataset) new lhd -> %f ' % (j, new_lhd))
                self.write_responsibility_to_file("IDR-HMM-"+str(i))
            if break_flag or i == N-1:
                self.print_result()
                self.write_responsibility_to_file("IDR-HMM-final")
                break

