import sys
from multiprocessing import Pool
import numpy as np
import copy
import math
from enum import Enum
from reactIDR.prob_optimize import *

def is_rep(h, rep_class):
    return 0 if h == rep_class else 1


class Parameter:
    def __init__(self, arg):
        self.mu = arg.mu
        self.sigma = arg.sigma
        self.rho = arg.rho
        self.pi = arg.pi

class ForwardBackward:
    """docstring for ForwardBackward"""
    def __init__(self, length, dim, hidden=None):
        self.INF = float('infinity')
        self.forward = np.matrix([[-self.INF]*dim]*(length+1), dtype=float)
        self.backward = np.matrix([[-self.INF]*dim]*(length+1), dtype=float)
        self.responsibility = np.matrix([[-self.INF]*(dim**2)]*(length+1), dtype=float)
        self.emission = np.matrix([[-self.INF]*dim]*(length+1), dtype=float)
        if hidden is None: # -1 any. 0,1,2 -> predefined hidden class.
            self.hidden = -np.ones(shape=(length+1), dtype=np.int8)
        else:
            assert hidden.shape[0] == length+1, str(hidden.shape[0])+' != '+str(length+1)
            self.hidden = hidden

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

class HMM:
    """docstring for ClassName"""
    unmap_prior = [0.0, -float('inf'), -float('inf')]
    EPS = 1e-6
    def __init__(self, hclass, v, stop_sites, hidden, thread, independent = False):
        self.bases = "ACGTU"
        self.hclass = hclass
        self.v = v
        self.stop_sites = stop_sites
        self.hidden_class = hidden
        self.fb = None
        self.pseudo_value = None
        self.transition_param = None
        self.params = []
        self.verbose = False
        if independent:
            verbose = True
        self.mp = 1
        self.resp_file = 'responsibility.txt'
        self.f_file = 'forward.txt'
        self.b_file = 'backward.txt'
        self.thread = thread
        if thread > 1:
            self.qfuncgrad = self.qfuncgrad_multi_core
            self.qfuncgrad_resp = self.qfuncgrad_resp_single_variable
            self.compute_pseudo_values = self.comp_pseudo_multi_core

        else:
            self.thread = 1 # assure thread number >= 1
            self.qfuncgrad = self.qfuncgrad_single_core
            self.qfuncgrad_resp = self.qfuncgrad_resp_single_variable
            self.compute_pseudo_values = self.comp_pseudo_single_core
        self.pool = None
        self.independent = (independent and len(self.stop_sites) > 2) # transcript
        self.independent_transition = [] # store transition responsibility between unmap (no emission) and the first base for each transcript.

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
                self.pseudo_value[i] = self.compute_pseudo_values(i)
                if self.verbose:
                    for j in range(len(self.pseudo_value[i])):
                        print("HMM: set pseudo value: ", i, j, "->", self.pseudo_value[i][j][0:10], "...")
                    for j in range(len(self.v[i])):
                        print("HMM: (original value) ", i, j, "->", self.v[i][j][0:10], "...")

    def comp_pseudo_multi_core(self, i):
        param = self.get_IDR_params(i)
        self.pool = Pool(self.thread)
        pseudo = self.pool.starmap(hmm_compute_pseudo_values, [(x, param[0], param[1], param[3]) for x in self.v[i]])
        self.pool.close()
        return pseudo

    def comp_pseudo_single_core(self, i):
        param = self.get_IDR_params(i)
        return [hmm_compute_pseudo_values(x, param[0], param[1], param[3]) for x in self.v[i]]

    def qfuncgrad_multi_core(self, params, values):
        self.pool = Pool(self.thread)
        qlist = [None]+self.pool.starmap(QfuncGrad23dim, [(params, variable) for i, variable in enumerate(zip(*values)) if i != 0])
        self.pool.close()
        return qlist

    def qfuncgrad_single_core(self, params, values):
        return [None]+[QfuncGrad23dim(params, variable) for i, variable in enumerate(zip(*values)) if i != 0]

    def qfuncgrad_resp_single_variable(self, qlist, rep_class, p):
        grad = 0.0
        cond = self.q_function_state_cond
        for h in range(self.hclass):
            rep_flag = is_rep(h, rep_class)
            grad += sum([self.responsibility_state(i, h)*q.grad_list_log[rep_flag][p] for i, q in enumerate(qlist) if cond(i, h)])
        return grad

    def qfuncgrad_resp_single_core(self, qlist, rep_class):
        grad = []
        cond = self.q_function_state_cond
        for p in range(3):
            temp = 0.0
            for h in range(self.hclass):
                rep_flag = is_rep(h, rep_class)
                temp += sum([self.responsibility_state(i, h)*q.grad_list_log[rep_flag][p] for i, q in enumerate(qlist) if cond(i, h)])
            grad.append(temp)
        return grad

    def responsibility_state(self, index, h):
        return sum([self.fb.responsibility[index, h*self.hclass+k] for k in range(self.hclass)])

    def responsibility_transition(self, index, h, k):
        return self.fb.responsibility[index, h*self.hclass+k]

    def set_responsibility(self):
        if self.verbose:    print("HMM: set responsibility")
        for i in range(self.length-1):
            for h in range(self.hclass):
                for k in range(self.hclass):
                    prob = self.fb.forward[i, h]+self.transition_param[h, k]+self.fb.backward[i+1, k]-self.fb.pf()
                    # print(i, h, k, prob, self.fb.forward[i, h], self.transition_param[h, k], self.fb.backward[i+1, k], self.fb.pf())
                    if abs(prob) == float('infinity'):
                        self.fb.responsibility[i, h*self.hclass+k] = 0.0
                    else:
                        self.fb.responsibility[i, h*self.hclass+k] = exp(prob)

    def set_responsibility_independent(self):
        if self.verbose:    print("HMM: set responsibility independently")
        stop_sites = copy.deepcopy(self.stop_sites)+[self.length-1] # Add the last unmappable.
        unmap_prior = (0., float('-inf'), float('-inf'))
        for i in range(self.length-1):
            if i > stop_sites[0]:
                stop_sites.pop(0)
            pf = self.fb.forward[stop_sites[0], 0] if i != stop_sites[0] else self.fb.forward[stop_sites[1], 0]
            self.fb.responsibility[i, :] = 0.0
            h_list, k_list = range(self.hclass), range(self.hclass)
            for h in h_list:
                for k in k_list:
                    forward = self.fb.forward[i, h] if i != stop_sites[0] else unmap_prior[h]
                    backward = self.fb.backward[i+1, k]
                    prob = forward+self.transition_param[h, k]+backward-pf
                    if abs(prob) == float('inf'):
                        self.fb.responsibility[i, h*self.hclass+k] = 0.0
                    else:
                        self.fb.responsibility[i, h*self.hclass+k] = exp(prob)
            # stop_sites position -> responsibility for a hidden unmappable nucleotide and the next.

    def emission_prob_log_value(self, rep, param, x1, x2, x3=None):
        if x3 is None:
            deno = logS(x1, x2, param)
            if rep:
                return fast_param_fit.c_calc_gaussian_lhd(x1, x2, param)-deno
            else:
                return fast_param_fit.c_calc_gaussian_lhd(x1, x2, [0., 1., 0., 0.])-deno
        else:
            deno = logS_3dim(x1, x2, x3, param)
            if rep:
                return fast_param_fit.c_calc_gaussian_lhd_3dim(x1, x2, x3, param)-deno
            else:
                return fast_param_fit.c_calc_gaussian_lhd_3dim(x1, x2, x3, [0., 1., 0., 0.])-deno

    def emission_prob_log_value_ris(self, rep, params, x1, x2, x3=None):
        if x3 is None:
            if rep:
                return logR(x1, x2, params)-logS(x1, x2, params)
            else:
                return logI(x1, x2, params)-logS(x1, x2, params)
        else:
            if rep:
                return logR_3dim(x1, x2, x3, params)-logS_3dim(x1, x2, x3, params)
            else:
                return logI_3dim(x1, x2, x3, params)-logS_3dim(x1, x2, x3, params)

    def emission_prob_log(self, index, h, params):
        def rep(i, h):
            return ((i == Sample.case.value and h == Hidden.acc.value) or (i == Sample.cont.value and h == Hidden.stem.value))
        assert index > 0
        return sum([self.emission_prob_log_value(rep(i, h), params[i], *[self.pseudo_value[i][j][index] for j in range(len(self.pseudo_value[i]))]) for i in range(len(self.pseudo_value)) ])

    def set_emission(self):
        if self.verbose:    print("HMM: set emission")
        for i in range(1, self.length-1):
            for h in range(self.hclass):
                self.fb.emission[i, h] = self.emission_prob_log(i, h, self.params)

    def q_function_state_cond(self, i, h):
        return i > 0 and self.responsibility_state(i, h) > 0

    def q_function_state_cond_independent(self, i, h):
        return i > 0 and self.responsibility_state(i, h) > 0 and i not in self.stop_sites # stop_sites do not emit.

    def q_function_state(self, rec=False):
        cond = self.q_function_state_cond_independent if self.independent else self.q_function_state_cond
        if not rec:
            return sum([ self.responsibility_state(i, h)*self.emission_prob_log(i, h, self.params) for i in range(self.length-1) for h in range(self.hclass) if cond(i, h)])
        else:
            return sum([ self.responsibility_state(i, h)*self.fb.emission[i, h] for i in range(self.length-1) for h in range(self.hclass) if cond(i, h)])

    def q_function_transition(self):
        return sum([self.responsibility_transition(i, h, k)*self.transition_param[h, k] for k in range(self.hclass) for h in range(self.hclass) for i in range(0, self.length-1) if self.responsibility_transition(i, h, k) > 0.0])

    def debug_print_q_function(self, sindex, params):
        for h in range(self.hclass):
            for i in range(1, self.length-1):
                value = self.responsibility_state(i, h)*self.emission_prob_log(i, h, params)
                print(value, end=" ")
        print('')
        state = self.q_function_state()
        trans = self.q_function_transition()
        q = sum(state, trans)
        print('q ', q)
        sys.exit(1)

    def q_function(self, sindex = -1, param = None):
        if sindex >= 0:
            self.set_IDR_params(sindex, param)
            self.set_pseudo_value(sindex)
        params = self.get_IDR_params()
        state = self.q_function_state(sindex < 0)
        trans = self.q_function_transition()
        return state+trans

    def q_function_const(self, sindex = -1, param = None): #Set initial theta values after computation.
        original_params = None
        if sindex >= 0:
            original_params = self.get_IDR_params().copy()
            if self.verbose:
                print("HMM: calc q_function (%s)." % ['new aparam', 'new sparam', 'no renewal'][sindex], end="")
                print(original_params, "->", param)
        q = self.q_function(sindex, param)
        if sindex >= 0:
            self.set_IDR_params(sindex, original_params[sindex])
            self.set_pseudo_value(sindex)
        return q

    def calc_q_function_grad_at_once(self, s, rep_class):
        grad = [0.]*3
        params = self.get_IDR_params(s)
        qlist = self.qfuncgrad(params, self.pseudo_value[s])
        return self.qfuncgrad_resp_single_core(qlist, rep_class)

    def q_function_grad(self, grad_max = 3):
        grad = []
        for s in range(len(self.v)):
            if s == 0:
                rep_class = 1 # acc == reproducible
            elif s == 1:
                rep_class = 2 # stem == reproducible
            grad.append(self.calc_q_function_grad_at_once(s, rep_class))
        return grad

    def q_function_grad_single_variable(self, s, p):
        if s == 0:
            rep_class = 1 # acc == reproducible
        elif s == 1:
            rep_class = 2 # stem == reproducible
        params = self.get_IDR_params(s)
        qlist = self.qfuncgrad(params, self.pseudo_value[s])
        return self.qfuncgrad_resp_single_variable(qlist, rep_class, p)

    def q_function_discrete(self, sindex, pindex, pthres, mthres):
        if pindex == 2 or pindex == 3:
            mthres = max(0.0, mthres)
            pthres = min(1.0, pthres)
        if pindex == 1:
            mthres = max(HMM.EPS, mthres)
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

    def numerical_differential(self, x1, x2, params, rep, func, name, grad_max=3, log=True): # not fully implemented for triplicates.
        grad = []
        alpha = 0.001
        if len(params) != 4:
            params = params[0]
        for i, p in enumerate(params):
            if i == grad_max:  break
            ps = clip_params(i, p+alpha, p-alpha)
            tparams = params.copy()
            qs = []
            for j in range(2):
                tparams[i] = ps[j]
                qp.append(func(rep, tparams, x1, x2))
            if not log:
                qs = [exp(q) for q in qs]
            grad.append((qs[0]-qs[1])/(ps[0]-ps[1]))
            if self.verbose:
                print('HMM:', i, name, (qs[0]-qs[1])/(ps[0]-ps[1]), qs[0], qs[1], ps[0], ps[1])
        return grad

    def q_function_emi_grad_num(self, x1, x2, params, rep, grad_max = 3, log = True):
        return self.numerical_differential(x1, x2, params, rep, self.emission_prob_log_value, "emi_gradient", grad_max, log)

    def q_function_comp_grad_num(self, x1, x2, params, rep, grad_max = 3, log = True):
        return self.numerical_differential(x1, x2, params, rep, self.emission_prob_log_value_ris, "comp_gradient", grad_max, log)

    def q_function_grad_num(self, grad_max = 3): # not fully implemented for triplicates.
        assert len(self.v[0]) == 2
        grad = []
        alpha = 0.001
        for s in range(len(self.v)):
            rep_class = 1 if s == 0 else 2 # acc == reproducible or stem == reproducible
            grad.append(self.numerical_differential())
            for i, p in enumerate(self.get_IDR_params(s)):
                if i == grad_max:  break
                pthres, mthres = clip_params(i, p+alpha, p-alpha)
                qp, qm, diff = self.q_function_discrete(s, i, pthres, mthres)
                grad[s].append((qp-qm)/diff)
                if self.verbose:
                    print('HMM:', i, "gradient", qp-qm, qp, qm)
        return grad

    def check_partition_funciton(self):
        flag = True
        if self.independent:
            for i in range(1, len(self.stop_sites)):
                if i > 1:
                    forward = self.fb.forward[self.stop_sites[i], 0]
                    backward = logsumexp_inf([self.fb.backward[self.stop_sites[i-1]+1, k]+self.transition_param[0, k] for k in range(self.hclass)])
                    diff=forward-backward
                    print("Check difference", i, self.stop_sites[i], self.stop_sites[i-1], forward, backward, diff)
                    flag = (abs(diff) < HMM.EPS and flag)
                else:
                    diff=self.fb.forward[self.stop_sites[i], 0]-self.fb.backward[self.stop_sites[i-1], 0]
                    print("Check difference", i, self.stop_sites[i], self.stop_sites[i-1], self.fb.forward[self.stop_sites[i], 0], self.fb.backward[self.stop_sites[i-1], 0], diff)
                    flag = (abs(diff) < HMM.EPS and flag)
            assert flag
        else:
            assert abs(self.fb.pf()-self.fb.backward[0, 0]) < HMM.EPS

    def hidden_list_forward(self, i):
        return (range(self.hclass) if self.fb.hidden[i] < 0 else [self.fb.hidden[i]],
                range(self.hclass) if self.fb.hidden[i-1] < 0 else [self.fb.hidden[i-1]])

    def hidden_list_backward(self, i):
        return (range(self.hclass) if self.fb.hidden[i] < 0 else [self.fb.hidden[i]],
                range(self.hclass) if self.fb.hidden[i+1] < 0 else [self.fb.hidden[i+1]])

    def forward(self, i, h, k_list):
        if i == self.length-1:
            return logsumexp_inf([self.fb.forward[self.length-2, k]+self.transition_param[k, 0] for k in k_list])
        else:
            return logsumexp_inf([self.fb.forward[i-1, k]
                                  +self.transition_param[k, h]
                                  +self.fb.emission[i, h] for k in k_list])

    def backward(self, i, h, k_list):
        if i == 0:
            return logsumexp_inf([self.fb.backward[1, k]+self.transition_param[0, k] for k in k_list])
        else:
            return logsumexp_inf([self.fb.backward[i+1, k]
                                  +self.transition_param[h, k]
                                  +self.fb.emission[i, h] for k in k_list])

    def forward_independent(self, i, h, k_list):
        emission = HMM.unmap_prior[h] if i in self.stop_sites or i == self.length-1 else self.fb.emission[i, h]
        tk_list = [0] if i-1 > 0 and i-1 in self.stop_sites else k_list
        return logsumexp_inf([(HMM.unmap_prior[k] if i-1 > 0 and i-1 in self.stop_sites else self.fb.forward[i-1,k])
                              +self.transition_param[k, h]
                              +emission for k in tk_list])

    def backward_independent(self, i, h, k_list):
        if i == 0:
            return logsumexp_inf([HMM.unmap_prior[h]
                                 +self.fb.backward[i+1, k]
                                 +self.transition_param[h, k] for k in k_list])
        elif i in self.stop_sites:
            return HMM.unmap_prior[h]
        else:
            return logsumexp_inf([self.fb.backward[i+1, k]
                                  +self.transition_param[h, k]
                                  +self.fb.emission[i, h] for k in k_list])

    def fill_fb(self):
        forward, backward = self.forward, self.backward
        self.set_emission()
        self.fb.forward[0, 0] = 0.
        self.fb.backward[self.length-1,0] = 0.
        if self.verbose:
            print('HMM: start forward and backward')
        for i in range(1, self.length):
            h_list = (range(self.hclass) if i not in self.stop_sites else [0])
            for h in h_list:
                self.fb.forward[i, h] = forward(i, h, range(self.hclass))
        for i in range(self.length-2, -1, -1): # transition_param is used as transposed.
            h_list = (range(self.hclass) if i not in self.stop_sites else [0])
            for h in h_list:
                self.fb.backward[i, h] = backward(i, h, range(self.hclass))
        self.check_partition_funciton()
        self.set_responsibility()
        if self.verbose:
            print('HMM: Done all fb process.', (self.fb.pf(), self.fb.backward[0, 0]), '(partition function)')
            self.print_result()

        """
        state and output location
                0           1          ...     length-2     length-1
        state   unmappable  any        ...     unmappable   unmappable
        output  u1          u2         ...     uN        none
        stop_sites = [0, n1, n2, ..., length-2] (end of each transcript should be unmappable)

        """

    def fill_fb_train(self):
        forward, backward = self.forward, self.backward
        self.set_emission()
        self.fb.forward[0, 0] = 0.
        self.fb.backward[self.length-1,0] = 0.
        if self.verbose:
            print('HMM: start forward and backward (train mode)')
        for i in range(1, self.length):
            h_list, k_list = self.hidden_list_forward(i)
            for h in h_list:
                self.fb.forward[i, h] = forward(i, h, k_list)
        for i in range(self.length-2, -1, -1): # transition_param is used as transposed.
            h_list, k_list = self.hidden_list_backward(i)
            for h in h_list:
                self.fb.backward[i, h] = backward(i, h, k_list)
        self.check_partition_funciton()
        self.set_responsibility()
        if self.verbose:
            print('HMM: Done all fb process.', (self.fb.pf(), self.fb.backward[0, 0]), '(partition function)')
            self.print_result()


    def fill_fb_train_independent(self):
        forward, backward = self.forward_independent, self.backward_independent
        self.set_emission()
        self.fb.forward[0, 0] = 0.
        self.fb.backward[self.length-1,0] = 0.
        if self.verbose:
            print('HMM: start forward and backward (train mode)')
        for i in range(1, self.length):
            h_list, k_list = self.hidden_list_forward(i)
            for h in h_list:
                self.fb.forward[i, h] = forward(i, h, k_list)
        for i in range(self.length-2, -1, -1): # transition_param is used as transposed.
            h_list, k_list = self.hidden_list_backward(i)
            for h in h_list:
                self.fb.backward[i, h] = backward(i, h, k_list)
        self.set_responsibility_independent()
        if self.verbose:
            print('HMM: Done all fb process.', (self.fb.pf(), float('nan')), '(partition function of the last transcript)')
            self.print_result()
        self.check_partition_funciton()

    def fill_fb_independent(self):
        forward, backward = self.forward_independent, self.backward_independent
        self.set_emission()
        self.fb.forward[0, 0] = 0.
        self.fb.backward[self.length-1,0] = 0.
        if self.verbose:
            print('HMM: start forward and backward')
        for i in range(1, self.length):
            for h in (range(self.hclass) if i not in self.stop_sites else [0]):
                self.fb.forward[i, h] = forward(i, h, range(self.hclass))
            if i%10000 == 0:    print('# HMM forward', i)
        for i in range(self.length-2, -1, -1): # transition_param is used as transposed.
            for h in (range(self.hclass) if i not in self.stop_sites else [0]):
                self.fb.backward[i, h] = backward(i, h, range(self.hclass))
            if i%10000 == 0:    print('# HMM backward', i)
        self.check_partition_funciton()
        self.set_responsibility_independent()
        if self.verbose:
            print('HMM: Done all fb process.', (self.fb.pf(), float('nan')), '(partition function of the last transcript)')


    def check_prob(self):
        for i in range(1, self.length-1):
            temp = [self.responsibility_state(i, h) for h in range(self.hclass)]
            assert abs(sum(temp)-1.0) < HMM.EPS
        for i in range(1, self.length-1):
            temp = [[self.responsibility_transition(i, h, k) for k in range(self.hclass)] for h in range(self.hclass)]
            # print(sum([sum(x) for x in temp]))
            assert abs(sum([sum(x) for x in temp])-1.0) < HMM.EPS

    def debug_print(self):
        self.check_prob()

    def print_result(self):
        print('HMM: Write fb to file ...')
        np.set_printoptions(threshold=np.nan)
        np.savetxt(self.f_file, np.hstack((np.array([[0] if i in self.stop_sites else [i] for i in range(self.length)]), self.fb.forward)))
        np.savetxt(self.b_file, np.hstack((np.matrix([[0] if i in self.stop_sites else [i] for i in range(self.length)]), self.fb.backward)))
        np.savetxt(self.resp_file, np.hstack((np.array([[0] if i in self.stop_sites else [i] for i in range(self.length)]), self.fb.responsibility)), fmt="%.7f")

    def forward_backward(self, transition_param, params):
        if self.verbose:    print("HMM: forward backward.")
        self.set_IDR_params(-1, params)
        self.set_transition_param_log(transition_param)
        self.set_pseudo_value()
        if self.fb == None or self.fb.forward.shape[0] != len(self.pseudo_value[0][0]):
            self.fb = ForwardBackward(len(self.pseudo_value[0][0]), self.hclass, self.hidden_class)
            self.length = self.fb.forward.shape[0]
        else:
            self.fb.store_old()
            self.fb.fill_inf()
        if self.fb.hidden is not None:
            if self.independent:
                self.fill_fb_train_independent()
            else:
                self.fill_fb_train()
        else:
            if self.independent:
                self.fill_fb_independent()
            else:
                self.fill_fb()
        self.debug_print()
        return self.q_function()
