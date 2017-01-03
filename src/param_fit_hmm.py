from scipy.stats import rankdata
from scipy.misc import logsumexp
import numpy as np
import copy
import math
from enum import Enum
from prob_optimize import *
# import random
# import subprocess
# import os
# import math
# from statsmodels.tsa.stattools import acf
# import pvalue
# import pylab
# from idr_wrapper import *
# from plot_image import *
from idr.optimization import compute_pseudo_values, EM_iteration, CA_iteration, log_lhd_loss, grid_search

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
        self.forward = [[-INF]*(length+1)]*dim
        self.backward = [[-INF]*(length+1)]*dim

class ParamFitHMM(object):
    """docstring for ParameterEStimationkword"""
    class Hidden(Enum):
        unmappable = 0
        acc = 1
        stem = 2
    class Sample(Enum):
        case = 0
        cont = 1

    def __init__(self, hclass, data, sample = -1, param=None):
        self.data = data
        self.hclass = hclass
        self.fb = None
        assert hclass == 2 or hclass == 3
        if self.hclass == 2:
            self.init_transition_param = np.matrix('0.95 0.05; 0.8 0.2', dtype=float)
            self.transition_param = self.init_transition_param.copy()
        else:
            self.init_transition_param = np.matrix('0.9 0.05 0.05; 0.7 0.2 0.1; 0.7 0.1 0.2', dtype=float)
            self.transition_param = self.init_transition_param.copy()
        if type(param) == type(None):
            param = (1, 0.2, 0.8, 0.2) # mu, sigma, rho, pi
        param = (math.log(x) for x in param)
        self.sparam = copy.deepcopy(param)
        self.aparam = copy.deepcopy(param)
        self.sample_size = sample
        self.v, self.stop_sites = set_hmm_transcripts()

    def set_hmm_transcripts(self):
        rankdata = [[], []]
        if len(self.data) == 1:
            seta = self.data.keys()
        else:
            seta = common_transcript(self.data[0], self.data[1])
        if self.sample > 0:
            seta = random.sample(seta, min(len(seta), self.sample))
        for i, x in enumerate(self.data):
            self.rankdata[i] = only_build_rank_vectors([0]+x[0], [0]+x[1])
        temp = [len(self.data[0][x]) for x in seta]
        stop_sites = [0]+[sum(temp[0:(i+1)]) for i in range(self.stop_sites)]
        return rankdata, stop_sites

    def emission_prob_log(self, index, h):
        result = []
        for i in range(len(self.data)):
            param = [self.aparam, self.sparam][i]
            deno = logS(param, self.pseudo_value[i][0][index], self.pseudo_value[i][t][0][index])
            if (i == Sample.case and h == Hidden.acc) or (i == Sample.cont and h == Hidden.stem):
                result.append(logR(param, self.data[i][0][index], self.data[i][t][1][index])-deno) # reproducible
            else:
                result.append(logI(param, self.data[i][0][index], self.data[i][t][1][index])-deno) # irreproducible
        return sum(result)

    def fill_fb(self):
        length = self.fb.forward.shape[0]
        self.fb.forward[0,0] = 0
        self.fb.backward[length-1,0] = 0
        for i in range(1, length):
            if i in self.stop_sites:
                self.fb.forward[i,0] = [self.fb.forward[i-1][0]
                                        +self.transition_param[0][h]
                                        +self.emission_prob_log(i-1, h) for h in range(self.hclass)]
            else:
                self.fb.forward[i,:] = [logsumexp([self.fb.forward[i-1][k]
                                                  +self.transition_param[k][h]
                                                  +self.emission_prob_log(i-1, h) for k in range(self.hclass)])
                                                                                 for h in range(self.hclass)]
        for i in range(length-2, 0, -1):
            if i in self.stop_sites:
                self.fb.backward[i, 0] = [self.fb.backward[i+1][0]
                                    +self.transition_param[h][0]
                                    +self.emission_prob_log(i, h) for h in range(self.hclass)]
            else:
                self.fb.backward[i,:] = [logsumexp([self.fb.backward[i+1][k]
                                                   +self.transition_param[h][k]
                                                   +self.emission_prob_log(i, h) for k in range(self.hclass)])
                                                                                for h in range(self.hclass)]

    def forward_backword(self, index):
        self.fb = ForwardBackward(len(self.rankdata[0][0]), hclass)
        self.fill_fb()

    def debug_print(self):
        print(self.rankdata)
        print(self.fb)

    def calc_posterior():
        pass

    def parms_EM():
        pass

    def only_EMP_with_pseudo_value_algorithm(
        N=100, EPS=1e-4,
        fix_mu=False, fix_sigma=False, image=False, header="", grid=True):
        self.pseudo_value = [[], []]
        for i in range(len(self.v)):
            if i == 0:
                param = self.aparam
            else:
                param = self.sparam
            self.pseudo_value[i] = [compute_pseudo_values(self.v[i][0], param[0], param[1], param[3]),
                                compute_pseudo_values(self.v[i][1], param[0], param[1], param[3])]
            max_num_EM_iter = 100
            self.forward_backward()
#     lhd = []
#     thetas = []
#     if grid:
#         gtheta = grid_search(r1, r2)
#         gtheta = [gtheta[0][0], gtheta[1][0], gtheta[2], gtheta[3]]
#         print("# Grid search: ", gtheta, log_lhd_loss(r1, r2, gtheta))
#         if log_lhd_loss(r1, r2, theta) > log_lhd_loss(r1, r2, gtheta):  theta = gtheta

#     for i in range(N):
#         prev_theta = theta
#         # EM only works in the unconstrained case
#         if not fix_mu and not fix_sigma:
#             theta, new_lhd, changed_params = EM_iteration(
#                 z1, z2, prev_theta, max_num_EM_iter,
#                 fix_mu=fix_mu, fix_sigma=fix_sigma, eps=EPS/10.)

#         if fix_mu or fix_sigma or changed_params:
#             theta = prev_theta
#             theta, new_lhd = CA_iteration(
#                 z1, z2, prev_theta, max_num_EM_iter,
#                 fix_mu=fix_mu, fix_sigma=fix_sigma, eps=EPS/10.)

#         sum_param_change = np.abs(theta - prev_theta).sum()
#         prev_z1 = z1
#         z1 = compute_pseudo_values(r1, theta[0], theta[1], theta[3])
#         prev_z2 = z2
#         z2 = compute_pseudo_values(r2, theta[0], theta[1], theta[3])
#         mean_pseudo_val_change = (
#             np.abs(prev_z1-z1).mean() + np.abs(prev_z2-z2).mean())
#         # utility.log(("Iter %i" % i).ljust(12)+(" %.2e" % sum_param_change)+(" %.2e" % mean_pseudo_val_change)+(" %.4e" % log_lhd_loss(r1, r2, theta))+" "+str(theta))
#         if i > 3 and ((sum_param_change < EPS and mean_pseudo_val_change < EPS)):# or (new_lhd-lhd[-1] < -EPS) or (new_lhd > 0)):
#             # theta = prev_theta
#             lhd.append(new_lhd)
#             thetas.append(theta)#
             # break
#         lhd.append(new_lhd)
#         thetas.append(theta)
#     if image:
#         plot_pseudo_value(z1, z2, header)
#         plot_lhd_value(lhd, header)
#     index = lhd.index(min(lhd))
#     return thetas[index], log_lhd_loss(r1, r2, thetas[index])
#     # return theta, log_lhd_loss(r1, r2, theta)

    def fit_model():
        while True:
            forward_backword()
            calc_posterior()
            params_EM()
            if False:
                break
        # return param, hidden
