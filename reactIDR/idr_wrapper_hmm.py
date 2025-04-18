import sys
import os
import reactIDR.utility
from multiprocessing import Pool
import numpy as np
from scipy.optimize import fminbound
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from reactIDR.prob_optimize import *

def plot_pseudo_value(z1, z2, header):
    plt.scatter(z1, z2)
    ofile = header+"_pseudo.png"
    plt.savefig(ofile)
    plt.clf()
    utility.log("# "+ofile+" min="+str(min(min(z1), min(z2)))+" max="+str(max(max(z1), max(z2))))

def plot_lhd_value(lhd, header):
    plt.plot(lhd)
    ofile = header+"_lhd.png"
    plt.savefig(ofile)
    plt.clf()
    utility.log("# "+ofile+" min="+str(min(lhd))+" max="+str(max(lhd)))

def rank_vector(rep):
    return np.lexsort((np.random.random(len(rep)), rep)).argsort()

def only_build_rank_vectors(s1, s2):
    rank1 = rank_vector(s1)
    rank2 = rank_vector(s2)
    return (np.array(rank1, dtype=int), np.array(rank2, dtype=int))

def only_build_rank_vector_23dim(reps):
    return tuple([np.array(rank_vector(rep), dtype=int) for rep in reps])


def comp_pseudo_multi_core(array, param, thread):
    pool = Pool(thread)
    pseudo = pool.starmap(hmm_compute_pseudo_values, [(x, param[0], param[1], param[3]) for x in array if x is not None])
    answer = []
    for x in array:
        if x is None:
            answer.append(None)
        else:
            answer.append(pseudo.pop(0))
    pool.close()
    return answer

def calc_IDR_23dim(r1, r2, r3, theta, thread=1):
    mu, sigma, rho, p = theta
    data = comp_pseudo_multi_core([r1, r2, r3], theta, thread)
    # data = [hmm_compute_pseudo_values(r, mu, sigma, p) if r is not None else None for r in [r1, r2, r3]]
    localIDR = 1.-calc_post_membership_prbs_23dim(*data, theta=theta)
    if r3 is not None:
        localIDR[data[0] + data[1] + data[2] < 0] = 1
    else:
        localIDR[data[0] + data[1] < 0] = 1
    localIDR = np.clip(localIDR, CONVERGENCE_EPS_DEFAULT, 1)
    local_idr_order = localIDR.argsort()
    ordered_local_idr = localIDR[local_idr_order]
    ordered_local_idr_ranks = rankdata(ordered_local_idr, method='max')
    IDR = [ ordered_local_idr[:int(rank)].mean() for i, rank in enumerate(ordered_local_idr_ranks) ]
    IDR = np.array(IDR)[local_idr_order.argsort()]
    return localIDR, IDR

def get_idr_value_23dim(theta, thread, r1, r2, r3=None):
    localIDRs, IDR = calc_IDR_23dim(r1, r2, r3, np.array(theta), max(1, thread))
    return localIDRs, IDR


def get_concatenated_score(seta, tdict):
    return np.asarray([ item for sublist in [tdict[i] for i in seta] for item in sublist])

def get_concatenated_scores(seta, dict1, dict2):
    s1 = get_concatenated_score(seta, dict1)
    s2 = get_concatenated_score(seta, dict2)
    return s1, s2


def get_rank_dictionary(data):
    ulist = sorted(list(set(data)))
    return dict(zip(ulist, rankdata(ulist)))

def remove_no_data(s1, s2, idr):
    thres = 0
    l = [(s1[i], s2[i], idr[i]) for i in range(len(s1)) if s1[i] > thres and s2[i] > thres]
    s1, s2, idr = [list(t) for t in zip(*l)]
    return s1, s2, idr

try:
    #import idr
    #from idr.optimization import EM_iteration, CA_iteration, grid_search
    #print('Import idr succeded.', file=sys.stderr)

    def CA_step_23dim(z1, z2, z3, theta, index, min_val, max_val):
        inner_theta = theta.copy()

        def f(alpha):
            inner_theta[index] = theta[index] + alpha
            return -calc_mix_gaussian_lhd_23dim(z1, z2, z3, inner_theta)

        assert theta[index] >= min_val
        min_step_size = min_val - theta[index]
        assert theta[index] <= max_val
        max_step_size = max_val - theta[index]

        alpha = fminbound(f, min_step_size, max_step_size)
        prev_lhd = -f(0)
        new_lhd = -f(alpha)
        if new_lhd > prev_lhd:
            theta[index] += alpha
        else:
            new_lhd = prev_lhd
        return theta, new_lhd

    def CA_iteration_23dim(z1, z2, z3, prev_theta, max_iter,
                           fix_mu=False, fix_sigma=False, eps=1e-12):
        init_lhd = calc_mix_gaussian_lhd_23dim(z1, z2, z3, prev_theta)
        prev_lhd = init_lhd
        min_vals = [MIN_MU, MIN_SIGMA, MIN_RHO, MIN_MIX_PARAM]
        max_vals = [MAX_MU, MAX_SIGMA, MAX_RHO, MAX_MIX_PARAM]
        theta = np.array(prev_theta).copy()
        for i in range(max_iter):
            for index, (min_val, max_val) in enumerate(zip(min_vals, max_vals)):
                if index == 0 and fix_mu: continue
                if index == 1 and fix_sigma: continue
                theta, new_lhd = CA_step_23dim(z1, z2, z3, theta, index, min_val, max_val)

            theta, changed_params = clip_model_params(theta)
            assert changed_params == False

            if not changed_params:
                assert new_lhd + 1e-6 >= prev_lhd
                if new_lhd - prev_lhd < eps:
                    return theta, new_lhd

            prev_theta = theta
            prev_lhd = new_lhd

        return theta, new_lhd

    def EM_step_23dim(z1, z2, z3, starting_point):
        i_mu, i_sigma, i_rho, i_p = starting_point
        ez = calc_post_membership_prbs_23dim(z1, z2, z3, starting_point)
        if z3 is not None:
            data = [z1, z2, z3]
        else:
            data = [z1, z2]
        # just a small optimization
        ez_sum = ez.sum()

        mus = [(ez*z).sum()/(ez_sum) for z in data]
        mu = np.mean(mus)
        weighted_sum_sqs = [(ez*((z-mu)**2)).sum() for z in data]
        weighted_sum_prod = 0.0
        if len(data) == 3:
            weighted_sum_prod = (ez*(data[0]-mu)*(data[1]-mu)*(data[2]-mu)).sum()
        else:
            weighted_sum_prod = (ez*(data[0]-mu)*(data[1]-mu)).sum()
        sigma = math.sqrt(sum(weighted_sum_sqs)/(len(data)*ez_sum))
        rho = len(data)*weighted_sum_prod/sum(weighted_sum_sqs)
        p = ez_sum/len(ez)
        return np.array([mu, sigma, rho, p])


    def EM_iteration_23dim(z1, z2, z3, prev_theta, max_iter,
                     fix_mu=False, fix_sigma=False, eps=1e-12):

        init_lhd = calc_mix_gaussian_lhd_23dim(z1, z2, z3, prev_theta)
        prev_lhd = init_lhd
        for i in range(max_iter):
            theta = EM_step_23dim(z1, z2, z3, prev_theta)
            theta, changed_params = clip_model_params(theta)
            new_lhd = calc_mix_gaussian_lhd_23dim(z1, z2, z3, theta)
            # if the model is at the boundary, abort
            if changed_params:
                return theta, new_lhd, True
            if not new_lhd + 1e-6 >= prev_lhd:
                print('Error? not optimized', new_lhd, "<-", prev_lhd)
                return prev_theta, prev_lhd, False
            if new_lhd - prev_lhd < eps:
                return theta, new_lhd, False

            prev_theta = theta
            prev_lhd = new_lhd

        return theta, new_lhd, False

    def EMP_with_pseudo_value_algorithm_3dim(r1, r2, r3, theta_0,
                                                N=100, EPS=1e-4, fix_mu=False, fix_sigma=False, image=False, header="", grid=True):
        theta = theta_0
        data = [hmm_compute_pseudo_values(z, theta[0], theta[1], theta[3]) if z is not None else None for z in [r1, r2, r3]]
        max_num_EM_iter = 100
        max_num_EM_iter, N = 2, 2
        lhd = [calc_mix_gaussian_lhd_23dim(*data, theta=theta_0)]
        thetas = [theta_0]
        if grid:
            theta = hmm_grid_search(r1, r2, r3)
        for i in range(N):
            prev_theta = theta
            # EM only works in the unconstrained case
            if not fix_mu and not fix_sigma:
                theta, new_lhd, changed_params = EM_iteration_23dim(*data, prev_theta=prev_theta, max_iter=max_num_EM_iter,
                                                                    fix_mu=fix_mu, fix_sigma=fix_sigma, eps=EPS/10.)
            if fix_mu or fix_sigma or changed_params:
                theta = prev_theta
                theta, new_lhd = CA_iteration_23dim(*data, prev_theta=prev_theta, max_iter=max_num_EM_iter,
                                                    fix_mu=fix_mu, fix_sigma=fix_sigma, eps=EPS/10.)

            sum_param_change = sum([np.abs(c-p) for c, p in zip(theta , prev_theta)])
            prev = data
            data = [hmm_compute_pseudo_values(z, theta[0], theta[1], theta[3]) if z is not None else None for z in [r1, r2, r3]]
            mean_pseudo_val_change = np.sum([ np.abs(p-z).mean() for p, z in zip(prev, data) if p is not None])
            if i > 3 and ((sum_param_change < EPS and mean_pseudo_val_change < EPS)):# or (new_lhd-lhd[-1] < -EPS) or (new_lhd > 0)):
                lhd.append(new_lhd)
                thetas.append(theta)
                break
            lhd.append(new_lhd)
            thetas.append(theta)
        index = lhd.index(min(lhd))
        return thetas[index], log_lhd_loss_23dim(thetas[index], r1, r2, r3)

    def estimate_copula_params(r1, r2, r3=None, theta_0=(1., 1., 0.1, 0.3),
                                max_iter=100, convergence_eps=1e-10, fix_mu=False, fix_sigma=False, image=False, header="", grid=True):
        theta, loss = EMP_with_pseudo_value_algorithm_3dim(r1, r2, r3, theta_0, N=max_iter, EPS=convergence_eps,
                                                            fix_mu=fix_mu, fix_sigma=fix_sigma, image=image, header=header, grid=grid)
        return theta, loss


    def fit_copula_parameters(xvar, mu=1, sigma=0.3, rho=DEFAULT_RHO, p=DEFAULT_MIX_PARAM,
                                max_iter=MAX_ITER_DEFAULT,
                                convergence_eps=CONVERGENCE_EPS_DEFAULT,
                                image=False, header="",
                                fix_mu=False, fix_sigma=False, grid=False):
        starting_point = (mu, sigma, rh, p)
        if not grid:
            grid = (len(xvar[0]) < 100000)
        idr.log("Starting point: [%s]"%" ".join("%.2f" % x for x in starting_point))
        assert len(xvars) >= 2
        data = [xvar[i] if i < len(xvars) else None for i in range(3)]
        theta, loss = estimate_copula_params(*xvar, theta_0=starting_point,
            max_iter=max_iter, convergence_eps=convergence_eps, fix_mu=fix_mu, fix_sigma=fix_sigma, image=image, header=header, grid=grid)
        idr.log("Finished running IDR on the datasets", 'VERBOSE')
        idr.log("Final parameter values: [%s]"%" ".join("%.2f" % x for x in theta))
        return theta, loss

except ImportError:
    print('Import idr failed.', file=sys.stderr)
    def estimate_copula_params(r1, r2, r3=None, theta_0=(1., 1., 0.1, 0.3),
                                max_iter=100, convergence_eps=1e-10, fix_mu=False, fix_sigma=False, image=False, header="", grid=True):
        return theta_0, float('-inf')
