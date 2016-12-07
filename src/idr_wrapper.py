import idr
import idr.idr
# from idr.optimization import estimate_model_params, old_estimator
import utility
from idr.optimization import compute_pseudo_values, EM_iteration, CA_iteration, log_lhd_loss, grid_search
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt

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


def only_build_rank_vectors(s1, s2):
    rank1 = np.lexsort((np.random.random(len(s1)), s1)).argsort()
    rank2 = np.lexsort((np.random.random(len(s2)), s2)).argsort()
    return (np.array(rank1, dtype=np.int), np.array(rank2, dtype=np.int))

def only_EMP_with_pseudo_value_algorithm(
        r1, r2, theta_0,
        N=100, EPS=1e-4,
        fix_mu=False, fix_sigma=False, image=False, header="", grid=True):
    theta = theta_0
    z1 = compute_pseudo_values(r1, theta[0], theta[1], theta[3])
    z2 = compute_pseudo_values(r2, theta[0], theta[1], theta[3])
    max_num_EM_iter = 100
    lhd = []
    thetas = []
    if grid:
        gtheta = grid_search(r1, r2)
        gtheta = [gtheta[0][0], gtheta[1][0], gtheta[2], gtheta[3]]
        print("# Grid search: ", gtheta, log_lhd_loss(r1, r2, gtheta))
        if log_lhd_loss(r1, r2, theta) > log_lhd_loss(r1, r2, gtheta):  theta = gtheta

    for i in range(N):
        prev_theta = theta
        # EM only works in the unconstrained case
        if not fix_mu and not fix_sigma:
            theta, new_lhd, changed_params = EM_iteration(
                z1, z2, prev_theta, max_num_EM_iter,
                fix_mu=fix_mu, fix_sigma=fix_sigma, eps=EPS/10.)

        if fix_mu or fix_sigma or changed_params:
            theta = prev_theta
            theta, new_lhd = CA_iteration(
                z1, z2, prev_theta, max_num_EM_iter,
                fix_mu=fix_mu, fix_sigma=fix_sigma, eps=EPS/10.)

        sum_param_change = np.abs(theta - prev_theta).sum()
        prev_z1 = z1
        z1 = compute_pseudo_values(r1, theta[0], theta[1], theta[3])
        prev_z2 = z2
        z2 = compute_pseudo_values(r2, theta[0], theta[1], theta[3])
        mean_pseudo_val_change = (
            np.abs(prev_z1-z1).mean() + np.abs(prev_z2-z2).mean())
        # utility.log(("Iter %i" % i).ljust(12)+(" %.2e" % sum_param_change)+(" %.2e" % mean_pseudo_val_change)+(" %.4e" % log_lhd_loss(r1, r2, theta))+" "+str(theta))
        if i > 3 and ((sum_param_change < EPS and mean_pseudo_val_change < EPS)):# or (new_lhd-lhd[-1] < -EPS) or (new_lhd > 0)):
            # theta = prev_theta
            lhd.append(new_lhd)
            thetas.append(theta)
            break
        lhd.append(new_lhd)
        thetas.append(theta)
    if image:
        plot_pseudo_value(z1, z2, header)
        plot_lhd_value(lhd, header)
    index = lhd.index(min(lhd))
    return thetas[index], log_lhd_loss(r1, r2, thetas[index])
    # return theta, log_lhd_loss(r1, r2, theta)


def only_estimate_model_params(
        r1, r2,
        theta_0,
        max_iter=100, convergence_eps=1e-10,
        fix_mu=False, fix_sigma=False, image=False, header=""):

    theta, loss = only_EMP_with_pseudo_value_algorithm(
        r1, r2, theta_0, N=max_iter, EPS=convergence_eps,
        fix_mu=fix_mu, fix_sigma=fix_sigma, image=image, header=header)

    return theta, loss

def only_fit_model_and_calc_idr(r1, r2,
                            mu = 1,
                            sigma = 0.3,
                            rho = idr.DEFAULT_RHO,
                            mix = idr.DEFAULT_MIX_PARAM,
                           max_iter=idr.MAX_ITER_DEFAULT,
                           convergence_eps=idr.CONVERGENCE_EPS_DEFAULT,
                           image=False, header="",
                           fix_mu=False, fix_sigma=False):
    # in theory we would try to find good starting point here,
    # but for now just set it to something reasonable
    starting_point = (idr.DEFAULT_MU, idr.DEFAULT_SIGMA, idr.DEFAULT_RHO, idr.DEFAULT_MIX_PARAM)
    # max_iter = 1000
    # print(idr.DEFAULT_RHO)
    # starting_point = (1, 0.3, 0.8, 0.3)
    idr.log("Starting point: [%s]"%" ".join("%.2f" % x for x in starting_point))
    theta, loss = only_estimate_model_params(
        r1, r2,
        starting_point,
        max_iter=max_iter,
        convergence_eps=convergence_eps,
        fix_mu=fix_mu, fix_sigma=fix_sigma, image=image, header=header, grid=(len(r1) < 100000))

    idr.log("Finished running IDR on the datasets", 'VERBOSE')
    idr.log("Final parameter values: [%s]"%" ".join("%.2f" % x for x in theta))
    return theta, loss

def get_concatenated_score(seta, tdict):
    return np.asarray([ item for sublist in [tdict[i] for i in seta] for item in sublist])

def get_concatenated_scores(seta, dict1, dict2):
    s1 = get_concatenated_score(seta, dict1)
    s2 = get_concatenated_score(seta, dict2)
    return s1, s2

def common_transcript(dict1, dict2):
    return set(dict1.keys()) & set(dict2.keys())

def get_rank_dictionary(data):
    ulist = sorted(list(set(data)))
    return dict(zip(ulist, rankdata(ulist)))

def remove_no_data(s1, s2, idr):
    thres = 0
    l = [(s1[i], s2[i], idr[i]) for i in range(len(s1)) if s1[i] > thres and s2[i] > thres]
    s1, s2, idr = [list(t) for t in zip(*l)]
    return s1, s2, idr
