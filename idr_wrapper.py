import idr
import idr.idr
from idr.optimization import estimate_model_params, old_estimator
import numpy as np
from scipy.stats import rankdata


def only_build_rank_vectors(s1, s2):
    rank1 = np.lexsort((np.random.random(len(s1)), s1)).argsort()
    rank2 = np.lexsort((np.random.random(len(s2)), s2)).argsort()
    return ( np.array(rank1, dtype=np.int),
             np.array(rank2, dtype=np.int) )

def only_fit_model_and_calc_idr(r1, r2,
                           starting_point=None,
                           max_iter=idr.MAX_ITER_DEFAULT,
                           convergence_eps=idr.CONVERGENCE_EPS_DEFAULT,
                           fix_mu=False, fix_sigma=False ):
    # in theory we would try to find good starting point here,
    # but for now just set it to something reasonable
    if type(starting_point) == type(None):
        starting_point = (1, idr.DEFAULT_SIGMA,
                          idr.DEFAULT_RHO, idr.DEFAULT_MIX_PARAM)
    idr.log("Starting point: [%s]"%" ".join("%.2f" % x for x in starting_point))
    theta, loss = estimate_model_params(
        r1, r2,
        starting_point,
        max_iter=max_iter,
        convergence_eps=convergence_eps,
        fix_mu=fix_mu, fix_sigma=fix_sigma)

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
