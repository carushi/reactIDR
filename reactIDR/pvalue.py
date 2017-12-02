from scipy.stats import rankdata
from scipy.misc import factorial
from scipy.optimize import curve_fit
from scipy.stats import poisson
import scipy.stats as ss
import scipy.optimize as so
import numpy as np
import matplotlib.pyplot as plt
import math
import zero_inflated_p


DIST_LIST = ["coverage", "rank", "poisson", "nb", "zip", "zinb", "p-cisgenome", "nb-cisgenome", "ec"]


eps = 0.00001

def uniq(data):
    return sorted(list(set(data)))

def poisson_lambda(ratio):
    global eps
    p_lambda = [ratio[i+1]/ratio[i]/float(i+1) for i in range(len(ratio)-1)] # actually not float(i+1) but !(i+1)
    return max(np.mean(p_lambda), eps)

def nb_lambda(ratio):
    global eps
    alpha = ratio[0]/(2.*ratio[1]-ratio[0])
    beta = 1./(2.*ratio[1]-ratio[0])-1.
    alpha = max(alpha, 1)
    beta = max(beta, eps)
    return alpha, beta

def log_likelihood_f_nb(P, x, sign = 1):
    alpha = np.round(P[0])
    beta = P[1]
    loc = np.round(P[2])
    return sign*np.log(ss.nbinom.pmf(x, alpha, beta, loc)).sum()

class Pvalue_fitting:
    """docstring for ClassName"""

    def __init__(self, data):
        self.data = data
        self.uniq_list = uniq(data)
        self.params = {}

    def set_params(self, name, para):
        self.params[name] = para

    def plot_fitting_poisson(self, head):
        mu = self.params["poisson"][0]
        print([mu])
        weights = np.ones_like(self.data)/len(self.data)
        n, bins, patches = plt.hist(self.data, 20, weights=weights, facecolor='green', alpha=0.75)
        # print(n, bins)
        cdfs, prev = [], -1
        for idx, x in enumerate(bins):
            if x >= 0 and np.round(x) > prev:
                cdfs.append(poisson.cdf(np.round(x), mu)-poisson.cdf(prev, mu))
                prev = np.round(x)
        # print(cdfs)
        plt.plot([np.round(x) for x in bins if x >= 0], cdfs, label='poisson pmf', color='black')
        plt.scatter([np.round(x) for x in bins if x >= 0], cdfs, label='poisson pmf', color='black')
        plt.savefig(head+"_poisson.png")
        plt.close()

    def plot_fitting_nb(self, head):
        a, b, loc = self.params["nb"][0], self.params["nb"][1], 0
        if len(self.params["nb"]) >= 3:   loc = self.params["nb"][2]
        print([a, b, loc])
        weights = np.ones_like(self.data)/len(self.data)
        n, bins, patches = plt.hist(self.data, 20, weights=weights, facecolor='green', alpha=0.75)
        # print(n, bins)
        cdfs, prev = [], -1
        for idx, x in enumerate(bins):
            if x >= 0 and np.round(x) > prev:
                cdfs.append(ss.nbinom.cdf(np.round(x), np.round(a), b, np.round(loc))-ss.nbinom.cdf(int(prev), np.round(a), b, np.round(loc)))
                prev = np.round(x)
        # print(cdfs)
        plt.plot([np.round(x) for x in bins if x >= 0], cdfs, label='nb pmf', color='black')
        plt.scatter([np.round(x) for x in bins if x >= 0], cdfs, label='nb pmf', color='black')
        plt.savefig(head+"_nb.png")
        plt.close()

    def plot_fitting_zip(self, head):
        mu, psi, zero = self.params["zip"][0], self.params["zip"][1], self.params["zip"][2]
        print([mu, psi, zero])
        zero = (1.0-psi)+np.exp(-mu)
        weights = np.ones_like(self.data)/len(self.data)
        n, bins, patches = plt.hist(self.data, 20, weights=weights, facecolor='green', alpha=0.75)
        # print(n, bins)
        cdfs, prev = [], -1
        for idx, x in enumerate(bins):
            if x >= 0 and np.round(x) > prev:
                if prev < 0:    cdfs.append(psi*poisson.cdf(np.round(x), mu)+(1.0-psi))
                else:    cdfs.append(psi*poisson.cdf(np.round(x), mu)-psi*poisson.cdf(np.round(prev), mu))
                prev = np.round(x)
        # print(cdfs)
        plt.plot([int(x) for x in bins if x >= 0], cdfs, label='zip', color='black')
        plt.scatter([int(x) for x in bins if x >= 0], cdfs, label='zip', color='black')
        plt.savefig(head+"_zip.png")
        plt.close()

    def pvalue_nb(self, start = 80, end = 120):
        result = []
        for i in range(start, end):
            _ = so.fmin(log_likelihood_f_nb, [i, 0.5, 0], args=(self.data,-1), full_output=True, disp=False)
            result.append((_[1], _[0]))
        P = sorted(result, key=lambda x: x[0])[0][1]
        self.set_params("nb", [np.round(P[0]), P[1], np.round(P[2])])
        # print(result) #ll, (mu, psi, loc)
        return [ 1.-ss.nbinom.cdf(x-1, np.round(P[0]), P[1], np.round(P[2])) for x in self.uniq_list ]

    def get_ratio_cisgenome(self):
        ratio = []
        for i in [0, 1, 2]:
            if i in self.data:
                ratio.append(float(len([x for x in self.data if x == i])))
            else:
                ratio.append(1.) # avoid 0 division
        return ratio
        # return [i if i not in self.data else float(map(x == i for x in self.data).count(True)) for i in [0, 1, 2]]

    def pvalue_nb_cisgenome(self):
        ratio = self.get_ratio_cisgenome()
        alpha, beta = nb_lambda(ratio)
        self.set_params("nb", [alpha, beta, 0.])
        return [ 1.-ss.nbinom.cdf(x-1, alpha, beta, 0.) for x in self.uniq_list ]

    def pvalue_poisson(self):
        mu = np.mean(self.data)
        self.set_params("poisson", [mu])
        # mu = float(sum(self.data))/float(len(self.data)) # 5 baseのsumカウントなので、それを考える。L/G (gene length)。
        p = [1.-poisson.cdf(x-1, mu) for x in self.uniq_list]
        return p

    def pvalue_poisson_cisgenome(self):
        ratio = self.get_ratio_cisgenome()
        mu = poisson_lambda(ratio)
        self.set_params("poisson", [mu])
        # 0,1,2でnormalizeして、sisgenome peak caller ()
        return [1.-poisson.cdf(x-1, mu) for x in self.uniq_list]

    def pvalue_zero_inflated_poisson(self, output = False):
        mu, psi, M = zero_inflated_p.zip_param(self.data, max(self.data), 10000)
        if output: print(M)
        zero = (1.0-psi)+np.exp(-mu)
        self.set_params("zip", [mu, psi, zero])
        return [1.0-poisson.cdf(x-1, mu)*psi for x in self.uniq_list]

    def num_to_pvalue_dict(self, score_ind):
        assert score_ind > 1
        if score_ind == 2:
            return self.pvalue_poisson()
        elif score_ind == 3:
            return self.pvalue_nb()
        elif score_ind == 4:
            return self.pvalue_zero_inflated_poisson()
        elif score_ind == 5: # TODO
            return []
        elif score_ind == 6:
            return self.pvalue_poisson_cisgenome()
        elif score_ind == 7:
            return self.pvalue_nb_cisgenome()
        else:
            assert False


## score_ind = ["coverage", "rank", "poisson", "nb", "zinb", "p-cisgenome" "nb-cisgenome"]

def dict_to_real_value(data, dictionary, score_ind):
    if score_ind in [-1, 0, 1]:
        return [dictionary[x] for x in data]
    else:
        return [100 if dictionary[x] < 1e-100 else int(np.round(-np.log10(dictionary[x]))) for x in data]

def score_to_pvalue(data, score_ind, output=False, head='plot_fitting_'):
    if score_ind <= 0:
        return data
    if score_ind == 1:
        return list(map(lambda x: int(np.round(float(x-1)/len(data)*1000)), rankdata(data, method='min')))
    pvalue = Pvalue_fitting(data)
    dictionary = dict(zip(pvalue.uniq_list, pvalue.num_to_pvalue_dict(score_ind)))
    if output:
        if score_ind in [2, 6]: pvalue.plot_fitting_poisson(head)
        if score_ind in [3, 7]: pvalue.plot_fitting_nb(head)
        if score_ind in [4]:    pvalue.plot_fitting_zip(head)
    assert len(list(dictionary.keys())) == len(pvalue.uniq_list)
    return dict_to_real_value(data, dictionary, score_ind)
    # return [dictionary[x] for x in data]

def get_dict_for_score_to_pvalue_conversion(data, score_ind):
    assert score_ind > 1
    dictionary = dict(zip(pvalue.uniq_list, pvalue.num_to_pvalue_dict(score_ind)))
    return dictionary

def check_dist_model(dist_model):
    global DIST_LIST
    if dist_model in DIST_LIST:
        if dist_model == "ec":
            return -1
        else:
            return DIST_LIST.index(dist_model)
    else:
        return 0
