from scipy.misc import logsumexp
import numpy as np
import math
from scipy.optimize import brentq
from idr.inv_cdf import cdf, cdf_i, c_compute_pseudo_values

from idr.optimization import compute_pseudo_values, EM_iteration, CA_iteration, log_lhd_loss, grid_search
from math import pi, exp, expm1, sqrt
from scipy.misc import logsumexp
from scipy.special import erf
from __init__ import *

def logsumexp_inf(x):
    temp = [v for v in x if v != -float('infinity')]
    if len(temp) == 0:  return -float('infinity')
    return logsumexp(temp)

def diffexp_inf(x, y):
    if x == -float('infinity'): return -exp(y)
    elif y == -float('infinity'): return exp(x)
    elif x < y:   return exp(y)*expm1(x-y)
    else:   return exp(x)*expm1(y-x)

    temp = [v for v in x if v != -float('infinity')]
    if len(temp) == 0:  return -float('infinity')
    return logsumexp(temp)

def log_lhd_loss(r1, r2, theta):
    mu, sigma, rho, p = theta
    z1 = hmm_compute_pseudo_values(r1, mu, sigma, p)
    z2 = hmm_compute_pseudo_values(r2, mu, sigma, p)
    return calc_mix_gaussian_lhd(z1, z2, theta)

def calc_gaussian_lhd(x1, x2, theta):
    mu, sigma, rho, _ = theta
    diff_x1 = x1-mu
    diff_x2 = x2-mu
    loglik = (-math.log(2.*pi*sigma)-0.5*math.log(1-rho**2)-(diff_x1**2-2*rho*diff_x1*diff_x2+diff_x2**2)/(2*(1-rho**2)*sigma))
    return loglik

def calc_gaussian_lhd_1dim(x, mu, sigma):
    norm_x = (x-mu)/sqrt(sigma)
    loglik = (-0.5*(math.log(2.*pi*sigma))-(norm_x**2)/2.)
    return loglik

def calc_mix_gaussian_lhd(x1, x2, theta):
    mu, sigma, rho, p = theta
    signal = calc_gaussian_lhd(x1, x2, (mu, sigma, rho, 0.))
    noise = calc_gaussian_lhd(x1, x2, (0., 1., 0., 0))
    loglike = ([math.log(p*exp(s)+(1-p)*exp(n)) for s, n in zip(signal, noise)])
    return loglike.sum()

def csdf(x):
    return 0.5*(erf(x/sqrt(2))+1)

def logR(x1, x2, params):
    mu, sigma, rho, p = params
    return calc_gaussian_lhd(x1, x2, params)

def logI(x1, x2, params):
    mu, sigma, rho, p = params
    return calc_gaussian_lhd(x1, x2, (0., 1., 0., 0))

def logS_single(x1, mu, sigma, rho, p):
    return math.log(p*exp(calc_gaussian_lhd_1dim(x1, mu, sigma))+(1.-p)*exp(calc_gaussian_lhd_1dim(x1, 0., 1.)))

def logS(x1, x2, params):
    mu, sigma, rho, p = params
    f1 = logS_single(x1, mu, sigma, rho, p)
    f2 = logS_single(x2, mu, sigma, rho, p)
    return f1+f2

def coordinate_ascent(r1, r2, theta, gradient_magnitude, fix_mu=False, fix_sigma=False):
    for j in range(len(theta)):
        if fix_mu and j == 0: continue
        if fix_sigma and j == 1: continue

        prev_loss = calc_loss(r1, r2, theta)

        # find the direction of the gradient
        gradient = np.zeros(len(theta))
        gradient[j] = gradient_magnitude
        init_alpha = 5e-12
        while init_alpha < 1e-2:
            pos = calc_loss(r1, r2, theta - init_alpha*gradient)
            neg = calc_loss(r1, r2, theta + init_alpha*gradient)
            if neg < prev_loss < pos:
                gradient[j] = gradient[j]
                break
            elif neg > prev_loss > pos:
                gradient[j] = -gradient[j]
                break
            else:
                init_alpha *= 10

        #log( pos - prev_loss, neg - prev_loss )
        assert init_alpha < 1e-1

        min_step = 0
        max_step = find_max_step_size(
            theta[j], gradient[j], (False if j in (0,1) else True))

        if max_step < 1e-12: continue

        alpha = fminbound(
            lambda x: calc_loss( r1, r2, theta + x*gradient ),
            min_step, max_step)


        loss = calc_loss( r1, r2, theta + alpha*gradient )
        #log( "LOSS:", loss, prev_loss, loss-prev_loss )
        if loss < prev_loss:
            theta += alpha*gradient

    return theta


def EM_step(z1, z2, starting_point):
    i_mu, i_sigma, i_rho, i_p = starting_point

    ez = calc_post_membership_prbs(starting_point, z1, z2)

    # just a small optimization
    ez_sum = ez.sum()

    mu_1 = (ez*z1).sum()/(ez_sum)
    mu_2 = (ez*z2).sum()/(ez_sum)
    mu = (mu_1 + mu_2)/2

    weighted_sum_sqs_1 = (ez*((z1-mu)**2)).sum()
    weighted_sum_sqs_2 = (ez*((z2-mu)**2)).sum()
    weighted_sum_prod = (ez*(z2-mu)*(z1-mu)).sum()

    sigma = sqrt((weighted_sum_sqs_1+weighted_sum_sqs_2)/(2*ez_sum))

    rho = 2*(ez*(z1-mu)*(z2-mu)).sum()/(
        weighted_sum_sqs_1 + weighted_sum_sqs_2)

    p = ez_sum/len(ez)

    return np.array([mu, sigma, rho, p])

def my_cdf(x, mu, sigma, p):
    norm_x = (x-mu)/math.sqrt(sigma)
    return 0.5*(p*erf(norm_x/math.sqrt(2)) + (1.-p)*erf(x/math.sqrt(2)) + 1. )

def my_cdf_i(r, mu, sigma, p, lb, ub):
    return brentq(lambda x: my_cdf(x, mu, sigma, p) - r, lb, ub)

def my_compute_pseudo_values(r, mu, sigma, p):
    pseudo_values = []
    for x in r:
        new_x = float(x+1)/(len(r)+1)
        pseudo_values.append( my_cdf_i( new_x, mu, sigma, p, -10, 10) )
    return pseudo_values

def hmm_compute_pseudo_values(vec, mu, sigma, p):
    return my_compute_pseudo_values(vec, mu, sigma, p)
    # return compute_pseudo_values(vec, mu, math.sqrt(sigma), 1-p)

def hmm_grid_search(r1, r2):
    res = []
    best_theta = None
    max_log_lhd = -1e100
    iter_count = 20
    for mu in np.linspace(0.1, 5, num=iter_count):
        print("# ", mu)
        for sigma in np.linspace(0.5, 3, num=iter_count):
            for rho in np.linspace(0.1, 0.9, num=iter_count):
                for pi in np.linspace(0.1, 0.9, num=iter_count):
                    z1 = hmm_compute_pseudo_values(r1, mu, sigma, p)
                    z2 = hmm_compute_pseudo_values(r2, mu, sigma, p)
                    log_lhd = calc_gaussian_lhd(z1, z2, (mu, sigma, rho, pi)).sum()
                    if log_lhd > max_log_lhd:
                        best_theta = (mu, sigma, rho, pi)
                        max_log_lhd = log_lhd
    return best_theta

def expR_grad_x(x1, x2, params): # divided by R
    mu, sigma, rho, p = params
    return -(x1-mu-rho*(x2-mu))/(sigma*(1.-rho**2))

def expI_grad_x(x1, x2, params): # divided by I
    return -x1

def expS_grad_x(x1, x2, params):
    mu, sigma, rho, p = params
    N1r = exp(calc_gaussian_lhd_1dim(x1, mu, sigma))
    N2r = exp(calc_gaussian_lhd_1dim(x2, mu, sigma))
    N1i = exp(calc_gaussian_lhd_1dim(x1, 0., 1.))
    N2i = exp(calc_gaussian_lhd_1dim(x2, 0., 1.))
    return -((x1-mu)/sigma*p*N1r+x1*(1-p)*N1i)*(p*N2r+(1-p)*N2i)

def fx_dfdx(x1, x2, index, params, rep):
    mu, sigma, rho, p = params
    if rep:
        exp_grad, log_func = expR_grad_x, logR
    else:
        exp_grad, log_func = expI_grad_x, logI
    if index > 0:
        x1, x2 = x2, x1 # x1 is set to the target of defferentiation.
    dfdx = (exp_grad(x1, x2, params)*exp(logS(x1, x2, params))-expS_grad_x(x1, x2, params))*exp(log_func(x1, x2, params)-2.*logS(x1, x2, params))
    print("fx_dfdx: ", dfdx/exp(logS_single(x1, mu, sigma, rho, p)), exp_grad(x1, x2, params), exp(logS(x1, x2, params)), -expS_grad_x(x1, x2, params), log_func(x1, x2, params), -2.*logS(x1, x2, params))
    return dfdx/exp(logS_single(x1, mu, sigma, rho, p))

def independent_grad(x1, x2, params):
    return 0.0

# def grad_merge_mu(x1, params):
#     mu, sigma, rho, p = params
#     return -p/math.sqrt(sigma)*exp(calc_gaussian_lhd_1dim(x1, mu, sigma))

# def grad_merge_sigma(x1, params):
#     mu, sigma, rho, p = params
#     return -p*(x1-mu)/(2.*sigma**(3/2))*exp(calc_gaussian_lhd_1dim(x1, mu, sigma))

# def expR_grad_mu(x1, x2, params): # divided by R
#     mu, sigma, rho, p = params
#     return (x1+x2-2*mu)/(sigma*(1+rho))

# def expS_grad_mu(x1, x2, params):
#     mu, sigma, rho, p = params
#     N1r = exp(calc_gaussian_lhd_1dim(x1, mu, sigma))
#     N2r = exp(calc_gaussian_lhd_1dim(x2, mu, sigma))
#     N1i = exp(calc_gaussian_lhd_1dim(x1, 0., 1.))
#     N2i = exp(calc_gaussian_lhd_1dim(x2, 0., 1.))
#     return p*N1r*(x1-mu)/sigma*(p*N2r+(1-p)*N2i)+p*N2r*(x2-mu)/sigma*(p*N1r+(1-p)*N1i)

# def expR_grad_sigma(x1, x2, params): # divided by R
#     mu, sigma, rho, p = params
#     return -1/sigma+((x1-mu)**2+(x2-mu)**2-2*rho*(x1-mu)*(x2-mu))/(2*sigma**2*(1-rho**2))

# def expS_grad_sigma(x1, x2, params):
#     mu, sigma, rho, p = params
#     N1r = exp(calc_gaussian_lhd_1dim(x1, mu, sigma))
#     N2r = exp(calc_gaussian_lhd_1dim(x2, mu, sigma))
#     N1i = exp(calc_gaussian_lhd_1dim(x1, 0., 1.))
#     N2i = exp(calc_gaussian_lhd_1dim(x2, 0., 1.))
#     x1_grad = -1/(2*sigma)+((x1-mu)**2)/(2*sigma**2)
#     x2_grad = -1/(2*sigma)+((x2-mu)**2)/(2*sigma**2)
#     return p*N1r*x1_grad*(p*N2r+(1-p)*N2i)+p*N2r*x2_grad*(p*N1r+(1-p)*N1i)

# def expR_grad_rho(x1, x2, params):
#     mu, sigma, rho, p = params
#     N12r = calc_gaussian_lhd(x1, x2, params)
#     first = (rho/(1-rho**2))
#     second = ((1+rho**2)*(x1-mu)*(x2-mu)-rho*((x1-mu)**2+(x2-mu)**2))/(sigma*(1-rho**2)**2)
#     return (first+second)

# def grad_cond(x1, x2, params, gfuncR, gfuncS, funcR):
#     print('grad_cond', gfuncR(x1, x2, params), exp(logS(x1, x2, params)), -gfuncS(x1, x2, params), exp(funcR(x1, x2, params)), exp(-2.*logS(x1, x2, params)))
#     return (gfuncR(x1, x2, params)*exp(logS(x1, x2, params))-gfuncS(x1, x2, params))*exp(funcR(x1, x2, params)-2.*logS(x1, x2, params))

# def emission_grad_mu(x1, x2, params, rep=True):
#     if rep:
#         grad_cond_mu = grad_cond(x1, x2, params, expR_grad_mu, expS_grad_mu, logR)
#     else:
#         grad_cond_mu = grad_cond(x1, x2, params, independent_grad, expS_grad_mu, logI)
#     print("mu_grad: ", grad_cond_mu, grad_merge_mu(x1, params)*fx_dfdx(x1, x2, 0, params, rep), grad_merge_mu(x2, params)*fx_dfdx(x1, x2, 1, params, rep))
#     return grad_cond_mu-grad_merge_mu(x1, params)*fx_dfdx(x1, x2, 0, params, rep)-grad_merge_mu(x2, params)*fx_dfdx(x1, x2, 1, params, rep)

# def emission_grad_sigma(x1, x2, params, rep=True):
#     if rep:
#         grad_cond_sigma = grad_cond(x1, x2, params, expR_grad_sigma, expS_grad_sigma, logR)
#     else:
#         grad_cond_sigma = grad_cond(x1, x2, params, independent_grad, expS_grad_sigma, logI)
#     print('sigma_grad: ', grad_cond_sigma, grad_merge_sigma(x1, params)*fx_dfdx(x1, x2, 0, params, rep), grad_merge_sigma(x2, params)*fx_dfdx(x1, x2, 1, params, rep))
#     return grad_cond_sigma-grad_merge_sigma(x1, params)*fx_dfdx(x1, x2, 0, params, rep)-grad_merge_sigma(x2, params)*fx_dfdx(x1, x2, 1, params, rep)

# def emission_grad_rho(x1, x2, params, rep=True):
#     if rep:
#         grad = grad_cond(x1, x2, params, expR_grad_rho, independent_grad, logR)
#         print('rho_grad: ', grad)
#         return grad
#     else:
#         return 0.0

class QfuncGrad:
    """docstring for ClassName"""
    def __init__(self, x1, x2, params, msize = 2):
        self.x1 = x1
        self.x2 = x2
        self.params = params
        self.msize = msize
        self.init_all_variables()
        self.set_gaussian_component()
        self.set_grad_component()
        self.set_grad()
        self.set_grad_log()

    def init_all_variables(self):
        self.N1r = 0.
        self.N2r = 0.
        self.N1i = 0.
        self.N2i = 0.
        self.N12r = 0.
        self.N12i = 0.
        self.S1 = 0.
        self.S2 = 0.
        self.S = 0.
        self.logR = 0.
        self.logI = 0.
        self.logS = 0.
        self.dfdx_list = [[], []] # order x1, x2
        self.fx_dfdx_list = [[], []] # order x1, x2
        self.dfdx_list_log = [[], []] # order x1, x2
        self.fx_dfdx_list_log = [[], []] # order x1, x2
        self.grad_merge = [[0]*2, [0]*2] # order mu x1,x2, sigma x1,x2
        self.grad_list_raw = [[0]*3, [0]*3] # order rep mu,sigma,rho, irep mu,sigma,rho, without merginal terms
        self.grad_list_log_raw = [[0]*3, [0]*3] # order rep mu,sigma,rho, irep mu,sigma,rho, without merginal terms
        self.grad_list = [[0]*3, [0]*3] # order rep mu,sigma,rho, irep mu,sigma,rho
        self.grad_list_log = [[0]*3, [0]*3] # order rep mu,sigma,rho, irep mu,sigma,rho

    def debug_print(self):
        print('Debug_qfunc_grad_class', self.x1, self.x2, self.params, '-------')
        for c, n in zip(["N1r", "N2r", "N1i", "N2i", "N12r", "N12i", "S1", "S2", "S", "dfdx(x1,x2)", "fx_dfdx(x1,x2)", "dfdx_log(x1,x2)", "fx_dfdx_log(x1,x2)"],
                        [self.N1r, self.N2r, self.N1i, self.N2i, self.N12r, self.N12i, self.S1, self.S2, self.S, self.dfdx_list, self.fx_dfdx_list, self.dfdx_list_log, self.fx_dfdx_list_log]):
            print("-", c, n)
        for c, n in zip(["grad_merge(mu,sig)", "grad_raw(rep,irep)", "grad(rep,irep)", "grad_raw_log(rep,irep)", "grad_log(rep,irep)"], [self.grad_merge, self.grad_list_raw, self.grad_list, self.grad_list_log_raw, self.grad_list_log]):
            print("-", c, n)

    def set_gaussian_component(self):
        mu, sigma, rho, p = self.params
        self.N1r = exp(calc_gaussian_lhd_1dim(self.x1, mu, sigma))
        self.N2r = exp(calc_gaussian_lhd_1dim(self.x2, mu, sigma))
        self.N1i = exp(calc_gaussian_lhd_1dim(self.x1, 0., 1.))
        self.N2i = exp(calc_gaussian_lhd_1dim(self.x2, 0., 1.))
        self.N12r = exp(calc_gaussian_lhd(self.x1, self.x2, (mu, sigma, rho, p)))
        self.N12i = exp(calc_gaussian_lhd(self.x1, self.x2, (0., 1., 0., 0.)))
        self.S1 = p*self.N1r+(1-p)*self.N1i
        self.S2 = p*self.N2r+(1-p)*self.N2i
        self.s12 = (self.x1-mu)**2+(self.x2-mu)**2-2*rho*(self.x1-mu)*(self.x2-mu)
        self.S = self.S1*self.S2
        self.logR = calc_gaussian_lhd(self.x1, self.x2, (mu, sigma, rho, p))
        self.logI = calc_gaussian_lhd(self.x1, self.x2, (0., 1., 0., 0.))
        self.logS = math.log(self.S1)+math.log(self.S2)

    def expR_grad_x(self, index): # divided by R
        mu, sigma, rho, p = self.params
        x1, x2 = self.x1, self.x2
        if index > 0:
            x1, x2 = x2, x1
        return -(x1-mu-rho*(x2-mu))/(sigma*(1.-rho**2))

    def logR_grad_x(self, index):
        mu, sigma, rho, p = self.params
        x1, x2 = (self.x1, self.x2) if index == 0 else (self.x2, self.x1)
        return -(x1-mu-rho*(x2-mu))/(sigma*(1.-rho**2))

    def expI_grad_x(self, index): # divided by I
        if index > 0:
            return -self.x2
        else:
            return -self.x1

    def logI_grad_x(self, index):
        return self.expI_grad_x(index)

    def expS_grad_x(self, index):
        mu, sigma, rho, p = self.params
        if index > 0:
            x1, x2 = self.x2, self.x1
            return -((x1-mu)/sigma*p*self.N2r+x1*(1-p)*self.N2i)*(p*self.N1r+(1-p)*self.N1i)
        else:
            x1, x2 = self.x1, self.x2
            return -((x1-mu)/sigma*p*self.N1r+x1*(1-p)*self.N1i)*(p*self.N2r+(1-p)*self.N2i)

    def logS_grad_x(self, index):
        mu, sigma, rho, p = self.params
        x1, x2 = (self.x1, self.x2) if index == 0 else (self.x2, self.x1)
        if index > 0:
            return -((x1-mu)/sigma*p*self.N2r+x1*(1-p)*self.N2i)/(p*self.N2r+(1-p)*self.N2i)
        else:
            return -((x1-mu)/sigma*p*self.N1r+x1*(1-p)*self.N1i)/(p*self.N1r+(1-p)*self.N1i)

    def expR_grad_mu(self):
        mu, sigma, rho, p = self.params
        return (self.x1+self.x2-2.*mu)/(sigma*(1.+rho))

    def logR_grad_mu(self):
        mu, sigma, rho, p = self.params
        return self.expR_grad_mu()

    def expI_grad_mu(self):
        return 0.0

    def logI_grad_mu(self):
        return 0.0

    def expS_grad_mu(self):
        mu, sigma, rho, p = self.params
        first = p*self.N1r*(self.x1-mu)/sigma*(p*self.N2r+(1-p)*self.N2i)
        second = p*self.N2r*(self.x2-mu)/sigma*(p*self.N1r+(1-p)*self.N1i)
        return first+second

    def logS_grad_mu(self):
        mu, sigma, rho, p = self.params
        first = p*self.N1r*(self.x1-mu)/(sigma*self.S1)
        second = p*self.N2r*(self.x2-mu)/(sigma*self.S2)
        return first+second

    def expR_grad_sigma(self):
        mu, sigma, rho, p = self.params
        return -1/sigma+((self.x1-mu)**2+(self.x2-mu)**2-2*rho*(self.x1-mu)*(self.x2-mu))/(2*sigma**2*(1-rho**2))

    def logR_grad_sigma(self):
        mu, sigma, rho, p = self.params
        return self.expR_grad_sigma()

    def expI_grad_sigma(self):
        return 0.0

    def logI_grad_sigma(self):
        return 0.0

    def expS_grad_sigma(self):
        mu, sigma, rho, p = self.params
        x1_grad = -1/(2*sigma)+((self.x1-mu)**2)/(2*sigma**2)
        x2_grad = -1/(2*sigma)+((self.x2-mu)**2)/(2*sigma**2)
        return p*self.N1r*x1_grad*(p*self.N2r+(1-p)*self.N2i)+p*self.N2r*x2_grad*(p*self.N1r+(1-p)*self.N1i)

    def logS_grad_sigma(self):
        mu, sigma, rho, p = self.params
        x1_grad = p*self.N1r*(-1/(2*sigma)+((self.x1-mu)**2)/(2*sigma**2))*self.S2
        x2_grad = p*self.N2r*(-1/(2*sigma)+((self.x2-mu)**2)/(2*sigma**2))*self.S1
        return (x1_grad+x2_grad)/self.S

    def expR_grad_rho(self):
        mu, sigma, rho, p = self.params
        first = (rho/(1-rho**2))
        second = ((1-rho**2)*(self.x1-mu)*(self.x2-mu)-rho*self.s12)/(sigma*(1-rho**2)**2)
        return (first+second)

    def logR_grad_rho(self):
        mu, sigma, rho, p = self.params
        first = (rho/(1-rho**2))
        second = ((self.x1-mu)*(self.x2-mu)*(1-rho**2))-rho*self.s12
        return (first+second/(sigma*(1-rho**2)**2))

    def dfdx(self, index):
        dfdx = [(self.expR_grad_x(index)*self.S-self.expS_grad_x(index))*self.N12r/self.S**2]
        dfdx.append((self.expI_grad_x(index)*self.S-self.expS_grad_x(index))*self.N12i/self.S**2)
        return dfdx

    def log_dfdx(self, index):
        dfdx = [self.logR_grad_x(index)-self.logS_grad_x(index), self.logI_grad_x(index)-self.logS_grad_x(index)]
        return dfdx

    def grad_merge_mu(self, index):
        mu, sigma, rho, p = self.params
        if index > 0:
            return -p*self.N2r
        else:
            return -p*self.N1r

    def grad_merge_sigma(self, index):
        mu, sigma, rho, p = self.params
        if index > 0:
            return -p*(self.x2-mu)/(2*sigma)*self.N2r
        else:
            return -p*(self.x1-mu)/(2*sigma)*self.N1r

    def set_grad_component(self):
        mu, sigma, rho, p = self.params
        for i in range(0, self.msize):
            self.dfdx_list[i] = self.dfdx(i)
            self.dfdx_list_log[i] = self.log_dfdx(i)
        for i in range(0, self.msize):
            if i > 0:
                self.fx_dfdx_list[i] = [n/self.S2 for n in self.dfdx_list[i]]
                self.fx_dfdx_list_log[i] = [n/self.S2 for n in self.dfdx_list_log[i]]
            else:
                self.fx_dfdx_list[i] = [n/self.S1 for n in self.dfdx_list[i]]
                self.fx_dfdx_list_log[i] = [n/self.S1 for n in self.dfdx_list_log[i]]
        for i in range(0, self.msize):
            self.grad_merge[0][i] = self.grad_merge_mu(i)
            self.grad_merge[1][i] = self.grad_merge_sigma(i)
        print(self.grad_merge)

    def set_grad(self):
        for r in [0, 1]:
            rep = (r == 0)
            for p in range(0, 3):
                if p == 0:
                    if rep:
                        self.grad_list_raw[r][p] = (self.expR_grad_mu()*self.S-self.expS_grad_mu())*self.N12r/self.S**2
                    else:
                        self.grad_list_raw[r][p] = (self.expI_grad_mu()*self.S-self.expS_grad_mu())*self.N12i/self.S**2
                    self.grad_list[r][p] = self.grad_list_raw[r][p]-(self.grad_merge[p][0]*self.fx_dfdx_list[0][r]+self.grad_merge[p][1]*self.fx_dfdx_list[1][r])
                elif p == 1:
                    if rep:
                        self.grad_list_raw[r][p] = (self.expR_grad_sigma()*self.S-self.expS_grad_sigma())*self.N12r/self.S**2
                    else:
                        self.grad_list_raw[r][p] = (self.expI_grad_sigma()*self.S-self.expS_grad_sigma())*self.N12i/self.S**2
                    self.grad_list[r][p] = self.grad_list_raw[r][p]-(self.grad_merge[p][0]*self.fx_dfdx_list[0][r]+self.grad_merge[p][1]*self.fx_dfdx_list[1][r])
                else:
                    if rep:
                        self.grad_list[r][p] = self.grad_list_raw[r][p] = self.expR_grad_rho()*self.N12r/self.S
                    else:
                        self.grad_list[r][p] = self.grad_list_raw[r][p] = 0.0

    def set_grad_log(self):
        for r in [0, 1]:
            rep = (r == 0)
            for p in range(0, 3):
                if p == 2:
                    if rep:
                        self.grad_list_log[r][p] = self.grad_list_log_raw[r][p] = self.logR_grad_rho()
                    else:
                        self.grad_list_log[r][p] = self.grad_list_log_raw[r][p] = 0.0
                else:
                    if p == 0:
                        if rep:
                            self.grad_list_log_raw[r][p] = self.logR_grad_mu()-self.logS_grad_mu()
                        else:
                            self.grad_list_log_raw[r][p] = self.logI_grad_mu()-self.logS_grad_mu()
                    elif p == 1:
                        if rep:
                            self.grad_list_log_raw[r][p] = self.logR_grad_sigma()-self.logS_grad_sigma()
                        else:
                            self.grad_list_log_raw[r][p] = self.logI_grad_sigma()-self.logS_grad_sigma()
                    # print("grad_list", r, p, "", self.grad_list_log_raw[r][p], "- (", self.grad_merge[p][0], self.fx_dfdx_list_log[0][r], "+", self.grad_merge[p][1], self.fx_dfdx_list_log[1][r], end="=")
                    self.grad_list_log[r][p] = self.grad_list_log_raw[r][p]-(self.grad_merge[p][0]*self.fx_dfdx_list_log[0][r]+self.grad_merge[p][1]*self.fx_dfdx_list_log[1][r])
                    # print(self.grad_list_log[r][p])


def calc_gaussian_mix_q_func_gradient(param, x1, x2, rep, fix_mu, fix_sigma):
    mu, sigma, rho, p = theta
    if fix_mu:
        mu_grad = [0.]*len(x1)
    else:
        mu_grad = [emission_grad_mu(i, j, params, rep) for i, j in zip(x1, x2)]
    if fix_sigma:
        sigma_grad = [0.]*len(x1)
    else:
        sigma_grad = [emission_grad_sigma(i, j, params, rep) for i, j in zip(x1, x2)]
    rho_grad = [emission_grad_rho(i, j, params, rep) for i, j in zip(x1, x2)]
    # p_grad = [emission_grad_p(i, j, params, rep) for i, j in zip(x1, x2)]
    return np.vstack((mu_grad, sigma_grad, rho_grad))

# def clip_model_params(init_theta):
#     theta_changed = False

def calc_gaussian_mix_log_lhd_gradient(theta, z1, z2, fix_mu, fix_sigma):
    mu, sigma, rho, p = theta

    noise_log_lhd = calc_gaussian_lhd(z1, z2, (0., 1., 0., 0.))
    signal_log_lhd = calc_gaussian_lhd(z1, z2, (mu, sigma, rho, 0.))

    # calculate the likelihood ratio for each statistic
    ez = p*np.exp(signal_log_lhd)/(
        p*np.exp(signal_log_lhd)+(1-p)*np.exp(noise_log_lhd))
    ez_sum = ez.sum()

    # startndardize the values
    std_z1 = (z1-mu)/sigma
    std_z2 = (z2-mu)/sigma

    # calculate the weighted statistics - we use these for the
    # gradient calculations
    weighted_sum_sqs_1 = (ez*(std_z1**2)).sum()
    weighted_sum_sqs_2 = (ez*((std_z2)**2)).sum()
    weighted_sum_prod = (ez*std_z2*std_z1).sum()

    if fix_mu:
        mu_grad = 0
    else:
        mu_grad = (ez*((std_z1+std_z2)/(1-rho*rho))).sum()

    if fix_sigma:
        sigma_grad = 0
    else:
        sigma_grad = (
            weighted_sum_sqs_1
            + weighted_sum_sqs_2
            - 2*rho*weighted_sum_prod )
        sigma_grad /= (1-rho*rho)
        sigma_grad -= 2*ez_sum
        sigma_grad /= sigma

    rho_grad = -rho*(rho*rho-1)*ez_sum + (rho*rho+1)*weighted_sum_prod - rho*(
        weighted_sum_sqs_1 + weighted_sum_sqs_2)
    rho_grad /= (1-rho*rho)*(1-rho*rho)

    p_grad = np.exp(signal_log_lhd) - np.exp(noise_log_lhd)
    p_grad /= p*np.exp(signal_log_lhd)+(1-p)*np.exp(noise_log_lhd)
    p_grad = p_grad.sum()

    return np.array((mu_grad, sigma_grad, rho_grad, p_grad))

def clip_model_params(init_theta):
    theta_changed = False
    theta = init_theta
    if theta[0] < MIN_MU:
        theta[0] = MIN_MU
        theta_changed = True

    if theta[1] < MIN_SIGMA:
        theta[1] = MIN_SIGMA
        theta_changed = True

    if theta[2] < MIN_RHO:
        theta[2] = MIN_RHO
        theta_changed = True
    elif theta[2] > MAX_RHO:
        theta[2] = MAX_RHO
        theta_changed = True

    if theta[3] < MIN_MIX_PARAM:
        theta[3] = MIN_MIX_PARAM
        theta_changed = True
    elif theta[3] > MAX_MIX_PARAM:
        theta[3] = MAX_MIX_PARAM
        theta_changed = True

    return theta, theta_changed

