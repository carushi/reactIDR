from scipy.misc import logsumexp
import numpy as np
import math

def calc_gaussian_lhd(mu, sigma, rho, x1, x2):
    norm_x1 = (z1-mu)/sigma
    norm_x2 = (z2-mu)/sigma
    loglik = (-(math.log(2.)+math.log(pi)+math.log(sigma**2)+0.5*(1.-rho**2))
        -(norm_x1**2-2*rho*norm_x1*norm_x2+norm_x2**2)/(2.*(1-rho**2)))
    return loglik

def calc_gaussian_lhd_1dim(mu, sigma, rho, x):
    norm_x = (z1-mu)/sigma
    loglik = (-0.5(math.log(2.)+math.log(pi)+math.log(sigma**2))
        -(norm_x**2)/2.)
    return loglik

def calc_mix_gaussian_lhd(mu, sigma, rho, q, x1, x2):
    signal = calc_gaussian_lhd(mu, sigma, rho, x1, x2)
    noise = calc_gaussian_lhd(0., 1., 0., x1, x2)
    return logsumexp(math.log(q)+signal, math.log(1.-q)+noise)

# def calc_post_membership_prbs(theta, z1, z2):
#     mu, sigma, rho, p = theta

#     noise_log_lhd = calc_gaussian_lhd(0,0, 1,1, 0, z1, z2)
#     signal_log_lhd = calc_gaussian_lhd(
#         mu, mu, sigma, sigma, rho, z1, z2)

#     ez = p*numpy.exp(signal_log_lhd)/(
#         p*numpy.exp(signal_log_lhd)+(1-p)*numpy.exp(noise_log_lhd))

#     return ez

def logR(params, x1, x2):
    mu, sigma, rho, q = params
    return math.log(q)+calc_gaussian_lhd(mu, sigma, rho, x1, x2)

def logI(params, x1, x2):
    mu, sigma, rho, q = params
    return math.log(1-q)+calc_gaussian_lhd(0., 1., 0., x1, x2)

def logS(params, x1, x2):
    mu, sigma, rho, q = params
    f1 = logsumexp(math.log(q)+calc_gaussian_lhd_1dim(mu, sigma, rho, x1), math.log(1.-q)+calc_gaussian_lhd_1dim(0., 1., 0., x1, x2))
    f2 = logsumexp(math.log(q)+calc_gaussian_lhd_1dim(mu, sigma, rho, x1), math.log(1.-q)+calc_gaussian_lhd_1dim(0., 1., 0., x1, x2))
    return f1+f2
