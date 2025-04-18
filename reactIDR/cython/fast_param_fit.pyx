import numpy as np_py
cimport numpy as np_cy

np_cy.import_array()

from libc.math cimport exp, sqrt, pow, log, erf, fabs, log1p
from libc.stdlib cimport malloc, free

cdef inline double d_max(double a, double b): return a if a >= b else b
cdef inline double d_min(double a, double b): return a if a <= b else b


cpdef double c_my_cdf(double x, double mu, double sigma, double p):
    cdef double norm_x = (x-mu)/sqrt(sigma)
    return 0.5*(p*erf(norm_x/sqrt(2)) + (1.-p)*erf(x/sqrt(2)) + 1. )

cpdef double c_logsumexp(double x, double y):
    if x > y:
        return (x + log1p(exp(-x+y)))
    return (y + log1p(exp(-y+x)))

def c_logsumexp_inf(np_cy.ndarray[np_cy.double_t, ndim=1] x):
    cdef double total = 0.0
    for i in range(len(x)):
        if i == 0:
            total = x[i]
        else:
            total = c_logsumexp(total, x[i])
    return total

cpdef double c_my_cdf_i(double r, double mu, double sigma, double lamda,
                    double lb, double ub, double EPS):
    for i in range(1000):
        mid = lb + (ub - lb)/2.;
        guess = c_my_cdf(mid, mu, sigma, lamda)
        if fabs(guess - r) < EPS:
            return mid
        elif guess < r:
            lb = mid
        else:
            ub = mid

    return mid

cpdef double cdf_i(double r, double mu, double sigma, double lamda,
          double lb, double ub, double EPS):
    lb = d_min(0, mu) - 10/d_min(1, sigma)
    ub = d_max(0, mu) + 10/d_min(1, sigma)

    while c_my_cdf(lb, mu, sigma, lamda) > r:
        lb -= 10

    while c_my_cdf(ub, mu, sigma, lamda) < r:
        ub += 10

    cdef double res = c_my_cdf_i(r, mu, sigma, lamda, lb, ub, EPS)
    if c_my_cdf(res, mu, sigma, lamda) - r < EPS:
        return res

    assert False

def c_my_compute_pseudo_values(
        np_cy.ndarray[np_cy.int64_t, ndim=1] rs,
        np_cy.ndarray[np_cy.double_t, ndim=1] zs,
        double mu, double sigma, double lamda,
        double EPS):
    cdef int N = len(rs)
    cdef double pseudo_N = N+1
    cdef double* ordered_zs = <double * >malloc(N*sizeof(double))

    cdef double smallest_r = 1./pseudo_N
    lb = d_min(0, mu)
    while c_my_cdf(lb, mu, sigma, lamda) > smallest_r:
        lb -= 1

    ub = d_max(0, mu)
    while c_my_cdf(ub, mu, sigma, lamda) < 1-smallest_r:
        ub += 1

    lb = c_my_cdf_i(smallest_r, mu, sigma, lamda, lb, ub, EPS)
    ordered_zs[0] = lb

    cdef int i = 0, j= 0
    cdef double r = 0
    cdef double res = 10, prev_res = 10
    for i in range(1, N):
        r = (i+1)/pseudo_N
        if c_my_cdf(ub, mu, sigma, lamda) < r:
            ub += 10
        res = c_my_cdf_i(r, mu, sigma, lamda, lb, ub, EPS)

        ordered_zs[i] = res
        lb = res
        ub = lb + 2*(res - ordered_zs[i-1])

    for i in range(N):
        zs[i] = ordered_zs[rs[i]]
    free( ordered_zs )

    return zs



def c_calc_gaussian_lhd(double x1, double x2, list theta):
    cdef double mu      = theta[0]
    cdef double sigma   = theta[1]
    cdef double rho     = theta[2]
    cdef double diff_x1 = x1-mu
    cdef double diff_x2 = x2-mu
    cdef double pi      = 3.14159265358979323846264338327950288419716939937510582
    cdef double loglik  = (-log(2.*pi*sigma)-0.5*log(1-rho**2)-(diff_x1**2-2*rho*diff_x1*diff_x2+diff_x2**2)/(2*(1-rho**2)*sigma))
    return loglik


def c_calc_gaussian_lhd_3dim(double x1, double x2, double x3, list theta):
    cdef double mu      = theta[0]
    cdef double sigma   = theta[1]
    cdef double rho     = theta[2]
    cdef double diff_x1 = x1-mu
    cdef double diff_x2 = x2-mu
    cdef double diff_x3 = x3-mu
    cdef double pi      = 3.14159265358979323846264338327950288419716939937510582
    cdef double first  = -3.*log(2.*pi*sigma)-0.5*log((2.*rho+1.)*(1.-rho)**2)
    cdef double second = -((1.+rho)*(diff_x1**2+diff_x2**2+diff_x3**2)-2*rho*(diff_x1*diff_x2+diff_x2*diff_x3+diff_x3*diff_x1))/(2.*(2.*rho+1)*(1.-rho)*sigma)
    return first+second
