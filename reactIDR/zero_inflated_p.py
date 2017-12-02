#!/usr/bin/env python
"""
zero_inflated_p.py

Zero-inflated Poisson example using simulated data.
Downloaded from https://github.com/pymc-devs/pymc/blob/master/pymc/examples/zip.py.
Edited by carushi (github.com/carushi).
"""
import numpy as np
from pymc import Uniform, Beta, observed, rpoisson, poisson_like
from pymc import MCMC, Matplot

data = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,4,5,10]
mu = Uniform('mu', 0, 20)
psi = Beta('psi', alpha=1, beta=1)

def zip_param(data, mmax = 20, times = 100000):
	n = len(data)
	mu = Uniform('mu', 0, mmax)
	psi = Beta('psi', alpha=1, beta=1)
	M = MCMC(input=[mu, psi, data])
	M.sample(times, times/2)
	#print(mu, psi, data[0:5])
	# print(M.stats())
	# Matplot.plot(M)
	return M.stats()['mu']['mean'], M.stats()['psi']['mean'], M

@observed(dtype=int, plot=False)
def zip_ll(value=data, mu=mu, psi=psi):
    """ Zero-inflated Poisson likelihood """
    like = 0.0
    for x in value:
        if not x: # x == 0
            like += np.log((1. - psi) + psi * np.exp(-mu))
        else:
            like += np.log(psi) + poisson_like(x, mu)
    return like

if __name__ == '__main__':
	mu_true = 5
	psi_true = 0.75
	n = 1000
	data = np.array([rpoisson(mu_true) * (np.random.random() < psi_true)
                 for i in range(n)])
	mu, psi, M =  zip_param(data)
	print(mu, psi)
	print(M.stats())