import utility
import sys
import argparse
from param_fit_hmm import *
from utility import *

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", dest="case", nargs='+', type=str, help="read replicate data from FILE1, FILE2, ...", metavar="CASE")
    parser.add_argument("--control", dest="control", nargs='+', type=str, help="read replicate data from FILE1, FILE2, ...", metavar="CONTROL", required=False)
    parser.add_argument("-n", "--sample", dest="sample_size", type=int, help="sample size for test (default: 1000)", default=-1, metavar="SAMPLESIZE", required=False)
    parser.add_argument("--mu", dest="mu", type=float, help="mean of reproducible group", default=1., required=False)
    parser.add_argument("--sigma", dest="sigma", type=float, help="variance of reproducible group", default=0.2, required=False)
    parser.add_argument("--rho", dest="rho", type=float, help="dependency of reproducible group", default=0.8, required=False)
    parser.add_argument("--pi", dest="q", type=float, help="ratio of reproducible group", default=0.3, required=False)
    return parser


class IDRHmm:
    """docstring for IDRhmm"""
    def __init__(self, arg):
        self.input = [arg.case, arg.control]
        self.sample_size = arg.sample_size
        self.params = [arg.mu, arg.sigma, arg.rho]

    def read_data_pars_format(self):
        data = [{}, {}]
        for key, one, two in parse_score_iterator(self.input[0][0]):
            data[0][key] = [one, two]
        if len(self.input[1]) > 0:
            for key, one, two in parse_score_iterator(self.input[1][0]):
                data[1][key] = [one, two]
        return data

    def infer_reactive_sites(self):
        hclass = 2
        if len(self.input[1]) > 0:   hclass = 3
        data = self.read_data_pars_format()
        para = ParamFitHMM(hclass, data, self.sample_size, self.params)
        para.only_EMP_with_pseudo_value_algorithm()

# def grid_search(r1, r2 ):
#     res = []
#     best_theta = None
#     max_log_lhd = -1e100
#     for mu in numpy.linspace(0.1, 5, num=10):
#         for sigma in numpy.linspace(0.5, 3, num=10):
#             for rho in numpy.linspace(0.1, 0.9, num=10):
#                 for pi in numpy.linspace(0.1, 0.9, num=10):
#                     z1 = compute_pseudo_values(r1, mu, sigma, pi)
#                     z2 = compute_pseudo_values(r2, mu, sigma, pi)
#                     log_lhd = calc_gaussian_mix_log_lhd((mu, sigma, rho, pi), z1, z2)
#                     if log_lhd > max_log_lhd:
#                         best_theta = ((mu,mu), (sigma,sigma), rho, pi)
#                         max_log_lhd = log_lhd

#     return best_theta

# def only_estimate_model_params(
#         r1, r2,
#         theta_0,
#         max_iter=100, convergence_eps=1e-10,
#         fix_mu=False, fix_sigma=False, image=False, header="", grid=True):

#     theta, loss = only_EMP_with_pseudo_value_algorithm(
#         r1, r2, theta_0, N=max_iter, EPS=convergence_eps,
#         fix_mu=fix_mu, fix_sigma=fix_sigma, image=image, header=header, grid=grid)

#     return theta, loss

# def only_fit_model_and_calc_idr(r1, r2,
#                             mu = 1,
#                             sigma = 0.3,
#                             rho = idr.DEFAULT_RHO,
#                             mix = idr.DEFAULT_MIX_PARAM,
#                            max_iter=idr.MAX_ITER_DEFAULT,
#                            convergence_eps=idr.CONVERGENCE_EPS_DEFAULT,
#                            image=False, header="",
#                            fix_mu=False, fix_sigma=False, grid=False):
#     # in theory we would try to find good starting point here,
#     # but for now just set it to something reasonable
#     starting_point = (idr.DEFAULT_MU, idr.DEFAULT_SIGMA, idr.DEFAULT_RHO, idr.DEFAULT_MIX_PARAM)
#     # max_iter = 1000
#     # print(idr.DEFAULT_RHO)
#     # starting_point = (1, 0.3, 0.8, 0.3)
#     if not grid:
#         grid = (len(r1) < 100000)
#     idr.log("Starting point: [%s]"%" ".join("%.2f" % x for x in starting_point))
#     theta, loss = only_estimate_model_params(
#         r1, r2,
#         starting_point,
#         max_iter=max_iter,
#         convergence_eps=convergence_eps,
#         fix_mu=fix_mu, fix_sigma=fix_sigma, image=image, header=header, grid=grid)

#     idr.log("Finished running IDR on the datasets", 'VERBOSE')
#     idr.log("Final parameter values: [%s]"%" ".join("%.2f" % x for x in theta))
#     return theta, loss

#     def py_compute_pseudo_values(ranks, signal_mu, signal_sd, p,
#                                  EPS=DEFAULT_PV_COVERGE_EPS):
#         pseudo_values = []
#         for x in ranks:
#             new_x = float(x+1)/(len(ranks)+1)
#             pseudo_values.append( cdf_i( new_x, signal_mu, signal_sd, p,
#                                          -10, 10, EPS ) )
#         return numpy.array(pseudo_values)


if __name__ == '__main__':
    parser = get_parser()
    # try:
    options = parser.parse_args()
    hmm = IDRHmm(options)
    hmm.infer_reactive_sites()
    # except:
    #     options = parser.parse_args()
    #     print(options)
    #     parser.print_help()
    #     sys.exit(0)
