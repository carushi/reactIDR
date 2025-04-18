# from __init__ import *
import sys
import time
import resource
import os
from reactIDR.forward_backward import *
from scipy.optimize import fminbound, minimize, fmin_l_bfgs_b
from scipy.stats import rankdata
import scipy.linalg
from reactIDR.utility import *
from reactIDR.idr_wrapper_hmm import only_build_rank_vector_23dim, get_concatenated_scores, get_idr_value_23dim, estimate_copula_params

APPROX_GRAD = False
RRNA_MODE = False

def print_mat_one_line(mat, end=''):
    print([[mat[i,j] for j in range(mat.shape[1])] for i in range(mat.shape[0])], end=end)

def write_mat_one_line(mat, end=''):
    return str([[mat[i,j] for j in range(mat.shape[1])] for i in range(mat.shape[0])])

def dot_blacket_to_float(c): # structure 0 acc - 1 canonical, -1 unmappable, -2 any.
    if c == ".":    return 0.0
    elif c == "(" or c == ")":  return 1.0
    elif c == "*":  return -2.0
    elif c == "-":   return -1.0
    else:   return 0.5


def seq_to_float(c):
    if c == 'A' or c == 'a':
        return 0
    elif c == 'C' or c == 'c':
        return 1
    elif c == 'G' or c == 'g':
        return 2
    elif c in {'T', 't', 'U', 'u'}:
        return 3
    else:
        return 4

def convert_to_hidden_dict(dict, func):
    for key in dict:
        dict[key] = list(map(func, dict[key]))
    return dict

def filter_no_annotated_data(seta, hidden):
    seta = [key for key in seta if key == chr(0) or key in hidden]
    for key in list(hidden.keys()):
        if key not in seta:
            hidden.pop(key)
    return seta, hidden

def bound_variables():
    min_vals = [MIN_MU, MIN_SIGMA, MIN_RHO] #MIN_MIX_PARAM
    max_vals = [MAX_MU, MAX_SIGMA, MAX_RHO] #MAX_MIX_PARAM
    max_change = [MAX_MU, MAX_SIGMA, MAX_RHO]
    return min_vals, max_vals, max_change

def lbfgs_qfunc(alpha, *args):
    global APPROX_GRAD
    index, sindex, theta, hmm = args[0], args[1], args[2], args[3]
    theta[index] = float(alpha)
    if APPROX_GRAD:
        # value = -hmm.q_function(sindex, theta)
        update_amount = -hmm.q_function_grad_single_variable(sindex, index)
        return update_amount
    else:
        value = -hmm.q_function(sindex, theta)
        update_amount = -hmm.q_function_grad_single_variable(sindex, index)
        return value, np.array([update_amount])
    # value = -hmm.q_function(sindex, theta)
    # update_amount = -hmm.q_function_grad()[sindex][index]
    # return value, update_amount # value and gradient

def clip_params(pindex, pthres, mthres):
    if pindex == 2 or pindex == 3:
        mthres = max(0.0, mthres)
        pthres = min(1.0, pthres)
    if pindex == 1:
        mthres = max(EPS, mthres)
    return pthres, mthres

def list_length(L):
    if isinstance(L, (list, np.ndarray)):
        if len(L) > 0 and isinstance(L[0], (list, np.ndarray)):
            return [list_length(x) for x in L]
        else:
            return len(L)
    return 0

def get_target_transcript(data, sample_size, debug):
    global RRNA_MODE
    if RRNA_MODE: # Debug mode
        # seta = set([chr(0), "RNA18S5+", "RNA28S5+", "RNA5-8S5+", "ENSG00000201321|ENST00000364451+"])
        # seta = set(["RNA18S5+", "RNA28S5+", "RNA5-8S5+", "ENSG00000201321|ENST00000364451+", \
        #                     "RNA18S5-", "RNA28S5-", "RNA5-8S5-", "ENSG00000201321|ENST00000364451-"])
        seta = set(["RNA18S5+", "RNA28S5+", "RNA5-8S5+", "ENSG00000201321+", \
                            "RNA18S5-", "RNA28S5-", "RNA5-8S5-", "ENSG00000201321-"])
        seta = [x for x in seta if x in data[0][0].keys()]
    else:
        if len(data) == 1:
            seta = data[0][0].keys()
        else:
            seta = common_transcript(data[0][0], data[1][0], data[2][0] if len(data) == 3 else None)
        if sample_size > 0:
            seta = random.sample(seta, min(len(seta), sample_size))
    return set([chr(0)]+list(seta))

def truncate_transcript(rep, max_len, skip_start, skip_end):
    if max_len > 0:
        return rep[0:max_len]
    else:
        if len(rep) <= 1: # chr(0) hidden ([]), data ([0])
            return []
        elif max(skip_start, 0)+max(skip_end, 0) > len(rep):
            return [0]
        else:
            return rep[max(skip_start, 0):(len(rep)-max(skip_end, 0))]

def truncate_and_concatenate_score(seta, single_set, max_len, skip_start, skip_end):
    return np.asarray([ item for sublist in [single_set[i] for i in seta if i in single_set] \
                             for item in truncate_transcript(sublist, max_len, skip_start, skip_end)])


def get_target_rankdata(data, seta, max_len, skip_start, skip_end):
    rankdata = [ [] for i in range(len(data))]
    for i, x in enumerate(data):
        reps = [ truncate_and_concatenate_score(seta, x[j], max_len, skip_start, skip_end) for j in range(len(data[i]))]
        for j in range(len(data[i])):
            reps[j] = np.asarray([min(reps[j]-1)]+reps[j]) # Add minimum value at the first corresponding to chr(0)
        rankdata[i] = only_build_rank_vector_23dim(reps)

    return rankdata

def get_stop_sites(single_set, seta, max_len, skip_start, skip_end, verbose):
    temp = [len(truncate_transcript(single_set[x], max_len, skip_start, skip_end)) if x != chr(0) else 0 for x in seta]
    stop_sites = [sum(temp[0:(i+1)]) for i in range(len(temp))]
    if verbose:
        print("\tLength of each transcript:", temp)
        print("\tUnmappable sites:", stop_sites)
    return stop_sites

def set_to_same_length(seta, data, hidden):
    for key in seta:
        if key == chr(0):   continue
        length = max(len(hidden[key]), max([len(rep[key]) for tdata in data for rep in tdata]))
        if len(hidden[key]) < length:
            print("Warning! Reference is shorter than dataset", key, "sequence", len(hidden[key]), "<", length)
            hidden[key] += [-1]*(length-len(hidden[key]))
        for i in range(len(data)):
            for j in range(len(data[i])):
                if len(data[i][j][key]) < length:
                    print("Warning! Some of dataset is shorter", key, i, j, len(data[i][j][key]), "<", length)
                    data[i][j][key] += [0.]*(length-len(data[i][j][key]))
    return data, hidden

def set_hmm_transcripts(data, sample_size, max_len, skip_start, skip_end, hidden=None, debug = True, verbose=True):
    seta = get_target_transcript(data, sample_size, debug)
    seta = sorted(list(seta))
    if verbose:
        print("Dataset--------")
    hidden_mat = None
    if hidden is not None:
        seta, hidden = filter_no_annotated_data(seta, hidden)
        data, hidden = set_to_same_length(seta, data, hidden)
        hidden_mat = truncate_and_concatenate_score(seta, hidden, max_len, skip_start, skip_end)
        hidden_mat = np.append([0], hidden_mat) # Add unmappable at the first corresponding to chr(0)
        if verbose:
            print("\tHidden class: ", seta)
            print("\tHidden length: ", hidden_mat.shape)
    rankdata = get_target_rankdata(data, seta, max_len, skip_start, skip_end)
    stop_sites = get_stop_sites(data[0][0], seta, max_len, skip_start, skip_end, verbose)
    if hidden_mat is not None:
        for i in stop_sites:    hidden_mat[i] = 0
    length_list = [len(data[0][0][key]) for key in seta[1:]]
    return rankdata, stop_sites, seta, hidden_mat, length_list


class ParamFitHMM:
    """docstring for ParamFitHMM"""
    def __init__(self, hclass, data, sample = -1, param=None, debug = False, idr_output = 'idr_output.csv', ref = '',
                 start = -1, end = 35, max_len = -1, DMS_file="", train=False, iparam=None, oparam=None, core=1,
                 reverse=False, independent=False, idr=True, append=False):
        self.hclass = hclass
        self.append = append
        # self.fb = None
        assert hclass == 2 or hclass == 3, 'not allowed dimension'
        if self.hclass == 2:
            neutral = '0.7 0.3; 0.3 0.7'
            drastic = '0.95 0.05; 0.8 0.2'
            self.init_transition_param = np.matrix(neutral, dtype=float)
            self.transition_param = self.init_transition_param.copy()
        else:
            neutral = '0.5 0.25 0.25; 0.25 0.5 0.25; 0.25 0.25 0.5'
            drastic = '0.9 0.05 0.05; 0.7 0.2 0.1; 0.7 0.1 0.2'
            self.init_transition_param = np.matrix(neutral, dtype=float)
            self.transition_param = self.init_transition_param.copy()
        if type(param) == type(None):
            param = (1, 0.2, 0.8, 0.2) # mu, sigma, rho, pi
        assert len(param) > 0, 'len(param) > 0'
        if len(param) == 4 or isinstance(param[0], float):
            self.params = []
            for i in range(hclass-1):
                self.params.append(param.copy())
        else:
            self.params = param
        # self.skip_start, self.skip_end, self.max_len = -1, 35, -1 # Truncate from start and end for each transcript.
        self.skip_start, self.skip_end, self.max_len = start, end, max_len
        self.train = train
        self.ref = ref
        self.max_len_trans = 1000000
        self.verbose = True
        self.idr_output = idr_output
        self.DMS_file = DMS_file
        self.cond_name = ['case', 'cont']
        if reverse:
            self.cond_name = self.cond_name[::-1]
        self.iparam = iparam
        self.oparam = oparam
        self.independent = independent
        self.set_dataset_and_hidden_class(data, sample, debug, core, reverse)
        self.default_param_file = "final_param.txt"
        self.time = None
        self.core = core # multi core processes
        self.idr = idr


    def set_dataset_and_hidden_class(self, data, sample, debug, core, reverse):
        hidden = None
        if self.train:
            hidden = self.allocate_struct_based_hclass(self.ref, reverse)
        if len(self.DMS_file) > 0:
            assert not reverse, 'not reverse'
            hidden = self.allocate_seq_based_hclass(self.DMS_file, hidden)
        if hidden is not None:
            hidden[chr(0)] = []
        elif self.verbose:
            print('No reference information for unsupervised learning')
        self.v, self.stop_sites, self.keys, hidden, self.length_list = set_hmm_transcripts(data, sample, self.max_len, self.skip_start, self.skip_end, hidden, debug)
        if self.verbose:
            print('Transcript--')
            print(self.keys)
        self.length = len(self.v[0][0])
        self.HMM = HMM(self.hclass, self.v, self.stop_sites, hidden, core, self.independent)
        self.init_result_file(data)

    def set_IDR_params(self, index, theta):
        self.params[index] = theta.copy()

    def set_IDR_param(self, index, pos, theta):
        self.params[index][pos] = theta

    def get_IDR_params(self, index = -1):
        if index < 0:
            return self.params
        return self.params[index].copy()

    def get_pseudo_value(self, index):
        assert index+1 < self.hclass, 'index+1 < self.hclass'
        return self.HMM.pseudo_value[index]

    def normalized_prob(self, vec):
        vec = np.asarray(vec).reshape(-1)
        if len([x for x in vec if x < 0.]) == len(vec):
            vec = -vec
        count = [np.real(i) if i > EPS else EPS for i in vec]
        return [c/float(sum(count)) for c in count]

    def set_new_transition(self, head = ''):
        for h in range(self.hclass):
            count = [sum([self.HMM.responsibility_transition(i, h, k) for i in range(self.length)]) for k in range(self.hclass)]
            self.transition_param[h,:] = self.normalized_prob(count)
        if self.verbose:
            print('Set new_transition ->', end='\t')
            print(head, end='\t')
            print_mat_one_line(self.transition_param, '\n')

    def set_new_p_eigen(self, value, vector):
        index = [i for i in range(len(value)) if abs(value[i]-1.0) < EPS]
        if len(index) == 0:
            print(self.transition_param)
            sys.exit('Transition probability matrix error!')
        print(value)
        print(vector)
        vector = self.normalized_prob(i, 3, vector[:,index[0]])
        for i in range(len(self.params)):
            self.set_IDR_param(i, 3, vector[i+1])

    def set_new_p(self, first=False):
        value, vector = scipy.linalg.eig(self.transition_param)
        vector = np.matrix(vector)
        N = self.length-1
        if self.verbose:
            np.set_printoptions(linewidth=200)
            print('computed eigenvector', value, end="")
            print_mat_one_line(vector, '\n')
        # if N > self.max_len_trans: # get even probability!
        #   self.set_new_p_eigen(value, vector, index)
        u, u_inv = vector, np.linalg.inv(vector)
        An = self.transition_param[0,].A1
        # An = np.sum([(u*np.diag(value**n)*u_inv)[0] for n in range(N)], axis=0)
        for n in range(N):
            Dn = np.diag(value**n)
            An1 = An+(u*Dn*u_inv)[0,:].A1
            if max([abs(x-y) for x,y in zip(An1/sum(An1), An/sum(An))]) < EPS:
                break
            An = An1
        An = self.normalized_prob(An)
        for i in range(self.hclass-1):
            self.set_IDR_param(i, 3, An[i+1])
        if self.verbose:
            print('Set new_p ->', self.params, sep='\t')
        self.HMM.set_pseudo_value(-1)

    def EM_CA_step(self, sindex, theta, index, min_val, max_val):
        def f(alpha):
            inner_theta = theta.copy()
            inner_theta[index] = theta[index] + alpha
            q = -self.HMM.q_function_const(sindex, inner_theta)
            return q
        min_step_size = min_val - theta[index]
        max_step_size = max_val - theta[index]
        assert theta[index] >= min_val and theta[index] <= max_val, 'theta[index] precedes the limit'
        alpha = fminbound(f, min_step_size, max_step_size)
        prev_lhd, new_lhd = -f(0), -f(alpha)
        print("CA step -> ", new_lhd, prev_lhd, alpha, theta)
        return alpha, new_lhd

    def EM_CA_iteration(self, sindex, prev_theta, prev_lhd):
        min_vals, max_vals, _ = bound_variables()
        update_amount = [0., 0., 0., 0.]
        new_lhd = [0., 0., 0., 0.]
        for index, (min_val, max_val) in enumerate(zip(min_vals, max_vals)):
            theta = prev_theta.copy()
            alpha, lhd = self.EM_CA_step(sindex, theta, index, min_val, max_val)
            update_amount[index] = alpha
            new_lhd[index] = lhd
        return update_amount, new_lhd

    def EM_step(self, sindex, init_theta, init_lhd, fix_mu, fix_sigma, eps, alpha, update_amount, new_lhd = [0.0]):
        prev_theta, prev_lhd = init_theta, init_lhd
        min_vals, max_vals, max_change = bound_variables()
        for index, (min_val, max_val) in enumerate(zip(min_vals, max_vals)):
            if index == 0 and fix_mu: continue
            if index == 1 and fix_sigma: continue
            theta = prev_theta.copy()
            # theta[index] = theta[index]+alpha*(min(max(min_vals[index], update_amount[index]), max_vals[index]))
            change = np.sign(update_amount[index])*alpha*min(max_change[index], abs(update_amount[index]))
            theta[index] = theta[index]+change
            if len(new_lhd) < 3 or new_lhd[index] + eps >= prev_lhd:
                theta, changed_params = clip_model_params(theta)
                prev_theta = theta
        sys.stdout.flush()
        return prev_theta, max(new_lhd)

    def EM_LBFGS_step(self, sindex, init_theta, init_lhd, fix_mu, fix_sigma, eps, new_lhd = [0.0]):
        global APPROX_GRAD
        prev_theta, prev_lhd = init_theta, init_lhd
        min_vals, max_vals, max_change = bound_variables()
        for index, (min_val, max_val) in enumerate(zip(min_vals, max_vals)):
            if index == 0 and fix_mu: continue
            if index == 1 and fix_sigma: continue
            theta = prev_theta.copy()
            #if self.verbose:
            #    print('\tStart lbfgs-b', self.cond_name[sindex], ['mu', 'sigma', 'rho', 'q'][index], theta[index])
            alpha = fmin_l_bfgs_b(lbfgs_qfunc, theta[index], approx_grad=APPROX_GRAD, args=(index, sindex, theta, self.HMM), bounds=[(min_vals[index], max_vals[index])], factr=eps)
            # alpha = fmin_l_bfgs_b(lbfgs_qfunc, theta[index], args=(index, sindex, theta, self.HMM), bounds=[(min_vals[index], max_vals[index])], factr=eps)
            #if self.verbose:
            #   print('\tEnd lbfgs-b', self.cond_name[sindex], ['mu', 'sigma', 'rho', 'q'][index], alpha)
            #    sys.stdout.flush()
            theta[index] = float(alpha[0][0])
            if len(new_lhd) < 3 or new_lhd[index] + eps >= prev_lhd:
                theta, changed_params = clip_model_params(theta)
                prev_theta = theta
        sys.stdout.flush()
        return prev_theta, max(new_lhd)

    def EM_iteration_grad(self, iter_count, N, lhd, fix_mu, fix_sigma, alpha):
        break_flag = False
        # update_amount = self.HMM.q_function_grad()
        thetas, pseudo_lhds = [], []
        for j in range(len(self.v)):
            prev_theta = self.get_IDR_params(j)
            # theta, pseudo_lhd = self.EM_step(j, prev_theta, lhd, fix_mu=fix_mu, fix_sigma=fix_sigma, eps=EPS/10., alpha=alpha, update_amount=update_amount[j])
            theta, pseudo_lhd = self.EM_LBFGS_step(j, prev_theta, lhd, fix_mu=fix_mu, fix_sigma=fix_sigma, eps=1e7) # (extremely high precision), or 1e7 for moderate accuracy
            thetas.append(copy.deepcopy(theta))
            pseudo_lhds.append(pseudo_lhd)
        for j in range(len(self.v)):
            prev_theta = self.get_IDR_params(j)
            sum_param_change, mean_pseudo_val_change = self.check_value_change(iter_count, j, prev_theta, thetas[j], pseudo_lhds[j])
            if not (iter_count > N/2. and (sum_param_change < EPS and mean_pseudo_val_change < EPS)):
                pass
            else:
                break_flag = True
        return break_flag

    def EM_iteration_grad_previous(self, iter_count, lhd, fix_mu, fix_sigma, alpha):
        break_flag = True
        for j in range(len(self.v)):
            prev_theta = self.get_IDR_params(j)
            # theta, pseudo_lhd = self.EM_step(j, prev_theta, lhd, fix_mu=fix_mu, fix_sigma=fix_sigma, eps=EPS/10., alpha=alpha, update_amount=update_amount[j])
            theta, pseudo_lhd = self.EM_LBFGS_step(j, prev_theta, lhd, fix_mu=fix_mu, fix_sigma=fix_sigma, eps=EPS/10.)
            sum_param_change, mean_pseudo_val_change = self.check_value_change(iter_count, j, prev_theta, theta, pseudo_lhd)
            if not (iter_count > 5 and (sum_param_change < EPS and mean_pseudo_val_change < EPS)):
                break_flag = False
        return break_flag

    def EM_iteration_numeric(self, iter_count, lhd, fix_mu, fix_sigma, alpha):
        break_flag = True
        prev_lhd = lhd
        for j in range(len(self.v)):
            prev_theta = self.get_IDR_params(j)
            update_amount, new_lhd = self.EM_CA_iteration(sindex, prev_theta, prev_lhd)
            theta, max_new_lhd = self.EM_step(j, prev_theta, lhd, fix_mu=fix_mu, fix_sigma=fix_sigma, eps=EPS/10., alpha=alpha, update_amount=update_amount, new_lhd=new_lhd)
            if self.verbose:
                for i in range(len(new_lhd)):
                    print(['mu', 'sigma', 'rho', 'q'][index], '('+str(new_lhd[index])+'-'+str(prev_lhd)+')', prev_theta[index], '->', theta[index] )
            sum_param_change, mean_pseudo_val_change = self.check_value_change(iter_count, j, prev_theta, theta, max_new_lhd)
            if not (iter_count > 5 and (sum_param_change < EPS and mean_pseudo_val_change < EPS)):
                break_flag = False
        return break_flag

    def print_dataset_for_each_sample(self, index, IDR, head):
        flag = 'a'
        with open(self.idr_output, flag) as f:
            for i in range(0, len(self.stop_sites)-1):
                start, end = self.stop_sites[i], self.stop_sites[i+1]
                if end-start == 1:
                    tIDR = [float('nan')]*self.length_list[i]
                else:
                    tIDR = IDR[start:end]
                    if self.skip_start > 0:
                        tIDR = np.append([float('nan')]*self.skip_start, tIDR)
                    if self.skip_end > 0:
                        tIDR = np.append(tIDR, [float('nan')]*self.skip_end)
                f.write(head+"\t"+self.keys[i+1]+"\t"+self.cond_name[index]+"\t"+";".join([("%.8e" % x) for x in tIDR])+"\n")


    def print_header_for_each_sample(self, index, data, head):
        if index == 0 and not self.append:
            flag = 'w'
        else:
            flag = 'a'
        with open(self.idr_output, flag) as f:
            for i, temp in enumerate(data):
                f.write(head+"\t"+self.keys[i+1]+"\t"+self.cond_name[index]+"\t"+";".join(list(map(str, temp)))+"\n")

    def print_reference_for_each_sample(self, index, ref, head, key):
        flag = 'a'
        with open(self.idr_output, flag) as f:
            f.write(head+"\t"+key+"\t"+self.cond_name[index]+"\t"+";".join([("%.4e" % x) for x in ref])+"\n")

    def write_header_to_file(self):
        for i in range(len(self.v)):
            param = self.get_IDR_params(i)
            offset = max(0, self.skip_start)+max(0, self.skip_end)
            position = [list(range(0, self.stop_sites[i+1]-self.stop_sites[i]+offset)) for i in range(len(self.stop_sites)-1)]
            self.print_header_for_each_sample(i, position, "type")

    def write_idr_value_to_file(self):
        for i in range(len(self.v)):
            param = self.get_IDR_params(i)
            localIDRs, IDR = get_idr_value_23dim(param, self.core, *self.v[i])
            self.print_dataset_for_each_sample(i, IDR, "IDR")

    def write_count_value_to_file(self, data):
        for i in range(len(data)):
            with open(self.idr_output, 'a') as f:
                for key in self.keys[1:]:
                    mean_value = np.mean([[x if x == x else 0.0 for x in temp[key]] for temp in data[i]], axis=0)
                    f.write('count'+"\t"+key+"\t"+self.cond_name[i]+"\t"+";".join([str(x) for x in mean_value])+"\n")

    def write_responsibility_to_file(self, head):
        for i in range(len(self.v)):
            param = self.get_IDR_params(i)
            if self.idr:
                IDR = [1.-self.HMM.responsibility_state(x, i+1) for x in range(self.length)]
            else:
                IDR = [self.HMM.responsibility_state(x, i+1) for x in range(self.length)]
            self.print_dataset_for_each_sample(i, IDR, head)

    def write_reference_to_file(self):
        if len(self.ref) > 0:
            struct_dict = get_struct_dict(self.ref, dot_blacket_to_float)
            for key in self.keys:
                if key == chr(0):   continue
                for i in range(len(self.v)):
                    if self.train:
                        assert key in struct_dict, key+' in struct_dict'
                    if key in struct_dict:
                        self.print_reference_for_each_sample(i, struct_dict[key], "ref", key)

    def hmm_grid_search(self, index):
        if self.core > 1:
            return hmm_grid_search_multi_cores(self.core, *self.v[index])
        else:
            return hmm_grid_search(*self.v[index])

    def set_init_theta(self, grid, noHMM=False, omit_unmappaple=False, fix_mu=False, fix_sigma=False):
        for i in range(len(self.v)):
            gtheta = self.get_IDR_params(i)
            lhd = log_lhd_loss_23dim(gtheta, *self.v[i])
            if not fix_mu and not fix_sigma:
                if grid:
                    gtheta = self.hmm_grid_search(i)
                    lhd = log_lhd_loss_23dim(gtheta, *self.v[i])
            if noHMM:
                if len(self.v[i]) == 3: # replicate
                    gtheta, lhd = estimate_copula_params(*self.v[i], theta_0=gtheta, grid=False, fix_mu=fix_mu, fix_sigma=fix_sigma)
                else:                   # duplicate
                    gtheta, lhd = estimate_copula_params(*self.v[i], theta_0=gtheta, grid=False, fix_mu=fix_mu, fix_sigma=fix_sigma)
            tlhd = log_lhd_loss_23dim(self.get_IDR_params(i), *self.v[i])
            if self.verbose:
                print("Initial: Grid search.", gtheta, lhd, tlhd, '(best_theta,new_lhd,cur_lhd)')
            if tlhd < lhd:
                self.set_IDR_params(i, list(gtheta))

    def apply_forward_backward(self, transition_param = None, params = None):
        if type(transition_param) == type(None):
            transition_param = self.transition_param
        if type(params) == type(None):
            params = self.get_IDR_params()
        return self.HMM.forward_backward(transition_param, params)

    def print_setting(self, header, N, grid, fix_mu, fix_sigma):
        if not self.verbose:    return
        print("Settings--------")
        print("\tHidden class:", self.hclass)
        print("\tMode:", header)
        print("\tRepeat:", N)
        if grid: print("\tGrid search: on")
        if fix_mu: print("\tFix mu: on")
        if fix_sigma: print("\tFix sigma: on")
        print("\tSkip start-end:", self.skip_start, self.skip_end)
        print("\tMax length setting:", self.max_len)
        if self.train:
            print("\tTraining: on")
        if self.independent:
            print("\tIndependent: on")
        print("\tReference file:", self.ref)
        print("\tOutput file:", self.idr_output)
        if len(self.DMS_file) > 0:
            print("\tDMS setting: on (", self.DMS_file, ")")
        print("First copula parameters:")
        print(self.params, sep="")
        print("Default transition_param:")
        print_mat_one_line(self.transition_param, '\n')
        sys.stdout.flush()

    def print_result(self):
        if self.verbose:
            print("HMM: print result.")
            self.HMM.print_result()

    def write_params(self, prefix=''):
        fname = self.oparam
        if fname is None:
            fname = self.default_param_file
        dir, base = os.path.dirname(fname), os.path.basename(fname)
        if len(self.keys) == 2 and not self.train:
            fname = os.path.join(dir, prefix + self.keys[1] + '_' + base)
        else:
            fname = os.path.join(dir, prefix + fname)
        if self.verbose:
            print("Write to param file: ", fname)
        with open(fname, 'w') as f:
            f.write(write_mat_one_line(self.transition_param))
            f.write("\n")
            f.write(str(self.params))
            f.write("\n")

    def read_params(self):
        fname = self.iparam
        with open(fname) as f:
            lines = f.readlines()
            try:
                self.transition_param = np.matrix(eval(lines[0]), dtype=float)
                self.params = eval(lines[1].rstrip('\n'))
                assert self.transition_param.shape == (self.hclass, self.hclass), 'transition_shape inconsistency'
                assert len(self.params) >= 1 and all([len(p) == 4 for p in self.params]), 'param_shape inconsistency'
                if self.verbose:
                    print("Read from param file: ", fname)
            except:
                if self.verbose:
                    print("Param file error: ", fname)

    def check_value_change(self, iindex, index, prev_theta, theta, new_lhd):
        prev_z_list = self.get_pseudo_value(index)
        self.set_IDR_params(index, theta)
        self.HMM.set_pseudo_value(index)

        z_list = self.get_pseudo_value(index)
        mean_pseudo_val_change = sum([np.mean([np.abs(p-z) for p, z in zip(prev_z_list, z_list)])])
        sum_param_change = sum([np.abs(x-y) for x, y in zip(theta, prev_theta) ]) #np.abs(theta - prev_theta).sum()
        if self.verbose:
            print(("Iter %i" % iindex), ("Dataset %i" % index),
                "%.2e" % sum_param_change,
                "%.2e" % mean_pseudo_val_change,
                "%.4e" % new_lhd,
                theta, )
            print('Set new_theta', theta)
        return sum_param_change, mean_pseudo_val_change

    def init_result_file(self, data):
        self.write_header_to_file()
        self.write_count_value_to_file(data)

    def write_first_result(self):
        self.write_reference_to_file()
        self.write_idr_value_to_file()

    def parameter_optimization_iter(self, index, N, space, fix_mu=False, fix_sigma=False, fix_trans=False):
        if not fix_trans:
            self.set_new_transition(str(index))
            self.set_new_p(first=(index == 0))
        lhd = self.apply_forward_backward()
        if self.verbose:
            print('(%d time) new lhd ->\t%f' % (index, lhd), end='\t')
            print(index, -1, self.get_IDR_params(), end='\t', sep='\t')
            print_mat_one_line(self.transition_param, '\n')
            # print('alpha->', 10**space[index], sep="\t")
        break_flag = self.EM_iteration_grad(index, N, lhd, fix_mu, fix_sigma, 10**space[index])
        # break_flag = self.EM_iteration_numeric(i, lhd, fix_mu, fix_sigma, space[i]) # Numerical differentiation.
        return lhd, break_flag

    def allocate_seq_based_hclass(self, ref, hidden):
        if self.verbose:
            print('Set hidden class for (semi) supervised learning: Sequence based')
        struct_dict = get_struct_dict(ref, seq_to_float)
        if hidden is None:
            if self.hclass == 3:
                hidden = convert_to_hidden_dict(struct_dict, lambda x: -1 if (x == 0 or x == 1) else 0)
            else:
                hidden = convert_to_hidden_dict(struct_dict, lambda x: 1 if (x == 0 or x == 1) else 0)
        else:
            hidden_seq = convert_to_hidden_dict(struct_dict, lambda x: 1 if (x == 0 or x == 1) else 0)
            print(hidden_seq)
            for x in hidden:
                if x not in hidden_seq: hidden.pop(x)
                else:
                    hidden[x] = [int(i*j) for i, j in zip(hidden[x], hidden_seq[x])]
        return hidden

    def allocate_struct_based_hclass(self, ref, reverse):
        if self.verbose:
            print('Set hidden class for (semi) supervised learning: Structure based')
        struct_dict = get_struct_dict(ref, dot_blacket_to_float)
        if self.hclass == 3:
            hidden = convert_to_hidden_dict(struct_dict, lambda x: 2 if x > 0.0 else 0 if x == -1.0 else -1 if x == -2.0 else 1)
            if reverse:
                hidden = [[0, 2, 1, -1][x] for x in hidden]
        else:
            if reverse:
                hidden = convert_to_hidden_dict(struct_dict, lambda x: 1 if x > 0.0 else 0 if x == -1.0 else -1 if x == -2.0 else 0)
            else:
                hidden = convert_to_hidden_dict(struct_dict, lambda x: 0 if x > 0.0 else 0 if x == -1.0 else -1 if x == -2.0 else 1)
        return hidden

    def train_hmm_EMP(self, grid=False, N=100, EPS=1e-4, fix_mu=False, fix_sigma=False, fix_trans=False):
        self.estimate_hmm_based_IDR(grid, N, EPS, fix_mu, fix_sigma, fix_trans)

    def test_hmm_EMP(self, grid=False, N=100, EPS=1e-4, fix_mu=False, fix_sigma=False, fix_trans=False):
        if self.iparam is not None:
            self.read_params()
        self.estimate_hmm_based_IDR(grid, 0, EPS, fix_mu=fix_mu, fix_sigma=fix_sigma, fix_trans=fix_trans, test=True)

    def fit_hmm_EMP(self, grid=False, N=100, EPS=1e-4, fix_mu=False, fix_sigma=False, fix_trans=False):
        if self.iparam is not None:
            self.read_params()
        self.estimate_hmm_based_IDR(grid, N, EPS, fix_mu=fix_mu, fix_sigma=fix_sigma, fix_trans=fix_trans, test=True, prefix='fit_')

    def estimate_global_IDR(self, grid=False, fix_mu=False, fix_sigma=False, fix_trans=False):
        self.set_init_theta(grid, noHMM=True, omit_unmappaple=True, fix_mu=fix_mu, fix_sigma=fix_sigma)
        self.estimate_hmm_based_IDR(grid, N=-1, fix_mu=fix_mu, fix_sigma=fix_sigma, fix_trans=fix_trans)

    def estimate_hmm_based_IDR_debug(self, grid=False, N=100, EPS=1e-4, fix_mu=False, fix_sigma=False, fix_trans=False):
        self.estimate_hmm_based_IDR(grid, N, EPS, fix_mu, fix_sigma, fix_trans, debug=True)

    def print_time(self, head):
        if self.time is None:
            self.time = [time.time(), time.time()]
            print("Time check:", head, self.time[0], self.time[1], '(process,time,clock)')
        else:
            current_time = time.time()
            current_clock = time.time()
            print("Time check:", head, current_time - self.time[0], current_clock-self.time[1], '(process,time,clock)')
            self.time = [current_time, current_clock]
            if head == 'global' or head == 'last_iter':
                print("Memory check:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "(bytes)")

    def estimate_hmm_based_IDR(self, grid=False, N=100, EPS=1e-4, fix_mu=False, fix_sigma=False, fix_trans=False, debug=False, test=False, prefix=''):
        """ N=-1: no forward-backward (IDR computation). N=0 -> forward-backward once with trained parameters."""
        self.print_time('start')
        self.print_setting("IDR-HMM", N, grid, fix_mu, fix_sigma)
        if N > 0 and grid:
            self.set_init_theta(grid, fix_mu=fix_mu, fix_sigma=fix_sigma)
        self.write_first_result()
        self.print_time('global')
        if N < 0:   return
        lhd = self.apply_forward_backward()
        if self.verbose:
            print('--------')
            print('(initial) new lhd ->\t%f ' % (lhd), end='\t')
            print(0, -1, self.get_IDR_params(), end=' ', sep='\t')
        print_mat_one_line(self.transition_param, '\n')
        space = np.linspace(-1., np.log10(EPS), N)
        break_flag = False
        for i in range(N):
            if self.verbose:
                print('--------')
            try:
                lhd, break_flag = self.parameter_optimization_iter(i, N, space, fix_mu, fix_sigma, fix_trans)
            except:
                print('Unexpected error', sys.exc_info())
            if debug:
                self.write_responsibility_to_file("IDR-HMM-"+str(i))
            self.print_time(str(i)+"_iter")
            if break_flag:
                break
        if self.verbose:
            if debug: self.print_result()
            print('--------')
        self.write_responsibility_to_file("IDR-HMM-final")
        if N > 0:
            if self.train:
                self.write_params()
            else:
                self.write_params('learn_' if len(prefix) == 0 else prefix)
        self.print_time("last_iter")
