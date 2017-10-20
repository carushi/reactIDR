from sklearn import metrics
import math
import numpy as np
import sys

def log_count(x):
    if x != "nan":
        return math.log10(float(x)+1.)
    else:
        return 0.

def nan_float(x):
    if x != "nan" and x != 'NA' and x != 'None':
        return float(x)
    else:
        return float("nan")

def one_minus_nan_float(x):
    if x != 'nan' and x != 'NA' and x != 'None':
        return 1.0-float(x)
    else:
        return float('nan')

def set_positive_negative(answer, sp):
    def get_all(x):
        if float(x) > 0.0:  return 1
        else:   return 0
    def get_canonical(x):
        if float(x) == 1.0: return 1
        else:   return 0
    if sp == "all":
        return list(map(get_all, answer))
    else:
        return list(map(get_canonical, answer))


# def calc_tp_fp(answer, pred, pos, na_rm=True, verbose=True):
#     if na_rm:
#         index = [j for j in range(len(answer)) if pred[j] == pred[j]]
#     tup = [(pred[i], answer[i]) for i in index if answer[i] >= 0.0]
#     tup = sorted(tup, key=lambda x: x[0], reverse=True)
#     p = len([x for x in answer if int(x) == pos])
#     n = len(answer)-p
#     fpr, tpr = [0], [0]
#     tp, fp = 0, 0
#     for i, (x, y) in enumerate(tup):
#         if y == pos: tp += 1
#         else: fp += 1
#         fpr.append(float(fp)/n)
#         tpr.append(float(tp)/p)
#     # fpr.append(1)
#     # tpr.append(1)
#     if verbose:
#         print("tp", p, "fp", n)
#     auc = metrics.auc(fpr, tpr)
#     return tpr, fpr, auc

def calc_tp_fp(answer, pred, pos, verbose=True, narm=True, negative=False, assym=False):
    assert len(pred) == len(answer)
    if assym:
        converted_p, answer = pred_conversion_assym(pred, narm, negative, pos, answer)
    else:
        converted_p, answer = pred_conversion(pred, narm, negative, pos, answer)
    fpr, tpr, threshold = metrics.roc_curve(np.array(answer), np.array(converted_p), pos)
    if len(fpr) >= 2:
        auc = metrics.auc(fpr, tpr)
    else:
        auc = float('nan')
    return tpr, fpr, auc

def pred_conversion(pred, narm, negative, pos, answer):
    STEP = 1e10
    converted = pred.copy()
    if negative:
        converted = [-x for x in converted]
    if narm:
        # answer = [answer[i] for i in range(len(answer)) if pred[i] == pred[i]]
        # pred = [pred[i] for i in range(len(pred)) if pred[i] == pred[i]]
        inf = ("-1e20" if pos == int(0) else "1e20")
        converted = [x if x == x else float(inf) for x in converted]
    converted = [x*STEP for x in converted]
    return converted, answer

def pred_conversion_assym(pred, narm, negative, pos, answer):
    converted = [x*100. for x in pred]
    if negative:
        converted = [-x for x in converted]
    if narm:
        inf = min(min([x for x in converted if x == x])-1, -1e5)
        converted = [x if x == x else float(inf) for x in converted]
    return converted, answer

def calc_precision_recall(answer, pred, pos, verbose=True, narm=True, negative=False, assym=False):
    assert len(pred) == len(answer)
    if assym:
        print('nan number', len([x for x in pred if x != x]))
        converted_p, answer = pred_conversion_assym(pred, narm, negative, pos, answer)
    else:
        converted_p, answer = pred_conversion(pred, narm, negative, pos, answer)
    precision, recall, thresholds = metrics.precision_recall_curve(answer, converted_p, pos_label=pos)
    # print("precision", precision, file=sys.stderr)
    # print("recall", recall, file=sys.stderr)
    # print("thresholds", "\t".join(list(map(str, thresholds))), file=sys.stderr)
    auc = metrics.auc(recall, precision)
    return precision, recall, auc

def filter_ambig_answer(pos, score, IDR_case, IDR_cont, th_case, th_cont, acc=True):
    if IDR_cont is None:
        if acc:
            return [pos if score[i] == pos and IDR_case[i] >= th_case else 1-pos for i in range(len(score))]
        else:
            return [pos if score[i] == pos and IDR_case[i] < th_case else 1-pos for i in range(len(score))]
    else:
        if acc:
            return [pos if score[i] == pos and IDR_case[i] >= th_case and IDR_cont[i] < th_cont else 1-pos for i in range(len(score))]
        else:
            return [pos if score[i] == pos and IDR_case[i] < th_case and IDR_cont[i] >= th_cont else 1-pos for i in range(len(score))]

def compute_ratio(IDR_case, IDR_cont, IDR=True, acc=True):
    if IDR:
        IDR_case = [one_minus_nan_float(x) for x in IDR_case]
        IDR_cont = [one_minus_nan_float(x) for x in IDR_cont]
        EPS = 0.00001
        if acc:
            return [math.log(case+EPS) - math.log(cont+EPS) if case != float('nan') and cont != float('nan') else float('nan') for case, cont in zip(IDR_case, IDR_cont)]
        else:
            return [math.log(cont+EPS) - math.log(case+EPS) if case != float('nan') and cont != float('nan') else float('nan') for case, cont in zip(IDR_case, IDR_cont)]
    else:
        if acc:
            return [float(case) - float(cont) if case != float('nan') and cont != float('nan') else float('nan') for case, cont in zip(IDR_case, IDR_cont)]
        else:
            return [float(cont) - float(case) if case != float('nan') and cont != float('nan') else float('nan') for case, cont in zip(IDR_case, IDR_cont)]

def uniq_threshold(IDR):
    step = 50
    temp = sorted(IDR)[::step]
    prev = float('nan')
    threshold_list = []
    for th in temp:
        if prev == th:
            continue
        threshold_list.append(th)
        prev = th
    return threshold_list

def grid_search_auc(pos, answer, pred, IDR_case, IDR_cont, IDR=True, acc=True):
    if IDR:
        IDR_case = [one_minus_nan_float(x) for x in IDR_case]
        IDR_cont = [one_minus_nan_float(x) for x in IDR_cont]
    else:
        IDR_case = [float(x) for x in IDR_case]
        IDR_cont = [float(x) for x in IDR_cont]
    X, Y = uniq_threshold(IDR_case), uniq_threshold(IDR_cont)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros(shape=X.shape, dtype=float)
    sample_size = np.zeros(shape=X.shape, dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            th_case, th_cont = X[i,j], Y[i,j]
            tanswer = filter_ambig_answer(pos, answer, IDR_case, IDR_cont, th_case, th_cont)
            sample_size[i,j] = len([x for x in tanswer if x == pos])
            if sample_size[i,j] > 0:
                tpr, fpr, auc = calc_tp_fp(tanswer, pred, pos, True, False)
                Z[i,j] = auc
    return sample_size, (X, Y, Z)

def grid_search_auc_1dim(pos, answer, pred, IDR_case, IDR=True, acc=True):
    if IDR:
        IDR_case = [one_minus_nan_float(x) for x in IDR_case]
    else:
        IDR_case = [float(x) for x in IDR_case]
    X = uniq_threshold(IDR_case)
    Z = [0.]*len(X)
    sample_size = [0]*len(X)
    for i in range(len(X)):
        th_case = X[i]
        tanswer = filter_ambig_answer(pos, answer, IDR_case, None, th_case, None)
        sample_size[i] = len([x for x in tanswer if x == pos])
        if sample_size[i] > 0:
            tpr, fpr, auc = calc_tp_fp(tanswer, pred, pos, True, False)
            Z[i] = auc
    return sample_size, (X, Z)
