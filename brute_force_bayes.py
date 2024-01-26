from math import fabs

from idna import valid_contextj
import numpy as np
from scipy.optimize import brute
from standard_funcs.helpers import compute_qmean, compute_fmeasure
from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_no_a
from standard_funcs.helpers import get_params_fmeasure

def bayes(X_train, y_train, vec_eta, n_class):

    def objective_func(l_arr):
        L_matrix = -1*np.array([[l_arr[0], 0, 0], [0, l_arr[1], 0], [0, 0, l_arr[2]]])
        #print(L_matrix)
        cm = get_confusion_matrix_from_loss_no_a(L_matrix, X_train, y_train, vec_eta, n_class)
        val = compute_qmean(cm)
        #print(val)
        #assert False
        return val

    res = brute(objective_func, ranges=[(-10, 10), (-10, 10), (-10, 10)], full_output=True)

    print("Bayes brute force coodinates: ", res[0])

    return res[1] # function value at global minimum


def bayes_f(X_train, y_train, vec_eta, n_class):
    A, B = get_params_fmeasure(n_class)

    def objective_func(gamma):
        L = -(A - gamma[0]*B)

        cm = get_confusion_matrix_from_loss_no_a(L, X_train, y_train, vec_eta, n_class)
        val = compute_fmeasure(cm)

        return val

    res = brute(objective_func, ranges=[slice(0,1,0.05)], full_output=True)

    print("Bayes brute force coodinates: ", res[0])

    return res[1] # function value at global minimum
