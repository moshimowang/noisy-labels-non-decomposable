import numpy as np
from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_no_a
from standard_funcs.helpers import compute_fmeasure, get_params_fmeasure


def bisection(X_train, y_train, vec_eta, T, n_class):
    A, B = get_params_fmeasure(n_class)
    clf = np.zeros(shape=(n_class, n_class))

    for i in range(n_class):
        for j in range(n_class):
            if(i != j):
                clf[i][j] = 1

    C_t = get_confusion_matrix_from_loss_no_a(clf, X_train, y_train, vec_eta, n_class)
    alpha, beta = 0, 1

    for t in range(T):
        gamma = (alpha+beta)/2        
        L = -(A - gamma*B)
        # print(L)
        g_t = get_confusion_matrix_from_loss_no_a(L, X_train, y_train, vec_eta, n_class)
        
        if 1 - compute_fmeasure(g_t) >= gamma:
            alpha = gamma
            clf = L
        else:
            beta = gamma

    return clf


def bisection_noisy(X_train, y_train, vec_eta, T, n_class, noise_T_inv):
    '''
    noise_T_inv is the inverse of the column-stochastic noise matrix.
    '''

    A, B = get_params_fmeasure(n_class)
    clf = np.zeros(shape=(n_class, n_class))

    for i in range(n_class):
        for j in range(n_class):
            if(i != j):
                clf[i][j] = 1

    C_t = get_confusion_matrix_from_loss_no_a(clf, X_train, y_train, vec_eta, n_class)
    alpha, beta = 0, 1

    for t in range(T):
        gamma = (alpha+beta)/2        
        L = - noise_T_inv.T @ (A - gamma*B)
        # print(L)
        g_t = get_confusion_matrix_from_loss_no_a(L, X_train, y_train, vec_eta, n_class)

        corrected_C_t = noise_T_inv @ g_t
        corrected_C_t[corrected_C_t<0] = 0
        corrected_C_t[corrected_C_t>1] = 1
        corrected_C_t = corrected_C_t / corrected_C_t.sum()
        
        if 1 - compute_fmeasure(corrected_C_t) >= gamma:
            alpha = gamma
            clf = L
        else:
            beta = gamma

    return clf


def corrected_fmeasure(A, B, noise_T_inv, noisy_conf):
    val = np.sum( np.multiply((noise_T_inv.T @ A), noisy_conf) ) / np.sum( np.multiply((noise_T_inv.T @ B), noisy_conf) )
    return 1 - val
