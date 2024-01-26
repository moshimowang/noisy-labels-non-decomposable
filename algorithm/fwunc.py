import numpy as np
from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_no_a
from solvers.fwpost import frank_wolfe_post


def frank_wolfe_unc(X_train, y_train, vec_eta, T, n_class, grad_func):
    Cs = [np.ones(shape = (n_class, n_class))*(1/n_class**2)]
    bgammas = []
    clfs = []

    p = np.zeros((n_class,))
    for i in range(n_class):
        p[i] = (y_train == i).mean()

    for t in range(T):
        C_t = Cs[-1]
        L_t = grad_func(C_t, n_class, p)
        bgamma_t = get_confusion_matrix_from_loss_no_a(L_t, X_train, y_train, vec_eta, n_class) 
        C_t_new = C_t*(1 - (2/(t+2))) + (2/(t+2))*bgamma_t
        Cs.append(C_t_new)
        bgammas.append(bgamma_t)
        clfs.append(L_t)

    return (clfs, frank_wolfe_post(Cs[1:], n_class, p, grad_func))


def frank_wolfe_unc_noisy(X_train, y_train, vec_eta, T, n_class, grad_func, noise_T_inv, noise_T=None):
    '''
    noise_T_inv is the inverse of the column-stochastic noise matrix.
    '''

    Cs = [np.ones(shape = (n_class, n_class))*(1/n_class**2)]
    bgammas = []
    clfs = []

    p = np.zeros((n_class,))
    for i in range(n_class):
        p[i] = (y_train == i).mean()

    for t in range(T):
        C_t = Cs[-1]

        corrected_C_t = noise_T_inv @ C_t
        corrected_C_t[corrected_C_t<0] = 0
        corrected_C_t[corrected_C_t>1] = 1
        corrected_C_t = corrected_C_t / corrected_C_t.sum()

        L_t = noise_T_inv.T @ grad_func(corrected_C_t, n_class, p)
        bgamma_t = get_confusion_matrix_from_loss_no_a(L_t, X_train, y_train, vec_eta, n_class) 
        C_t_new = C_t*(1 - (2/(t+2))) + (2/(t+2))*bgamma_t
        Cs.append(C_t_new)
        bgammas.append(bgamma_t)
        clfs.append(L_t)

    return (clfs, frank_wolfe_post(Cs[1:], n_class, p, grad_func))
