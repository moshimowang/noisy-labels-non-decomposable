import numpy as np
from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_no_a


def frank_wolfe_post(Cs_custom, n_class, p, grad_func):
    C_t = Cs_custom[0]
    clf_weight = np.zeros(len(Cs_custom))
    clf_weight[0] = 1
    for t in range(100):
        grad = grad_func(C_t, n_class, p)
        # print(grad)
        lin_value = []

        for C in Cs_custom:
            val = np.sum(np.multiply(grad, C))
            if abs(val) > 1e7:
                val = 1e7
            lin_value.append(val) 

        lin_index = np.argmin(lin_value)
        lin_min = Cs_custom[lin_index]
        gamma_t = (2/(t+3))
        C_t_new = C_t*(1 - gamma_t) + gamma_t*lin_min
        lin_index_one_hot = np.zeros(len(Cs_custom))
        lin_index_one_hot[lin_index] = 1

        clf_weight = clf_weight*(1 - gamma_t) + gamma_t*lin_index_one_hot
        C_t = C_t_new

    return clf_weight
