import numpy as np

def get_confusion_matrix(Y_label, Y_pred, n_class):
    confusion_matrix = np.zeros(shape=(n_class, n_class))
    np.add.at(confusion_matrix, (Y_label.astype(int), Y_pred.astype(int)), 1)
    return confusion_matrix/len(Y_label)

def get_confusion_matrix_threshold(threshold, X_train, y_train, vec_eta_1, inequality="leq"):
    confusion_matrix = np.zeros(shape=(2, 2))
    Y_pred = np.zeros(len(y_train))
    Pr_1 = vec_eta_1(X_train)
    if(inequality == "leq"):
        indices_1 = Pr_1 <= threshold
    else:
        indices_1 = Pr_1 >= threshold
    Y_pred[indices_1] = 1
    np.add.at(confusion_matrix, (y_train.astype(int), Y_pred.astype(int)), 1)
    return confusion_matrix/len(X_train)

def get_confusion_matrix_from_loss_a(L_t_array, X_train, y_train, unique_a, protected_indices, vec_eta_1):
    confusion_matrices = []
    
    for i in range(unique_a):
        L_t_a = L_t_array[i]
        confusion_matrix_a = np.zeros(shape=(2, 2))
        Y_pred = np.zeros(len(y_train[protected_indices[i]]))
        Pr_1 = vec_eta_1(X_train[protected_indices[i]])
        Pr_0 = 1 - Pr_1
        Loss_1 = L_t_a[1, 1]*Pr_1 + L_t_a[0, 1]*Pr_0
        Loss_0 = L_t_a[0, 0]*Pr_0 + L_t_a[1, 0]*Pr_1
        indices_1 = Loss_1 <= Loss_0
        Y_pred[indices_1] = 1
        np.add.at(confusion_matrix_a, (y_train[protected_indices[i]].astype(int), Y_pred.astype(int)), 1)
        confusion_matrix_a = confusion_matrix_a/len(Y_pred)
        confusion_matrices.append(confusion_matrix_a)
    
    return np.array(confusion_matrices)

def get_confusion_matrix_from_loss_no_a(L_t, X_train, y_train, vec_eta, n_class):
    confusion_matrix = np.zeros(shape=(n_class, n_class))
    Y_pred = np.zeros(len(y_train))
    Pr_matrix = vec_eta(X_train).T
    ### Loss if predicted class = row i, data_point j
    Loss_matrix = []
    
    for i in range(n_class):
        L_t_vec = L_t[:, i].T
        Loss_i = np.dot(L_t_vec, Pr_matrix)
        Loss_matrix.append(Loss_i)
    Loss_matrix = np.array(Loss_matrix)
    
    Y_pred = np.argmin(Loss_matrix, axis=0)
    np.add.at(confusion_matrix, (y_train.astype(int), Y_pred.astype(int)), 1)
    return confusion_matrix/len(X_train)

def weight_confusion_matrix(X, Y, clfs, weights, n_class, vec_eta):
    if isinstance(weights, str) and weights == "fwupdate":
        conf = np.zeros(shape = (n_class, n_class))
        for t in range(len(clfs)):
            b_gamma_t = get_confusion_matrix_from_loss_no_a(clfs[t], X, Y, vec_eta, n_class)
            conf = conf*(1 - (2/(t+2))) + (2/(t+2))*b_gamma_t
        return conf

    else:
        confs = []
        for i in range(len(clfs)):
            if i % 1000 == 0:
                print("weighting confusion matrix: ", i, " out of ", len(clfs))
            confs.append(get_confusion_matrix_from_loss_no_a(clfs[i], X, Y, vec_eta, n_class))
    
        return np.average(confs, weights=weights, axis=0)
