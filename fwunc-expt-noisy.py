import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from algorithm.fwunc import frank_wolfe_unc_noisy, frank_wolfe_unc
from models.logistic_regression import get_vec_eta
from standard_funcs.confusion_matrix import weight_confusion_matrix
from standard_funcs.helpers import compute_hmean, compute_qmean, compute_gmean
from standard_funcs.grads import grad_hmean_original, grad_qmean_original, grad_gmean_original

import sys


PERF_METRIC = sys.argv[2]

if PERF_METRIC == 'h':
    print("running hmean")
    grad_func = grad_hmean_original
    perf_func = compute_hmean
elif PERF_METRIC == 'q':
    print("running qmean")
    grad_func = grad_qmean_original
    perf_func = compute_qmean
elif PERF_METRIC == 'g':
    print("running gmean")
    grad_func = grad_gmean_original
    perf_func = compute_gmean
else:
    assert False


noise_rates = [0.1, 0.2, 0.3, 0.4]
dir_path = "./real_data/"

total_noise_rates = 4

NUM_TRIALS = 5


if int(sys.argv[1]) == 0:
    run_original = True
    print("running original")
else:
    run_original = False
    print("running noisy")


### Data Loading
for d in ["vehicle", "pageblocks", "satimage", "covtype", "abalone"]:
    print(d)

    data_dict = np.load(dir_path + d + "/" + d + "_data.npy", allow_pickle=True).item()
    X_data = data_dict['X']
    Y_data_clean = data_dict['Y']

    ### Number of Classes
    n_class = len(np.unique(Y_data_clean))

    # Load noisy matrices
    Ts = np.zeros((total_noise_rates, n_class, n_class))
    for i in range(total_noise_rates):
        Ts[i] = np.load(dir_path + d + "/C_"+str(i)+".npy", allow_pickle=True)


    for noise_idx in range(total_noise_rates):
        print("noise_idx: ", noise_idx)

        # noise_T_inv is the inverse of the column-stochastic noise matrix
        noise_T = Ts[noise_idx]
        noise_T_inv = np.linalg.inv(Ts[noise_idx])

        noisy_data_dict = np.load(dir_path + d + "/" + d + "_noisy_" + str(noise_idx) + ".npy", allow_pickle=True).item()
        Y_data = noisy_data_dict['Y']

        test_scores = []

        for i in range(NUM_TRIALS):
            X_train, X_test, y_train, y_test, y_train_clean, y_test_clean = train_test_split(X_data, Y_data, Y_data_clean, test_size=0.3, random_state=i)

            ### Number of Classes
            assert(n_class == len(np.unique(y_train)))

            ### Training CPE Model
            vec_eta = get_vec_eta(X_train, y_train)

            ### Getting the Classifiers and Weights
            if run_original:
                clfs, weights = frank_wolfe_unc(X_train, y_train, vec_eta, 5000, n_class, grad_func)
            else:
                clfs, weights = frank_wolfe_unc_noisy(X_train, y_train, vec_eta, 5000, n_class, grad_func, noise_T_inv, noise_T)

            ### Evaluate Performance on Train and Test Data
            test_conf = weight_confusion_matrix(X_test, y_test_clean, clfs, weights, n_class, vec_eta)
            test_score = 1 - perf_func(test_conf)

            # train_scores.append(train_score)
            test_scores.append(test_score)

        mu_test = 1 - round(np.mean(test_scores), 5)
        std_test = round(np.std(test_scores)/np.sqrt(len(test_scores)), 5)
        
        print(str(mu_test) + " (" + str(std_test) + ")")

        # if run_original:
        #     if PERF_METRIC == 'h':
        #         np.save(dir_path+"results/"+d+"_noisy_"+str(noise_idx)+"_H_results_original.npy", np.array([mu_test, std_test]), allow_pickle=True)
        #     elif PERF_METRIC == 'q':
        #         np.save(dir_path+"results/"+d+"_noisy_"+str(noise_idx)+"_Q_results_original.npy", np.array([mu_test, std_test]), allow_pickle=True)
        #     elif PERF_METRIC == 'g':
        #         np.save(dir_path+"results/"+d+"_noisy_"+str(noise_idx)+"_G_results_original.npy", np.array([mu_test, std_test]), allow_pickle=True)
        #     else:
        #         assert False
        # else:
        #     if PERF_METRIC == 'h':
        #         np.save(dir_path+"results/"+d+"_noisy_"+str(noise_idx)+"_H_results_NC.npy", np.array([mu_test, std_test]), allow_pickle=True)
        #     elif PERF_METRIC == 'q':
        #         np.save(dir_path+"results/"+d+"_noisy_"+str(noise_idx)+"_Q_results_NC.npy", np.array([mu_test, std_test]), allow_pickle=True)
        #     elif PERF_METRIC == 'g':
        #         np.save(dir_path+"results/"+d+"_noisy_"+str(noise_idx)+"_G_results_NC.npy", np.array([mu_test, std_test]), allow_pickle=True)
        #     else:
        #         assert False
