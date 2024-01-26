import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from algorithm.bisection import bisection_noisy, bisection
from models.logistic_regression import get_vec_eta
from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_no_a
from standard_funcs.helpers import compute_fmeasure

import sys


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
                clf = bisection(X_train, y_train, vec_eta, 200, n_class)
            else:
                clf = bisection_noisy(X_train, y_train, vec_eta, 200, n_class, noise_T_inv)

            ### Evaluate Performance on Test Data
            test_conf = get_confusion_matrix_from_loss_no_a(clf, X_test, y_test_clean, vec_eta, n_class)
            test_score = 1 - compute_fmeasure(test_conf)

            test_scores.append(test_score)

        mu_test = 1 - round(np.mean(test_scores), 5)
        std_test = round(np.std(test_scores)/np.sqrt(len(test_scores)), 5)
        
        print(str(mu_test) + " (" + str(std_test) + ")")

        # if run_original:
        #     np.save(dir_path+"results/"+d+"_noisy_"+str(noise_idx)+"_F_results_original.npy", np.array([mu_test, std_test]), allow_pickle=True)
        # else:
        #     np.save(dir_path+"results/"+d+"_noisy_"+str(noise_idx)+"_F_results_NC.npy", np.array([mu_test, std_test]), allow_pickle=True)
