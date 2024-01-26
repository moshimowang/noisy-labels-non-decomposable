import numpy as np
from algorithm.fwunc import frank_wolfe_unc
from algorithm.bisection import bisection
from models.logistic_regression import get_vec_eta
from standard_funcs.confusion_matrix import weight_confusion_matrix, get_confusion_matrix_from_loss_no_a
from standard_funcs.helpers import compute_fmeasure, compute_qmean
import sys
from sklearn.model_selection import train_test_split


run_f = False
run_q = False

if sys.argv[1] == "f":
    print("running f")
    run_f = True
elif sys.argv[1] == "q":
    print("running q")
    run_q = True
else:
    assert False


data_dir = "./synthetic_data/"

f_results_dir = data_dir+"f_results/"
q_results_dir = data_dir+"q_results/"

n_class = 3
num_T = 5
num_trials = 5

T = 5000

X_test = np.load(data_dir+"simulated_X_test.npy", allow_pickle=True)[:100000]
y_test = np.load(data_dir+"simulated_y_test.npy", allow_pickle=True)[:100000]

training_sizes = [100, 1000, 10000, 100000]

for trial_idx in range(num_trials):

    for training_size in training_sizes:
        all_X_train = np.load(data_dir+"simulated_X_train_"+str(training_size)+".npy", allow_pickle=True)
        all_y_train_clean = np.load(data_dir+"simulated_y_train_"+str(training_size)+".npy", allow_pickle=True)
        X_train = all_X_train[trial_idx]
        y_train_clean = all_y_train_clean[trial_idx]


        # Split data in half. One half is used for CPE, the other half is used for threshold estimation.
        X_CPE, X_Con, y_CPE, y_Con = train_test_split(X_train, y_train_clean, test_size=0.5, random_state=42, stratify=y_train_clean)


        ### Training CPE Model
        vec_eta = get_vec_eta(X_CPE, y_CPE)


        if run_q:
            # fw
            ### Getting the Classifiers and Weights
            T = 5000
            clfs, weights = frank_wolfe_unc(X_Con, y_Con, vec_eta, T, n_class)
            print(len(weights))
            print(len(clfs))
            ### Evaluate Performance on CLEAN Train and Test Data
            train_conf = weight_confusion_matrix(X_train, y_train_clean, clfs, weights, n_class, vec_eta)
            test_conf = weight_confusion_matrix(X_test, y_test, clfs, weights, n_class, vec_eta)

            train_score = compute_qmean(train_conf)
            test_score = compute_qmean(test_conf)

            np.save(q_results_dir+"results_clean_"+str(training_size)+"_"+str(trial_idx)+".npy", [train_score, test_score], allow_pickle=True)


        if run_f:
            # bisection
            ### Getting the Classifiers and Weights
            T = 200
            clf = bisection(X_Con, y_Con, vec_eta, T, n_class)
            print(clf)
            ### Evaluate Performance on CLEAN Train and Test Data
            train_conf = get_confusion_matrix_from_loss_no_a(clf, X_train, y_train_clean, vec_eta, n_class)
            train_score = compute_fmeasure(train_conf)
            test_conf = get_confusion_matrix_from_loss_no_a(clf, X_test, y_test, vec_eta, n_class)
            test_score = compute_fmeasure(test_conf)

            np.save(f_results_dir+"results_clean_"+str(training_size)+"_"+str(trial_idx)+".npy", [train_score, test_score], allow_pickle=True)


        print()
        print(np.round(train_score, 3))
        print(np.round(test_score, 3))
