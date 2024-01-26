import numpy as np
from sklearn.model_selection import train_test_split
from standard_funcs.confusion_matrix import get_confusion_matrix
from standard_funcs.helpers import compute_hmean, compute_qmean, compute_gmean, compute_fmeasure

import sys
import os
import time

import tensorflow as tf
import pickle

from algorithm.models import UCIModel


noise_rates = [0.1, 0.2, 0.3, 0.4]
dir_path = "./real_data/"

total_noise_rates = 4

NUM_TRIALS = 5

np.random.seed(1337)  # for reproducibility


def build_file_name(loc, dataset, loss, noise, run):

    return (os.path.dirname(os.path.realpath(__file__)) +
            '/output/' + loc +
            dataset + '_' +
            loss + '_' +
            str(noise) + '_' +
            str(run)+".hdf5")


for loss in ['crossentropy', 'backward', 'forward', 'plug']:
    ### Data Loading
    for d in ["vehicle", "pageblocks", "satimage", "covtype", "abalone"]:
        print(d)

        data_dict = np.load(dir_path + d + "/" + d + "_data.npy", allow_pickle=True).item()
        X_data = data_dict['X']
        Y_data_clean = data_dict['Y']

        # number of classes
        n_class = len(np.unique(Y_data_clean))

        # load noisy matrices
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

            test_scores = {}
            test_scores['h'] = []
            test_scores['q'] = []
            test_scores['g'] = []
            test_scores['f'] = []

            for i in range(NUM_TRIALS):
                X_train, X_test, y_train, y_test, y_train_clean, y_test_clean = train_test_split(X_data, Y_data, Y_data_clean, test_size=0.3, random_state=i)

                # number of classes
                assert(n_class == len(np.unique(y_train)))

                # print("X_train shape: ", X_train.shape, "X_train type: ", X_train.dtype)
                # print("y_train shape: ", y_train.shape, "y_train type: ", y_train.dtype)
                # print("y_train_clean shape: ", y_train_clean.shape, "y_train_clean type: ", y_train_clean.dtype)

                # print("X_test shape: ", X_test.shape, "X_test type: ", X_test.dtype)
                # print("y_test shape: ", y_test.shape, "y_test type: ", y_test.dtype)
                # print("y_test_clean shape: ", y_test_clean.shape, "y_test_clean type: ", y_test_clean.dtype)

                # time.sleep(3)

                if loss in ['crossentropy', 'backward', 'forward']:

                    # convert class vectors to binary class matrices
                    Y_train = tf.keras.utils.to_categorical(y_train, n_class)

                    # keep track of the best model
                    model_file = build_file_name('tmp_model/', d, loss, noise_idx, i)

                    # build and compile the model
                    val_split = 0.1
                    KerasModel = UCIModel(features=X_train.shape[1], classes=n_class)
                    KerasModel.build_model(loss, noise_T.T)

                    # fit the model
                    history = KerasModel.fit_model(model_file, X_train, Y_train, validation_split=val_split)

                    # document for writing history
                    history_file = build_file_name('history/', d, loss, noise_idx, i)
                    with open(history_file, 'wb') as f:
                        pickle.dump(history, f)
                        print('History dumped at ' + str(history_file))

                    # make predictions
                    pred = KerasModel.predict(X_test)

                    # clean models, unless it is vanilla_crossentropy (to be used by plug-in)
                    if loss != 'crossentropy':
                        os.remove(model_file)
                
                if loss == "plug":
                    vanilla_file = build_file_name('tmp_model/', d, 'crossentropy', noise_idx, i)

                    if not os.path.isfile(vanilla_file):
                        ValueError('Need to train with crossentropy first !')

                    # first compile the vanilla_crossentropy model with the saved weights
                    KerasModel = UCIModel(features=X_train.shape[1], classes=n_class)
                    KerasModel.build_model('crossentropy', noise_T.T)
                    KerasModel.load_model(vanilla_file)

                    # make predictions
                    pred = KerasModel.predict_plug(X_test, noise_T_inv)

                # evaluate performance on test data
                # print("pred shape: ", pred.shape, "pred type: ", pred.dtype)
                assert(pred.shape[0] == y_test_clean.shape[0])
                test_conf = get_confusion_matrix(y_test_clean, pred, n_class)

                f_score = 1 - compute_fmeasure(test_conf)
                h_score = 1 - compute_hmean(test_conf)
                q_score = 1 - compute_qmean(test_conf)
                g_score = 1 - compute_gmean(test_conf)

                # print("f_score: ", f_score, "h_score: ", h_score, "q_score: ", q_score, "g_score: ", g_score)
                # print("test_conf: ", test_conf)
                # time.sleep(3)
                    
                test_scores["f"].append(f_score)
                test_scores["h"].append(h_score)
                test_scores["q"].append(q_score)
                test_scores["g"].append(g_score)

                
            for performance_measure in ["f", "h", "q", "g"]:
                mu_test = 1 - round(np.mean(test_scores[performance_measure]), 5)
                std_test = round(np.std(test_scores[performance_measure])/np.sqrt(len(test_scores[performance_measure])), 5)

                print(performance_measure, str(mu_test) + " (" + str(std_test) + ")")

                # np.save(dir_path+"results/"+d+"_noisy_"+str(noise_idx)+"_"+loss+"_"+performance_measure+"_results.npy", np.array([mu_test, std_test]), allow_pickle=True)
