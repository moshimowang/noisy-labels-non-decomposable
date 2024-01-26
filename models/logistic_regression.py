import numpy as np
from sklearn.linear_model import LogisticRegressionCV
import pickle

def get_vec_eta(X_train, y_train):
    lr = LogisticRegressionCV(
        solver="newton-cg",
        max_iter=100000,
        cv=2,
        tol=1e-3,
        multi_class="auto"
            ).fit(X_train, y_train)

    def vec_eta(X):
        return lr.predict_proba(X)

    return vec_eta


def get_vec_eta_2(X_train, y_train):
    lr = LogisticRegressionCV(
        solver="newton-cg",
        max_iter=100000,
        cv=3,
        tol=1e-3,
        multi_class="multinomial"
            ).fit(X_train, y_train)

    def vec_eta(X):
        return lr.predict_proba(X)

    return vec_eta
