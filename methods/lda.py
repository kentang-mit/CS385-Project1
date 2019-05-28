##############Linear Discriminant Analysis#############
# Author: Haotian Tang                                                      # 
# E-Mail: kentang@sjtu.edu.cn                                            #
# Date: May, 2019                                                              #
#################################################

import numpy as np
from numba import jit
import time
import os
np.random.seed(1)

# helper: calculate the covariance matrix.
# using numba to accelerate the code.
@jit(nopython=True)
def calc_cov(x, verbose=True):
    # Should be better parallelized, but extremely memory-inefficient.
    """
    # N * 1 * P
    delta = x[:, None, :] - x.mean(0) 
    # N * P * 1
    delta_t = delta.transpose(0, 2, 1) 
    # N * P * P
    cov_mat = delta_t @ delta
    return cov_mat.mean(0)
    """
    print("Calculating the covariance matrix... (This might be slow.)")
    # Slow, but memory efficient version
    N, P = x.shape
    cov_mat = np.zeros((P, P))
    xmu = np.mean(x[0])
    for i in range(N):
        delta = x[i] - xmu
        cov_mat += delta.reshape(-1, 1)@delta.reshape(1, -1)
        
    return cov_mat / N
    
# helper: sigmoid
def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


# Linear Discriminant Analysis.
class LinearDiscriminantAnalysis:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.beta = None
    
    def fit(self, X_train, y_train):
        npos = np.sum(y_train == 1)
        nneg = np.sum(y_train == -1)
        sample_pos = X_train[y_train == 1]
        sample_neg = X_train[y_train == -1]
        cov_pos = calc_cov(sample_pos, self.verbose)
        cov_neg = calc_cov(sample_neg, self.verbose)
        Sw = cov_pos * npos + cov_neg * nneg
        mu_pos = np.mean(sample_pos, axis=0)
        mu_neg = np.mean(sample_neg, axis=0)
        beta = np.linalg.pinv(Sw)@(mu_pos-mu_neg)
        self.beta = beta
        np.save("lda.npy", beta)
    
    def predict(self, X_test):
        if self.beta is None:
            if os.path.exists("lda.npy"):
                self.beta = np.load("lda.npy")
            else:
                print("Please first run fit function!")
                return
        
        logits = X_test@self.beta
        return (logits > 0) * 2 - 1
        
    
    def score(self, X_test, y_test):
        if self.beta is None:
            if os.path.exists("lda.npy"):
                self.beta = np.load("lda.npy")
            else:
                print("Please first run fit function!")
                return
        
        logits = X_test@self.beta
        pred =  (logits > 0) * 2 - 1
        acc = np.sum(pred == y_test) / y_test.shape[0]
        print(acc)
        return acc
 

    def decision_function(self, X_test):
        if self.beta is None:
            if os.path.exists("lda.npy"):
                self.beta = np.load("lda.npy")
            else:
                print("Please first run fit function!")
                return
        
        logits = X_test@self.beta
        probs = sigmoid(logits)
        return [probs]


if __name__ == '__main__':
    # Unit test.
    X_train = np.random.randn(1000, 900)
    y_train = np.random.choice(2, 1000) * 2 - 1
    lr = LinearDiscriminantAnalysis()
    lr.fit(X_train, y_train)
    print(np.sum((lr.predict(X_train)>0)*2-1 == y_train))