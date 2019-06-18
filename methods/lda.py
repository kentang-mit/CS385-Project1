##############Linear Discriminant Analysis#############
# Author: Haotian Tang                                                      # 
# E-Mail: kentang@sjtu.edu.cn                                            #
# Date: May, 2019                                                              #
#################################################

import numpy as np
import pickle
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
        self.thresh = None
    
    def fit(self, X_train, y_train):
        npos = np.sum(y_train == 1)
        nneg = np.sum(y_train == -1)
        sample_pos = X_train[y_train == 1]
        sample_neg = X_train[y_train == -1]
        cov_pos = calc_cov(sample_pos, self.verbose)
        cov_neg = calc_cov(sample_neg, self.verbose)
        mu_pos = np.mean(sample_pos, axis=0)
        mu_neg = np.mean(sample_neg, axis=0)
        
        Sw = cov_pos * npos + cov_neg * nneg
        beta = np.linalg.pinv(Sw)@(mu_pos-mu_neg)
        self.beta = beta
        
        print("Running grid search to find the decision boundary...")
        X_logits = X_train @ self.beta
        train_acc = 0
        best_thresh = 0
        for thresh in np.arange(X_logits.min(),X_logits.max(),2e-6):
            X_train_pred = (X_logits > thresh) * 2 - 1
            acc = np.sum(X_train_pred == y_train) / y_train.shape[0]
            if acc > train_acc:
                train_acc = acc
                best_thresh = thresh
        
        self.thresh = best_thresh
        f = open('lda.pkl','wb')
        pickle.dump({'beta': self.beta, 'thresh': best_thresh}, f)
        f.close()
    
    def predict(self, X_test):
        if self.beta is None:
            if os.path.exists("lda.pkl"):
                with open("lda.pkl", "rb") as f:
                    dic = pickle.load(f)
                    self.beta = dic['beta']
                    self.thresh = dic['thresh']
            else:
                print("Please first run fit function!")
                return
        
        logits = X_test@self.beta
        return (logits > self.thresh) * 2 - 1
        
    
    def score(self, X_test, y_test):
        if self.beta is None:
            if os.path.exists("lda.pkl"):
                with open("lda.pkl", "rb") as f:
                    dic = pickle.load(f)
                    self.beta = dic['beta']
                    self.thresh = dic['thresh']
            else:
                print("Please first run fit function!")
                return
        
        logits = X_test@self.beta
        pred =  (logits > self.thresh) * 2 - 1
        acc = np.sum(pred == y_test) / y_test.shape[0]
        print(acc)
        
        npos = np.sum(pred == 1)
        nneg = np.sum(pred == -1)
        sample_pos = X_test[pred == 1]
        sample_neg = X_test[pred == -1]
        cov_pos = calc_cov(sample_pos, self.verbose)
        cov_neg = calc_cov(sample_neg, self.verbose)
        mu_pos = np.mean(sample_pos, axis=0)
        mu_neg = np.mean(sample_neg, axis=0)
        Sw = cov_pos * npos + cov_neg * nneg
        Sb = (mu_pos - mu_neg).reshape(-1, 1) @ (mu_pos - mu_neg).reshape(1, -1)
        var_intra = self.beta.reshape(1,-1)@Sw@self.beta.reshape(-1,1)
        var_inter = self.beta.reshape(1,-1)@Sb@self.beta.reshape(-1,1)
        print('intra variance:', var_intra[0,0], 'inter variance:', var_inter[0,0])
        
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