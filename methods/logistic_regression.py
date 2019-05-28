#############Logistic Regression with SGD#############
# Author: Haotian Tang                                                      # 
# E-Mail: kentang@sjtu.edu.cn                                            #
# Date: May, 2019                                                              #
#################################################

import numpy as np
import time
import os
np.random.seed(1)

# helper function: sigmoid
def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

# Logistic regression.
# C: regularization term.
class LogisticRegression:
    def __init__(self, C=0.001, epochs=100, batch_size=32, lr=1e-2, langevin=1e-4, solver='langevin', verbose=True):
        self.C = C
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.solver = solver
        self.beta = None
        self.verbose = verbose
        self.langevin = langevin
    
    
    def fit(self, X_train_, y_train_):
        # Add the bias term
        X_train = np.hstack([np.ones((X_train_.shape[0], 1)), X_train_])
        y_train = y_train_.reshape(-1, 1)
        # dimensionality
        data_dim = X_train.shape[1]
        # The parameter
        beta = np.random.randn(data_dim)
        # Objective L =  -\sigma_{i=1}^N log(1 + exp(-y_iX_i\beta))
        # Gradient \frac{\patial L}{\partial \beta} = \sigma_{i=1}^N sigmoid(-y_iX_i\beta) y_iX_i^T
        # = y*X^T sigmoid(-y*X\beta)
        
        if self.solver == 'sgd' or self.solver == 'langevin':
            batch_size = self.batch_size
            num_batchs = (X_train.shape[0] + batch_size - 1) // batch_size
             
            for epoch in range(self.epochs):
                if self.verbose:
                    print("Training logistic regression epoch %d/%d..."%(epoch+1, self.epochs))
                
                shuffled_idx = np.random.permutation(X_train.shape[0])
                X_train = X_train[shuffled_idx, ...]
                y_train = y_train[shuffled_idx]
                for batch_idx in range(num_batchs):
                    batch_start = batch_idx * batch_size
                    batch_end = min(X_train.shape[0], (batch_idx + 1) * batch_size)
                    X = X_train[batch_start:batch_end, ...]
                    y = y_train[batch_start:batch_end]
                    grad = (y.T*X.T)@sigmoid((-y*X)@beta) - self.C * beta
                    # Gradient ascent.
                    if not self.solver == 'langevin':
                        beta = beta + self.lr * grad
                    else:
                        beta = beta + self.lr * grad + self.langevin * np.random.randn(*beta.shape)
        else:
            raise NotImplementedError
        
        self.beta = beta
        np.save("logistic.npy", beta)
    
    def predict(self, X_test_):
        if self.beta is None:
            if os.path.exists("logistic.npy"):
                self.beta = np.load("logistic.npy")
            else:
                print("Please first run fit function!")
                return
        
        X_test = np.hstack([np.ones((X_test_.shape[0], 1)), X_test_])
        logits = X_test@self.beta 
        return (logits >= 0) * 2 - 1
    
    def score(self, X_test_, y_test):
        if self.beta is None:
            if os.path.exists("logistic.npy"):
                self.beta = np.load("logistic.npy")
            else:
                print("Please first run fit function!")
                return
        
        X_test = np.hstack([np.ones((X_test_.shape[0], 1)), X_test_])
        logits = X_test@self.beta
        pred = (logits >= 0) * 2 - 1
        acc = np.sum(pred==y_test) / y_test.shape[0]
        print(acc)
        return acc
    
    def predict_proba(self, X_test_):
        if self.beta is None:
            if os.path.exists("logistic.npy"):
                self.beta = np.load("logistic.npy")
            else:
                print("Please first run fit function!")
                return
        X_test = np.hstack([np.ones((X_test_.shape[0], 1)), X_test_])
        prob = sigmoid(X_test@self.beta)
        return [[1-prob, prob]]

if __name__ == '__main__':
    # Unit test.
    X_train = np.random.randn(1000, 900)
    y_train = np.random.choice(2, 1000) * 2 - 1
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    print(np.sum((lr.predict(X_train)>0)*2-1 == y_train))