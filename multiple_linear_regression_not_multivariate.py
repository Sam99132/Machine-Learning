import numpy as np
import pandas as pd

class Multiple_Linear_Regression_not_multivariate:
    def scale(self,x):
        x=x.astype(float)    
        x = (x-np.mean(x,axis=0))/np.std(x,axis=0)  # axis=0 tells the mean, std functions to compute mean and std column wise,cause standadization occurs column/feature wise, not sample wise
        return x


    def calc_cost(self,X,Y,W,b):
        cost=0
        n=X.shape[0]
        m=W.shape[0]
        for i in range(n):
            y_pred = np.dot(X[i],W) + b
            err = (y_pred-Y[i])**2
            cost += err
        return cost/(2*n)

    def calc_grad(self,X,Y,W,b):
        no_of_features = W.shape[0]
        no_of_train_cases = X.shape[0]
        dj_dw=np.zeros(no_of_features)
        dj_db=0
        for i in range(no_of_train_cases):
            err = (np.dot(X[i],W) + b)- Y[i]
            for j in range(no_of_features):
                dj_dw[j] += err*X[i][j]/no_of_train_cases
            dj_db += err
        return dj_dw,dj_db/no_of_train_cases


    def gradient_descent(self,X,Y,W,b,alpha,iterations,lamb):
        no_of_features = W.shape[0]
        X = self.scale(X)
        m=X.shape[0]

        for i in range(iterations):
            dj_dw,dj_db = self.calc_grad(X,Y,W,b)

            for j in range(no_of_features):
                W[j] -= alpha*dj_dw[j] + alpha*((lamb/m)*W[j]) # lambda is used here as the regularization parameter, regularization basically penalises the features that arent important

            b -= alpha*dj_db

        return W,b


"""
Input Format:-
    scale             - x          : np array of shape (n, m), n samples and m features

    calc_cost         - X          : np array of shape (n, m), n samples and m features
                        Y          : np array of shape (n,)(1D), target values
                        W, b       : W is np array of shape (m,), b is float, model weights

    calc_grad         - X          : np array of shape (n, m), n samples and m features
                        Y          : np array of shape (n,)(1D), target values
                        W, b       : W is np array of shape (m,), b is float, model weights

    gradient_descent  - X          : np array of shape (n, m), n samples and m features
                        Y          : np array of shape (n,)(1D), target values
                        W, b       : W is np array of shape (m,), b is float, model weights
                        alpha      : float, learning rate
                        iterations : int, max iterations for gradient descent
                        lamb       : float, regularization parameter (lambda)

Output Format:-
    scale             - np array of shape (n, m), standardized feature matrix
    calc_cost         - float, scalar cost value (MSE)
    calc_grad         - tuple (dj_dw, dj_db), dj_dw is np array of shape (m,), dj_db is float
    gradient_descent  - tuple (W, b), optimized weights, W is np array of shape (m,), b is float
"""

