import numpy as np
import pandas as pd


class Logistic_Regression: 

    def sigmoid(self,z):
        g_z = 1/(1+(np.exp(-z)))
        return g_z

    def scale(self,x):
        x=x.astype(float)    
        x = (x-np.mean(x,axis=0))/np.std(x,axis=0)  # axis=0 tells the mean, std functions to compute mean and std column wise,cause standadization occurs column/feature wise, not sample wise
        return x

    def cost(self,X,Y,w,b):
        cost=0
        m=X.shape[0]
        for i in range(m):
            y = np.dot(X[i],w)+b
            y_pred = self.sigmoid(y)
            cost += (-Y[i]*(np.log(y_pred)) - (1-Y[i])*(np.log(1-y_pred)))
        return cost/m

    def calc_grad(self,X,Y,w,b):
        no_feats = w.shape[0]
        no_cases=X.shape[0]
        grads_w = np.zeros(no_feats)
        grad_b=0
        for i in range(no_cases):
            for j in range(no_feats):
                grads_w[j]+= (self.sigmoid(np.dot(X[i],w)+b) - Y[i])*(X[i][j])
            grad_b += self.sigmoid(np.dot(X[i],w)+b) - Y[i]
        return grads_w/no_cases,grad_b/no_cases

    def grad_desc(self,X,Y,w,b,alpha,iter,lamb):
        m=X.shape[0]
        for i in range(iter):
            grads_w,grad_b = self.calc_grad(X,Y,w,b)
            w -= alpha*(grads_w + (lamb/m)*w)
            b-= alpha*grad_b
        return w,b



"""
Input Format:-
    sigmoid   - z     : scalar or np array, raw model output

    scale     - x     : np array of shape (m, n), m samples and n features

    cost      - X     : np array of shape (m, n), m samples and n features
                Y     : np array of shape (m,)(1D), binary labels (0 or 1)
                w, b  : w is np array of shape (n,), b is float, model weights

    calc_grad - X     : np array of shape (m, n), m samples and n features
                Y     : np array of shape (m,)(1D), binary labels (0 or 1)
                w, b  : w is np array of shape (n,), b is float, model weights

    grad_desc - X     : np array of shape (m, n), m samples and n features
                Y     : np array of shape (m,)(1D), binary labels (0 or 1)
                w, b  : w is np array of shape (n,), b is float, model weights
                alpha : float, learning rate
                iter  : int, max iterations for gradient descent
                lamb  : float, regularization parameter (lambda)

Output Format:-
    sigmoid   - scalar or np array, values squashed between 0 and 1
    scale     - np array of shape (m, n), standardized feature matrix
    cost      - float, scalar cost value (binary cross-entropy with L2 regularization)
    calc_grad - tuple (grads_w, grad_b), grads_w is np array of shape (n,), grad_b is float
    grad_desc - tuple (w, b), optimized weights, w is np array of shape (n,), b is float
"""