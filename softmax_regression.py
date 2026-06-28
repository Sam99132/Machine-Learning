import numpy as np

class Softmax_Regression:
    def e_z(self,z):
        return np.exp(z)

    def scale(self,x):
        x=x.astype(float)    
        x = (x-np.mean(x,axis=0))/np.std(x,axis=0)  # axis=0 tells the mean, std functions to compute mean and std column wise,cause standadization occurs column/feature wise, not sample wise
        return x

    def loss(self,X,Y,W,b):
        z = (W.T@X.T).T +b
        z -= np.max(z, axis=1, keepdims=True)
        logits2 = np.exp(z)
        a = logits2 / np.sum(logits2, axis=1, keepdims=True)
        return -np.sum(Y * np.log(np.clip(a, 1e-15, 1))) / len(X)

    def gradient_descent(self,X,Y,W,b,alpha,iters):
        for i in range(iters):
            z = (W.T@X.T).T +b
            z -= np.max(z, axis=1, keepdims=True)
            logits2 = np.exp(z)
            a = logits2 / np.sum(logits2, axis=1, keepdims=True)
            grads_w = (a-Y).T @ X 
            W-=alpha*grads_w.T
            b -= alpha*np.sum(a-Y,axis=0)
        return W,b
    



"""
Input Format:-
    scale             - x      : np array of shape (n, m), n samples and m features

    loss              - X      : np array of shape (n, m), n samples and m features
                        Y      : np array of shape (n, C), one-hot encoded labels, n samples and C classes
                        w, b   : w is np array of shape (C, m), b is np array of shape (C,), C = no of classes

    calc_grad         - X      : np array of shape (n, m), n samples and m features
                        Y      : np array of shape (n, C), one-hot encoded labels, n samples and C classes
                        w, b   : w is np array of shape (C, m), b is np array of shape (C,), C = no of classes

    gradient_descent  - X      : np array of shape (n, m), n samples and m features
                        Y      : np array of shape (n, C), one-hot encoded labels, n samples and C classes
                        w, b   : w is np array of shape (C, m), b is np array of shape (C,), C = no of classes
                        alpha  : float, learning rate
                        iters  : int, max iterations for gradient descent

Output Format:-
    scale             - np array of shape (n, m), standardized feature matrix
    loss              - float, scalar total cross-entropy loss
    calc_grad         - tuple (grads_w, grads_b), grads_w is np array of shape (C, m), grads_b is np array of shape (C,)
    gradient_descent  - tuple (w, b), optimized weights, w is np array of shape (C, m), b is np array of shape (C,)
"""
    
