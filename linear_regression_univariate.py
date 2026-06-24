import numpy as np
class Linear_Regression_Univariate:

    def calc_cost(self,x,y,w,b):
        n = x.shape[0]
        cost=0
        for i in range(n):
            y_pred = w*(x[i]) + b
            cost += ((y_pred-y[i])**2)
        return (cost)/(2*n)
    
    def scale(self,x):
        x=x.astype(float)    
        x = (x-np.mean(x))/np.std(x)
        return x
    
    def calc_gradient(self,x,y,w,b):
        m = x.shape[0]
        dj_dw=0
        dj_db=0
        for i in range(m):
            dj_dw += (w*x[i]+b - y[i])*x[i]
            dj_db += (w*x[i]+b-y[i])
        return ((dj_dw/m),(dj_db/m))

    def gradient_descent(self,x,y,w,b,alpha,itr):
        for i in range(itr):
            dj_dw,dj_db = self.calc_gradient(x,y,w,b)
            w-=alpha*dj_dw
            b-=alpha*dj_db
        return w,b
    

"""
Input Format:- 
    calc_cost - x : np array of shape (n,)(1D) containing all the test cases
                y : np array of shape (n,)(1D) containing all the outputs to the test cases
                w,b : both floats, weights to the models

    scale - x : np array of shape (n,)(1D) containing all the test cases

    calc_gradient - x : np array of shape (n,)(1D) containing all the test cases
                    y : np array of shape (n,)(1D) containing all the outputs to the test cases
                    w,b : both floats, weights to the models
    
    gradient_descent - x : np array of shape (n,)(1D) containing all the test cases
                       y : np array of shape (n,)(1D) containing all the outputs to the test cases
                       w,b : both floats, weights to the models
                       alpha : a float, learning rate for the model
                       itr : integer, maximum number of times the model can run gradient descent to minimise the cost
Output Format:-
    calc_cost        - float, scalar cost value
    scale            - np array of shape (n,)(1D), normalized input
    calc_gradient    - tuple (dj_dw, dj_db), both floats
    gradient_descent - tuple (w, b), both floats, optimized weights
"""