import numpy as np 
def normalized_error_VT(y_hat,y,V_0):
    """
    Calculate normalized error of predictions w.r.t true values,
    where the samples are distributed accoring to the measurement Q

    :param y_hat: predictions of the model at time T  (VX_T= f_X~(X))
    :type y_hat: np.ndarray( N x 1)
    :param y: true values at time T (V_T = f(X))
    :type y: np.ndarray(N x 1)
    :param V_0: average value of the outcomes at time T, given the information at time 0 ( = MC estimator of f_X)
    :type V_0: float
    :return: normalized, relative error. E.g. error = 0.12 means that the predictions are on average 12% off wrt the true values
    :rtype: float
    """
    #y_hat= f_X predicted
    # y = V_T = f_X
    Normalized_Error_T = np.sqrt(1/len(y)*np.sum((y_hat-y)**2, axis=0))/V_0
    ## see formula p4 for ||f(X)||2,Q t
    ## since samples are drawn according to measure - just sum them up
    return Normalized_Error_T

def flatten_X(X):
    """
    Flattens the 2D (d x T) drivers into a 1D array 
    used before prediction using GPR

    this function flattens as follows: x_1,0...x_1,T,x_2,0...x_2,T...,x_d,0...x_d,T

    :param X: set of drivers 
    :type X: np.ndarray( N x d x T )
    :return: flattened array of drivers
    :rtype: np.ndarray(N x (d*T))
    """
    f = lambda x:x.T.flatten()
    return np.array([f(x) for x in X])