import numpy as np 
def normalized_error_VT(y_hat,y,V_0):
        #y_hat= f_X predicted
        # y = V_T = f_X
        Normalized_Error_T = np.sqrt(1/len(y)*np.sum((y_hat-y)**2, axis=0))/V_0
        ## see formula p4 for ||f(X)||2,Q t
        ## since samples are drawn according to measure - just sum them up
        return Normalized_Error_T

def flatten_X(X):
    f = lambda x:x.T.flatten()
    return np.array([f(x) for x in X])