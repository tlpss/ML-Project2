import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

from stochastic_models import MaxCallStochasticModel

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

    this function flattens as follows: x_1,0 ... x_d,0  x_1,1 ... x_d,1  ...  x_1,T ... x_d,T

    :param X: set of drivers 
    :type X: np.ndarray( N x d x T )
    :return: flattened array of drivers
    :rtype: np.ndarray(N x (d*T))
    """
    f = lambda x:x.T.flatten()
    return np.array([f(x) for x in X])

def generate_V_0(N,Delta,d,generator=MaxCallStochasticModel):
    """
    generate V_0 using dataset of specified shape

    :param N: [description]
    :type N: [type]
    :param Delta: [description]
    :type Delta: [type]
    :param d: [description]
    :type d: [type]
    :return: V_0^
    :rtype: float
    """
    s_v = generator(N,d,Delta)
    s_v.generate_samples()
    V_0 = s_v.generate_true_V(0)
    V_0 = V_0.mean()
    return V_0

def generate_train_set(N,Delta,d,generator=MaxCallStochasticModel):
    """
    generate trainset with specified shape

    :param N: [description]
    :param Delta: [description]
    :param d: [description]
    :return: X_train, y_train
    :rtype: np.ndarray, np.ndarray
    """
    s_train = generator(N,d,Delta)
    s_train.generate_samples()
    y_train = s_train.y
    X_train = flatten_X(s_train.X)
    V_0 = s_train.generate_true_V(0)
    V_0 = V_0.mean()
    print(f"V_0_train of the set = {V_0}")
    return X_train, y_train

def generate_test_set(N,Delta,d,generator=MaxCallStochasticModel):
    """
    generate trainset with specified shape

    :param N: [description]
    :param Delta: [description]
    :param d: [description]
    :return: X_train, y_train
    :rtype: np.ndarray, np.ndarray
    """
    s_test = generator(N,d,Delta)
    s_test.generate_samples()
    y_test = s_test.y
    X_test = flatten_X(s_test.X)
    V_0 = s_test.generate_true_V(0)
    V_0 = V_0.mean()
    print(f"V_0_test of the set = {V_0}")
    return X_test, y_test

def create_GPR(N_train):
    """
    creates a GPR with the standard kernel used in the project

    :param N_train: [description]
    :type N_train: [type]
    :return: gpr
    :rtype: sklearn.GPR
    """
    lambda_range = (N_train*1e-9 , N_train*1e-3)
    alpha_range = (8.3*1e-5, 0.83)
    length_scale = np.sort(1/np.sqrt((2*alpha_range[0], 2*alpha_range[1])))

    #kernel
    kernel = RBF(length_scale= (length_scale[0] + length_scale[1])/2, length_scale_bounds=length_scale) \
            + WhiteKernel(noise_level= (lambda_range[0] + lambda_range[1])/2 , noise_level_bounds=lambda_range)

    gpr = GaussianProcessRegressor(kernel, copy_X_train=False)      
    return gpr

def create_reg_kernel(lambda_):
    alpha_range = (8.3*1e-5, 0.83)
    length_scale = np.sort(1/np.sqrt((2*alpha_range[0], 2*alpha_range[1])))

    #kernel
    kernel = RBF(length_scale= (length_scale[0] + length_scale[1])/2, length_scale_bounds=length_scale) \
            + WhiteKernel(noise_level= lambda_ , noise_level_bounds="fixed") # fix lambda 

    return kernel
def memory_efficient_predict(model, X, max_size = 20000):
    """
    this functions predicts X using the model but splits of the dataset to reduce the memory requirements for the 
    kernel evaluation with the matrix of size N_train x N_test

    :param model: GPR
    :param X: set to evaluate
    :param max_size: max size to predict at once, defaults to 20000
    :return: predictions for X
    """
    assert isinstance(model, GaussianProcessRegressor)
    if X.shape[0] > max_size:
        n = int(X.shape[0]/max_size)
        print(f"split in {n} subsets for prediction")
        datasets = np.array_split(X,n)
        mu_list = []
        sigma_list = []
        for dataset in datasets:
            mu,sigma = model.predict(dataset, return_std = True)
            mu_list.append(mu)
            sigma_list.append(sigma)
        mu = np.concatenate(mu_list)
        sigma = np.concatenate(sigma_list)
        return mu, sigma
    else: 
        return model.predict(X, return_std = True)

if __name__ == "__main__":
    x = np.arange(12).reshape((1,3,4))

    print(flatten_X(x))