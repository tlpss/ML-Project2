import os, psutil
import numpy as np
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from mpi.utils import generate_bagging_train_indices, train_and_evaluate
from stochastic_models import *
from aggregating.utils import flatten_X

class ProfileGPR:
    def __init__(self):
        self.N_train = 5000
        self.N_test  = 1000000
        self.d = 1
        self.T = 2


    def profile(self):

        lambda_range = (self.N_train*1e-9 , self.N_train*1e-3)
        alpha_range = (8.3*1e-5, 0.83)
        length_scale = np.sort(1/np.sqrt((2*alpha_range[0], 2*alpha_range[1])))

        #kernel
        kernel = RBF(length_scale= (length_scale[0] + length_scale[1])/2, length_scale_bounds=length_scale) \
            + WhiteKernel(noise_level= (lambda_range[0] + lambda_range[1])/2 , noise_level_bounds=lambda_range)

        gpr = GaussianProcessRegressor(kernel,copy_X_train=False)

        s_train = MaxCallStochasticModel(self.N_train,self.d,[1/12,11/12])
        s_train.generate_samples()

        s_test = MaxCallStochasticModel(self.N_test,self.d,[1/12,11/12])
        s_test.generate_samples()

        X_train, y_train = flatten_X(s_train.X), s_train.y
        X_test = flatten_X(s_test.X)

        print(f"memory before training = {self.get_mem_usage()}GB")

        gpr.fit(X_train,y_train)

        print(f"memory after training = {self.get_mem_usage()}GB")

        y = gpr.predict(X_test)

        print(f"memory after inference  = {self.get_mem_usage()}GB")


    

    def get_mem_usage(self):
        process = psutil.Process(os.getpid())
        return (process.memory_info().rss)  / (1024)**3 # GB

    def predict_train_mem(self):
        return (self.N_train**2 * 8)/(1024)**3 # N**2 * 8GB/entry

    def predict_inference_process_mem(self):
        return self.N_train * self.N_test * 8 #sklearn allocates (train,test) array during inference to find distance between all points --> limiting memory factor..

if __name__ == "__main__":
    profiler = ProfileGPR()
    print(f" predicted training mem {profiler.predict_train_mem()}")
    print(f" predicted inference temp mem {profiler.predict_inference_process_mem()}")

    profiler.profile()