import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import unittest

from mpi.utils import generate_bagging_train_indices, train_and_evaluate
from stochastic_models import *
from aggregating.utils import flatten_X

class TestMPIUtils(unittest.TestCase):
    def setUp(self):
        self.N_train = 10
        self.d = 3
        T = 2
        s_train = MaxCallStochasticModel(self.N_train,self.d,[1/12,11/12])
        s_train.generate_samples()
        self.y_train = s_train.y
        X_train = s_train.X
        self.X_train  = flatten_X(X_train)


    def test_bagging_split(self):
        indices = generate_bagging_train_indices(self.N_train,0.5,8)
        self.assertTrue(len(indices)==8)
        self.assertTrue(indices[3].shape == (5,))

    def test_train_and_evaluate(self):
        lambda_range = (self.N_train*1e-9 , self.N_train*1e-3)
        alpha_range = (8.3*1e-5, 0.83)
        length_scale = np.sort(1/np.sqrt((2*alpha_range[0], 2*alpha_range[1])))

        #kernel
        kernel = RBF(length_scale= (length_scale[0] + length_scale[1])/2, length_scale_bounds=length_scale) \
            + WhiteKernel(noise_level= (lambda_range[0] + lambda_range[1])/2 , noise_level_bounds=lambda_range)

        gpr = GaussianProcessRegressor(kernel,copy_X_train=False)

        s_test = MaxCallStochasticModel(20,self.d,[1/12,11/12])
        s_test.generate_samples()

        s_test2 = MaxCallStochasticModel(20,self.d,[1/12,11/12])
        s_test2.generate_samples()

        X_test = [flatten_X(s_test.X),flatten_X(s_test2.X)]

        mu_list, sigma_list  = train_and_evaluate(gpr,self.X_train,self.y_train,X_test)
        self.assertEqual(len(mu_list),len(X_test))
        self.assertEqual(len(sigma_list),len(X_test))
        self.assertEqual(mu_list[0].shape,(X_test[0].shape[0],))
        self.assertEqual(sigma_list[0].shape,(X_test[0].shape[0],))
        self.assertTrue((mu_list[0]-mu_list[1]).all())



